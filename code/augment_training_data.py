#!/usr/bin/env python3
"""Build privacy-safe local Qwen training rows from bank statements.

The generated dataset stays under ``autotaxfiler/data`` (gitignored). Source
paths, taxpayer identities, and transaction reference IDs are not persisted in
new rows or passed to the fine-tuning formatter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from .bank_classifier import RawTransaction, _parse_statement
    from .company_inference import infer_company_name
    from .pii_shield import mask_pii
    from .training_schema import NON_INCOME_CATEGORIES, SUPPORTED_CATEGORIES
except ImportError:
    from pipeline.bank_classifier import RawTransaction, _parse_statement
    from pipeline.company_inference import infer_company_name
    from pipeline.pii_shield import mask_pii
    from pipeline.training_schema import NON_INCOME_CATEGORIES, SUPPORTED_CATEGORIES


DEFAULT_DATA_ROOT = PACKAGE_ROOT.parent / "data"
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "training_data.json"
_STATEMENT_SUFFIXES = {".csv", ".pdf", ".xls", ".xlsx"}
_PROVISIONAL_CATEGORIES = {
    "expense_uncategorized",
    "transfer",
    "unclassified",
    "unclassified_credit",
}
_PERSONAL_CATEGORIES = {
    "family", "friends", "personal_transfer", "rental", "staff_salary", "transfer"
}
_LONG_REFERENCE_TOKEN = re.compile(
    r"(?<![A-Z0-9])(?=[A-Z0-9]*\d)[A-Z0-9]{8,}(?![A-Z0-9])",
    re.IGNORECASE,
)
_LONG_DIGIT_RUN = re.compile(r"\d{6,}")
_CONTEXT_REFERENCE_PATTERNS = (
    (re.compile(r"\b(CAM/)[A-Z0-9-]{4,}", re.IGNORECASE), r"\1<ID>"),
    (re.compile(r"(?<=/)\d{4,}(?=/|$)"), "<ID>"),
    (re.compile(r"\bWAR_NO:\s*[A-Z0-9-]{4,}", re.IGNORECASE), "WAR_NO: <ID>"),
)
_VPA = re.compile(r"([A-Z0-9._-]{2,})@([A-Z]{2,20})", re.IGNORECASE)
_SAFE_PERSONAL_SEGMENT_WORDS = {
    "ACH", "ATM", "BANK", "BILL", "BOOK", "BROADBAND", "CAB", "CAFE",
    "CAPITAL", "DEPOSIT", "FUND", "INCOME", "MANDATE", "MOBILE", "MUTUAL", "NACH",
    "PETROL", "PREMIUM", "PUMP", "RECHARGE", "REFUND", "SIP", "SUBSCRIPTION",
    "CARD", "CASH", "CHARITY", "COLLEGE", "CONSULTING", "CREDIT", "DEBIT",
    "DIVIDEND", "DONATION", "EDUCATION", "ELECTRICITY", "EMI", "ENTERTAINMENT",
    "FEE", "FITNESS", "FLIGHT", "FOOD", "GAS", "GROCERY", "HDFC", "HOSPITAL",
    "HOTEL", "ICICI", "IDFC", "IMPS", "INSURANCE", "INTEREST", "INVOICE",
    "KOTAK", "LOAN", "MEDICAL", "NEFT", "PAYMENT", "PHARMACY", "POS", "RENT",
    "RESTAURANT", "REVERSAL", "RTGS", "SALARY", "SBI", "SCHOOL", "SHOPPING",
    "STORE", "TAX", "TRADING", "TRANSFER", "TRAVEL", "TUITION", "UPI", "VEHICLE",
    "WATER", "WDL",
}


@dataclass
class AugmentationStats:
    discovered_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    parsed_transactions: int = 0
    existing_matches: int = 0
    low_confidence: int = 0
    provisional_labels: int = 0
    unsupported_categories: int = 0
    duplicate_instances: int = 0
    conflicting_keys: int = 0
    added_rows: int = 0
    company_labels: int = 0
    failure_types: dict[str, int] = field(default_factory=dict)


def normalize_description(description: str) -> str:
    """Return a stable uppercase transaction description."""
    return re.sub(r"\s+", " ", str(description)).strip().upper()


def sanitize_training_description(
    description: str,
    *,
    category: str,
    company_name: str | None = None,
) -> str:
    """Remove identifying/reference data while preserving merchant evidence."""
    sanitized = normalize_description(str(description))
    for pattern, replacement in _CONTEXT_REFERENCE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    sanitized = _LONG_REFERENCE_TOKEN.sub("<ID>", sanitized)
    sanitized = _LONG_DIGIT_RUN.sub("<ID>", sanitized)
    sanitized = normalize_description(mask_pii(sanitized))

    company_tokens = re.findall(r"[A-Z]+", normalize_description(company_name or ""))
    company_words = set(company_tokens)
    company_label = " ".join(company_tokens)
    company_identity = "".join(sorted(company_words))
    allowed_words = _SAFE_PERSONAL_SEGMENT_WORDS | company_words
    segments = []
    for segment in sanitized.split("/"):
        segment = segment.strip()
        if not segment:
            segments.append(segment)
            continue
        canonical_vpa = _VPA.fullmatch(segment)
        if (
            canonical_vpa
            and company_identity
            and canonical_vpa.group(1).upper() == company_identity
        ):
            segments.append(segment)
            continue
        compact_segment = re.sub(r"[^A-Z]", "", segment)
        if company_words and any(word in compact_segment for word in company_words):
            purpose_words = [
                word
                for word in re.findall(r"[A-Z]+", segment)
                if word in _SAFE_PERSONAL_SEGMENT_WORDS and word not in company_words
            ]
            segments.append(" ".join([company_label, *purpose_words]))
            continue
        placeholder_vpa = re.fullmatch(r"<PERSON>@([A-Z]{2,20})", segment)
        if placeholder_vpa:
            segments.append(segment)
            continue
        vpa = _VPA.search(segment)
        if vpa:
            identity = company_identity or "<PERSON>"
            segments.append(f"{identity}@{vpa.group(2).upper()}")
            continue
        visible_text = re.sub(r"<[^>]+>", " ", segment)
        words = set(re.findall(r"[A-Z]+", visible_text))
        if not words:
            segments.append("<ID>" if "<ID>" in segment else segment)
        else:
            safe_words = [
                word
                for word in re.findall(r"[A-Z]+", visible_text)
                if word in allowed_words
            ]
            segments.append(" ".join(safe_words) if safe_words else "<PERSON>")
    return "/".join(segments)


class UnsupportedStatementError(ValueError):
    """Raised when a discovered statement format has no safe parser."""


_JASPER_EXCEL_HEADERS = {
    "s no.", "value date", "transaction date", "cheque number",
    "transaction remarks", "withdrawal amount(inr)", "deposit amount(inr)",
    "balance(inr)",
}


def _has_supported_excel_schema(path: Path) -> bool:
    import pandas as pd

    dataframe = pd.read_excel(path, header=None)
    for row_index in range(min(25, len(dataframe))):
        values = {
            str(value).strip().lower()
            for value in dataframe.iloc[row_index].tolist()
            if pd.notna(value)
        }
        if _JASPER_EXCEL_HEADERS.issubset(values):
            return True
    return False


def parse_training_statement(path_string: str) -> list[RawTransaction]:
    """Parse only statement layouts whose debit/credit semantics are known."""
    path = Path(path_string)
    # PDF parsing is supported via pymupdf in bank_classifier._parse_pdf
    # Allow PDFs through — the existing _parse_statement handles them
    if path.suffix.lower() in {".xls", ".xlsx"} and not _has_supported_excel_schema(path):
        raise UnsupportedStatementError("unsupported Excel statement schema")
    return _parse_statement(str(path))


def discover_statement_files(data_root: Path) -> list[Path]:
    """Find unique likely bank statements without exposing their paths."""
    candidates = []
    if not data_root.exists():
        return candidates
    for path in data_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _STATEMENT_SUFFIXES:
            continue
        lowered = str(path).lower()
        if not any(token in lowered for token in ("bank", "statement", "passbook", "pass book", "account")):
            continue
        candidates.append(path)

    unique: dict[str, Path] = {}
    for path in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        unique.setdefault(digest, path)
    return [unique[digest] for digest in sorted(unique)]


def _default_classifier():
    from pipeline.classifier import ClassificationPipeline
    from pipeline.classifier.stages import (
        CatchAllStage,
        DescriptionRuleStage,
        MerchantDBStage,
        RegexRuleStage,
        UPIHeuristicStage,
    )

    return ClassificationPipeline([
        MerchantDBStage(),
        UPIHeuristicStage(),
        DescriptionRuleStage(),
        RegexRuleStage(),
        CatchAllStage(),
    ])


def _existing_key(row: dict) -> tuple[str, str]:
    company_name = row.get("company_name") or row.get("merchant") or row.get("counterparty")
    return (
        sanitize_training_description(
            row.get("description", ""),
            category=row.get("category", "unclassified"),
            company_name=company_name,
        ),
        row.get("type", ""),
    )


def augment_rows(
    existing_rows: list[dict],
    statement_paths: Iterable[str | Path],
    *,
    parse_statement: Callable[[str], list[RawTransaction]] = parse_training_statement,
    classifier=None,
    min_confidence: float = 0.85,
) -> tuple[list[dict], AugmentationStats]:
    """Extract unique, sanitized, high-confidence labels not already present."""
    classifier = classifier or _default_classifier()
    paths = list(statement_paths)
    stats = AugmentationStats(discovered_files=len(paths))
    existing_keys = {
        key
        for row in existing_rows
        if (
            "<PERSON>" not in (key := _existing_key(row))[0]
            or row.get("path") == "statement_augmentation"
        )
    }
    existing_raw_keys = {
        (normalize_description(row.get("description", "")), row.get("type", ""))
        for row in existing_rows
    }
    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    failures: Counter[str] = Counter()

    for path in paths:
        try:
            transactions = parse_statement(str(path))
        except Exception as error:
            stats.failed_files += 1
            failures[type(error).__name__] += 1
            continue

        stats.parsed_files += 1
        stats.parsed_transactions += len(transactions)
        for transaction in transactions:
            raw_key = (normalize_description(transaction.description), transaction.type)
            if raw_key in existing_raw_keys:
                stats.existing_matches += 1
                continue
            result = classifier.classify(transaction, learn=False)
            if result is None or result.confidence < min_confidence:
                stats.low_confidence += 1
                continue
            if result.category not in SUPPORTED_CATEGORIES:
                stats.unsupported_categories += 1
                continue
            if result.category in _PROVISIONAL_CATEGORIES:
                stats.provisional_labels += 1
                continue

            company_name = infer_company_name(
                transaction.description,
                category=result.category,
                explicit_name=result.counterparty,
            )
            description = sanitize_training_description(
                transaction.description,
                category=result.category,
                company_name=company_name,
            )
            key = (description, transaction.type)
            if not description:
                continue
            if key in existing_keys:
                stats.existing_matches += 1
                continue

            candidates[key].append({
                "description": description,
                "category": result.category,
                "type": transaction.type,
                "is_income": (
                    False
                    if result.category in NON_INCOME_CATEGORIES
                    else bool(result.is_income)
                ),
                "company_name": company_name,
                "confidence": round(float(result.confidence), 4),
                "path": "statement_augmentation",
            })

    added = []
    for key in sorted(candidates):
        rows = candidates[key]
        labels = {(row["category"], row["type"]) for row in rows}
        if len(labels) != 1:
            stats.conflicting_keys += 1
            continue
        stats.duplicate_instances += max(0, len(rows) - 1)
        added.append(max(rows, key=lambda row: row["confidence"]))

    stats.failure_types = dict(sorted(failures.items()))
    stats.added_rows = len(added)
    stats.company_labels = sum(row["company_name"] is not None for row in added)
    return added, stats


def _atomic_write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(*, data_root: Path, output: Path, dry_run: bool = False) -> AugmentationStats:
    existing_rows = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    statements = discover_statement_files(data_root)
    added_rows, stats = augment_rows(existing_rows, statements)
    if not dry_run:
        _atomic_write_json(output, [*existing_rows, *added_rows])
    print(json.dumps(asdict(stats), sort_keys=True))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    main(data_root=arguments.data_root, output=arguments.output, dry_run=arguments.dry_run)
