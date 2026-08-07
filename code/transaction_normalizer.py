"""Stable, versioned transaction normalization boundary for TaxSage."""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

from pipeline.bank_classifier import ClassifiedTransaction, RawTransaction, classify_with_rules

RULESET_VERSION = "2026.07.17.1"
MAX_NORMALIZE_BATCH = 500


class NormalizationError(ValueError):
    """Raised when a transaction cannot satisfy the normalization contract."""


def _first_nonempty(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _parse_timestamp(value: Any) -> tuple[date, str]:
    text = str(value or "").strip()
    if not text:
        raise NormalizationError("transaction date is required")

    try:
        parsed_date = date.fromisoformat(text)
        return parsed_date, parsed_date.isoformat()
    except ValueError:
        pass

    try:
        parsed_datetime = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed_datetime.date(), parsed_datetime.isoformat()
    except ValueError:
        pass

    for pattern in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            parsed_date = datetime.strptime(text, pattern).date()
            return parsed_date, parsed_date.isoformat()
        except ValueError:
            continue

    raise NormalizationError("transaction date is invalid")


def _detect_channel(narration: str) -> str:
    upper = narration.upper()
    for channel in ("UPI", "NEFT", "IMPS", "RTGS", "ATM", "NACH", "ECS"):
        if re.search(rf"\b{channel}\b", upper):
            return channel
    if re.search(r"\b(?:POS|ECOM|E-COM)\b", upper):
        return "POS"
    if re.search(r"\b(?:CARD|VISA|MASTERCARD|RUPAY)\b", upper):
        return "CARD"
    if re.search(r"\b(?:CHEQUE|CHQ)\b", upper):
        return "CHEQUE"
    if re.search(r"\b(?:INTERNET\s*BANKING|NETBANKING|I-BANK)\b", upper):
        return "INTERNET_BANKING"
    if re.search(r"\b(?:MOBILE\s*BANKING|MOB?BANK|M-BANK)\b", upper):
        return "MOBILE_BANKING"
    return "OTHER"


def _detect_reversal(narration: str) -> bool:
    return bool(
        re.search(r"\b(?:REVERSAL|REVERSED|REFUND|RVSL|CHARGEBACK)\b", narration, re.IGNORECASE)
    )


def _detect_partial(narration: str) -> bool:
    return bool(
        re.search(
            r"\b(?:PARTIAL|SPLIT|PART\s+\d+\s+OF\s+\d+)\b",
            narration,
            re.IGNORECASE,
        )
    )


def detect_transaction_metadata(narration: str) -> dict[str, Any]:
    """Derive non-classifying metadata without database reads or writes."""
    return {
        "channel": _detect_channel(narration),
        "is_reversal": _detect_reversal(narration),
        "is_partial": _detect_partial(narration),
    }


def _classification_path(classified: ClassifiedTransaction) -> str:
    rationale = classified.rationale.lower()
    if rationale.startswith("merchant db:"):
        return "merchant_db"
    if rationale.startswith("known upi merchant evidence:"):
        return "merchant_match"
    if rationale.startswith("upi narration evidence:"):
        return "narration_purpose"
    if rationale.startswith("strong personal-transfer evidence"):
        return "personal_transfer"
    if rationale.startswith("matched rule:"):
        return "regex_rule"
    if rationale.startswith("manual override"):
        return "manual_override"
    return "unclassified"


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.85:
        return "HIGH"
    if confidence >= 0.60:
        return "MEDIUM"
    return "LOW"


def serialize_classified_transaction(
    classified: ClassifiedTransaction,
    *,
    account: str = "default",
    description: str | None = None,
) -> dict[str, Any]:
    """Serialize an existing classification with additive normalization metadata."""
    narration = classified.raw.description if description is None else description
    confidence = min(1.0, max(0.0, float(classified.confidence)))
    rationale = classified.rationale or "No reliable classification evidence"
    path = _classification_path(classified)
    raw_date = classified.raw.date
    raw_date_text = str(raw_date)
    normalized_date = (
        ""
        if raw_date_text == "NaT"
        else raw_date.isoformat() if hasattr(raw_date, "isoformat") else raw_date_text
    )
    return {
        "date": normalized_date,
        "description": narration,
        "amount": float(classified.raw.amount),
        "type": classified.raw.type,
        "category": classified.category or "unclassified",
        "confidence": confidence,
        "is_income": bool(classified.is_income),
        "is_expense": bool(classified.is_expense),
        "counterparty": classified.counterparty,
        "rationale": rationale,
        "account": account or "default",
        "channel": _detect_channel(classified.raw.description),
        "is_reversal": _detect_reversal(classified.raw.description),
        "is_partial": _detect_partial(classified.raw.description),
        "classification_path": path,
        "ruleset_version": RULESET_VERSION,
        "explain": {"path": path, "rationale": rationale},
    }


def normalize_transaction(
    record: dict[str, Any],
    *,
    index: int = 0,
    require_timestamp: bool = True,
) -> dict[str, Any]:
    """Normalize one validated transaction record into the stable TaxSage schema."""
    if not isinstance(record, dict):
        raise NormalizationError("transaction must be an object")

    narration_value = _first_nonempty(record, "raw", "description")
    narration = str(narration_value or "").strip()
    if not narration:
        raise NormalizationError("transaction narration is required")

    amount_value = record.get("amount")
    if isinstance(amount_value, bool):
        raise NormalizationError("transaction amount must be finite")
    try:
        amount = float(amount_value)
    except (TypeError, ValueError) as error:
        raise NormalizationError("transaction amount must be finite") from error
    if not math.isfinite(amount):
        raise NormalizationError("transaction amount must be finite")
    if amount < 0:
        raise NormalizationError("transaction amount must be non-negative")

    transaction_type = str(record.get("type", "")).strip().lower()
    if transaction_type not in {"credit", "debit"}:
        raise NormalizationError("transaction type must be credit or debit")

    timestamp_value = _first_nonempty(record, "timestamp", "date")
    if (timestamp_value is None or str(timestamp_value).strip() == "NaT") and not require_timestamp:
        raw_date, normalized_timestamp = date.min, ""
    else:
        raw_date, normalized_timestamp = _parse_timestamp(timestamp_value)
    raw = RawTransaction(
        date=raw_date,
        description=narration,
        type=transaction_type,
        amount=amount,
    )
    classified = classify_with_rules(raw, learn_merchants=False)
    if classified is None:
        classified = ClassifiedTransaction(
            raw=raw,
            category="unclassified",
            confidence=0.30,
            rationale="No reliable classification evidence",
            is_income=False,
            is_expense=transaction_type == "debit",
        )

    confidence = min(1.0, max(0.0, float(classified.confidence)))
    rationale = classified.rationale or "No reliable classification evidence"
    return {
        "id": str(record.get("id", index)),
        "raw": narration,
        "merchant": classified.counterparty,
        "category": classified.category or "unclassified",
        "transaction_type": transaction_type.upper(),
        "channel": _detect_channel(narration),
        "amount": amount,
        "normalized_timestamp": normalized_timestamp,
        "is_reversal": _detect_reversal(narration),
        "is_partial": _detect_partial(narration),
        "confidence": confidence,
        "confidence_level": _confidence_level(confidence),
        "is_income": bool(classified.is_income),
        "is_expense": bool(classified.is_expense),
        "explain": {
            "path": _classification_path(classified),
            "rationale": rationale,
        },
        "ruleset_version": RULESET_VERSION,
    }
