"""Conservative company-name labels for transaction model training."""

from __future__ import annotations

import re
from typing import Optional

from pipeline.merchant_classifier import (
    CURATED_TRANSACTION_MARKERS,
    HANDLE_CATEGORY_MAP,
    MERCHANT_ALIASES,
    classify_upi_merchant,
    extract_upi_handle,
    get_merchant,
)

_PERSONAL_CATEGORIES = {
    "personal_transfer",
    "friends",
    "family",
    "staff_salary",
    "rental",
    "transfer",
    "cash_withdrawal",
}
_GENERIC_NAMES = {
    "",
    "unknown upi counterparty",
    "upi transfer",
    "payment",
    "transfer",
    "unknown",
}


def _name_key(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


_CANONICAL_COMPANIES = {
    _name_key(company[0]): company[0]
    for company in (
        *MERCHANT_ALIASES.values(),
        *CURATED_TRANSACTION_MARKERS.values(),
        *HANDLE_CATEGORY_MAP.values(),
    )
}
_CANONICAL_ALIASES = {
    "indian cle": "Indian Clearing Corporation",
    "iccl zerodha credit": "Indian Clearing Corporation",
    "iccl zerod": "Indian Clearing Corporation",
    "zerodha br": "Zerodha",
    "zerodha deposit": "Zerodha",
}


def _clean_company_name(value: object) -> Optional[str]:
    name = " ".join(str(value or "").split()).strip(" -/|")[:100]
    if name.lower() in _GENERIC_NAMES:
        return None
    return name or None


def _canonical_company_name(value: object) -> Optional[str]:
    cleaned = _clean_company_name(value)
    if not cleaned:
        return None
    key = _name_key(cleaned)
    if key.startswith("cred ") or key == "cred":
        return "CRED"
    return _CANONICAL_ALIASES.get(key) or _CANONICAL_COMPANIES.get(key)


def infer_company_name(
    description: str,
    *,
    category: str = "",
    explicit_name: object = None,
) -> Optional[str]:
    """Return a company only when merchant evidence is strong enough to label."""
    if category in _PERSONAL_CATEGORIES:
        return None

    explicit = _canonical_company_name(explicit_name)
    if explicit:
        return explicit

    handle = extract_upi_handle(description or "")
    if handle:
        try:
            merchant = get_merchant(handle)
        except Exception:
            merchant = None
        if (
            merchant
            and merchant.get("category") not in _PERSONAL_CATEGORIES | {"unclassified", "upi_spend"}
            and float(merchant.get("confidence", 0.0)) >= 0.70
        ):
            company = _canonical_company_name(merchant.get("display_name"))
            if company:
                return company

    try:
        evidence_description = "" if handle else (description or "")
        inferred = classify_upi_merchant(handle or "", evidence_description, learn=False)
    except Exception:
        return None

    if (
        inferred.get("category") in _PERSONAL_CATEGORIES | {"unclassified"}
        or float(inferred.get("confidence", 0.0)) < 0.70
    ):
        return None
    return _canonical_company_name(inferred.get("display_name"))
