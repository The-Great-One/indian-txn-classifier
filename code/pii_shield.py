"""
pii_shield.py — Lightweight PII Masking
=======================================
Masks sensitive data before logging or sending to external services.
Pure regex, zero dependencies — works in 512MB environments.

Detects & masks:
  - PAN numbers (ABCDE1234F pattern)
  - Aadhaar numbers (12 digits)
  - Bank account numbers (9-18 digits near keywords)
  - IFSC codes
  - Mobile numbers
  - Email addresses
"""

import re
import logging

logger = logging.getLogger(__name__)

PII_PATTERNS = [
    {
        "name": "PAN",
        "pattern": r'\b([A-Z]{5}\d{4}[A-Z])\b',
        "mask": "XXXXX0000X",
    },
    {
        "name": "Aadhaar",
        "pattern": r'\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b',
        "mask": "XXXX-XXXX-XXXX",
    },
    {
        "name": "Mobile",
        "pattern": r'(?:\+91[\s\-]?)?\b([6-9]\d{9})\b',
        "mask": "XXXXXXXXXX",
    },
    {
        "name": "Email",
        "pattern": r'\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b',
        "mask": "***@***.***",
    },
    {
        "name": "IFSC",
        "pattern": r'\b([A-Z]{4}0[A-Z0-9]{6})\b',
        "mask": "XXXX0XXXXXX",
    },
    {
        "name": "BankAccount",
        "pattern": r'(?:account[\s\-:]*(?:no|number|num)?[\s\-:]*)\b(\d{9,18})\b',
        "mask": "XXXXXXXXXX",
        "case_insensitive": True,
    },
]


def mask_pii(text: str) -> str:
    """Mask all PII patterns in the given text. Returns sanitized text."""
    masked = text
    for p in PII_PATTERNS:
        flags = re.IGNORECASE if p.get("case_insensitive") else 0
        pattern = p["pattern"]
        mask = p["mask"]
        count = 0
        
        def replace(m):
            nonlocal count
            count += 1
            return mask
        
        masked = re.sub(pattern, replace, masked, flags=flags)
        if count > 0:
            logger.debug(f"PII Shield: masked {count} {p['name']}(s)")
    
    return masked


def has_pii(text: str) -> bool:
    """Check if text contains any PII. Returns True if sensitive data detected."""
    for p in PII_PATTERNS:
        flags = re.IGNORECASE if p.get("case_insensitive") else 0
        if re.search(p["pattern"], text, flags):
            return True
    return False


def shield_transaction(description: str) -> str:
    """Shield a single transaction description — masks PAN/Aadhaar but preserves merchant info."""
    # Only mask PAN and Aadhaar — keep bank account numbers visible for classification
    for pattern_name in ["PAN", "Aadhaar"]:
        for p in PII_PATTERNS:
            if p["name"] == pattern_name:
                description = re.sub(p["pattern"], p["mask"], description)
    return description
