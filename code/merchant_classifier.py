"""
Merchant Classifier — LLM-powered UPI merchant identification with caching.

Flow:
1. Extract UPI handle from transaction description
2. Look up in SQLite merchant DB → return if found
3. If not found, call LLM to classify → store in DB → return
4. DB acts as persistent cache — LLM called only once per new merchant

DB tables:
- merchants: upi_handle → display_name, category, is_income, confidence
- merchant_aliases: canonical_name → upi_handle (for dedup)
"""

import sqlite3
import re
import json
from pathlib import Path
from typing import Optional, Tuple

DB_PATH = Path(__file__).parent.parent.parent / "data" / "merchants.db"
# Persist outside of rsync path so deploys don't wipe it


def get_merchant(upi_handle: str) -> Optional[dict]:
    """Look up a UPI handle in the merchant database."""
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM merchants WHERE upi_handle = ?", (upi_handle,)
        ).fetchone()
        return dict(row) if row else None
    except sqlite3.OperationalError:
        return None
    finally:
        if conn is not None:
            conn.close()


def normalize_description_key(desc: str) -> str:
    """Normalize description for rule matching: uppercase, strip digits, collapse whitespace."""
    if not desc:
        return ""
    text = str(desc).upper().strip()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Guard against very short keys that could match unrelated transactions
    if len(text) < 5:
        return ""
    return text[:200]


def _ensure_description_rules_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS description_rules (
            description_key TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            is_income INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.90,
            sample_desc TEXT,
            created_at TEXT
        )
        """
    )


def store_description_rule(desc: str, category: str, is_income: bool, confidence: float = 0.90) -> bool:
    key = normalize_description_key(desc)
    if not key:
        return False
    conn = sqlite3.connect(str(DB_PATH))
    try:
        _ensure_description_rules_table(conn)
        import datetime
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO description_rules
               (description_key, category, is_income, confidence, sample_desc, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key, category, 1 if is_income else 0, confidence, desc[:200], now_str)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error storing description rule {key}: {e}")
        return False
    finally:
        conn.close()


def get_description_rule(desc: str) -> Optional[dict]:
    key = normalize_description_key(desc)
    if not key:
        return None
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        _ensure_description_rules_table(conn)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM description_rules WHERE description_key = ?", (key,)
        ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()


def extract_upi_handle(description: str) -> Optional[str]:
    """Extract the merchant/counterparty handle from an Indian bank transaction narration.

    Supports all major Indian bank narration formats:
    - UPI:    UPI/merchant_handle/purpose/BANK/ref/txn_id (ICICI, HDFC, Axis)
    - IMPS:   IMPS/merchant_handle/... or IMPS-merchant_handle-...
    - NEFT:   NEFT/merchant_handle/... or NEFT CR/merchant_name/...
    - RTGS:   RTGS/merchant_handle/... or RTGS-merchant_handle-...
    - NACH:   NACH/merchant_handle/... or NACH-merchant_handle-...
    - Generic: any string containing @vpa_handle pattern
    """
    if not description:
        return None
    desc = description.strip()

    # Format: UPI/handle/... (ICICI, HDFC, Axis, etc.)
    if desc.upper().startswith('UPI/'):
        parts = desc.split('/')
        if len(parts) >= 2 and parts[1].strip():
            return parts[1].strip().lower()[:100]

    # Format: GENERIC-UPI/handle/... (SBI)
    if 'UPI/' in desc.upper():
        idx = desc.upper().index('UPI/')
        parts = desc[idx:].split('/')
        if len(parts) >= 2 and parts[1].strip():
            return parts[1].strip().lower()[:100]

    # Generic stop-words that indicate the narration segment is NOT a merchant handle
    _NARRATION_STOP_WORDS = frozenset({
        "transfer", "to", "from", "cr", "dr", "credit", "debit",
        "payment", "refund", "reversal", "charges", "fee",
        "salary", "interest", "dividend", "rent", "emi", "loan",
        "tax", "tds", "cash", "deposit", "withdrawal",
    })

    # Format: IMPS/handle/... or IMPS-handle-...
    if desc.upper().startswith('IMPS'):
        parts = desc.split('/')
        if len(parts) >= 2 and parts[1].strip():
            candidate = parts[1].strip().lower()[:100]
            if candidate not in _NARRATION_STOP_WORDS:
                return candidate
        # IMPS-merchant-bank format
        dash_parts = desc.split('-')
        if len(dash_parts) >= 2 and dash_parts[1].strip():
            candidate = dash_parts[1].strip().lower()[:100]
            if candidate not in _NARRATION_STOP_WORDS:
                return candidate

    # Format: NEFT/handle/... or NEFT CR/handle/... or NEFT DR/handle/...
    if desc.upper().startswith('NEFT'):
        parts = desc.split('/')
        # Skip CR/DR suffix in first segment
        start_idx = 1
        if len(parts) >= 2 and parts[0].strip().upper() in ('NEFT CR', 'NEFT DR'):
            start_idx = 1
        if len(parts) > start_idx and parts[start_idx].strip():
            candidate = parts[start_idx].strip().lower()[:100]
            if candidate not in _NARRATION_STOP_WORDS:
                return candidate
        # NEFT-merchant-bank format
        dash_parts = desc.split('-')
        if len(dash_parts) >= 2 and dash_parts[1].strip():
            candidate = dash_parts[1].strip().lower()[:100]
            if candidate not in _NARRATION_STOP_WORDS:
                return candidate

    # Format: RTGS/handle/... or RTGS-handle-...
    if desc.upper().startswith('RTGS'):
        parts = desc.split('/')
        if len(parts) >= 2 and parts[1].strip():
            candidate = parts[1].strip().lower()[:100]
            if candidate not in _NARRATION_STOP_WORDS:
                return candidate
        dash_parts = desc.split('-')
        if len(dash_parts) >= 2 and dash_parts[1].strip():
            candidate = dash_parts[1].strip().lower()[:100]
            if candidate not in _NARRATION_STOP_WORDS:
                return candidate

    # Format: NACH/handle/... or NACH-handle-...
    if desc.upper().startswith('NACH'):
        parts = desc.split('/')
        if len(parts) >= 2 and parts[1].strip():
            candidate = parts[1].strip().lower()[:100]
            if candidate not in _NARRATION_STOP_WORDS:
                return candidate
        dash_parts = desc.split('-')
        if len(dash_parts) >= 2 and dash_parts[1].strip():
            candidate = dash_parts[1].strip().lower()[:100]
            if candidate not in _NARRATION_STOP_WORDS:
                return candidate

    # Format: handle@vpa (direct UPI ID in description)
    m = re.search(r'([a-zA-Z0-9_.\-]{2,40}@[a-zA-Z]{2,20})', desc)
    if m:
        handle = m.group(1).lower()
        # Skip personal-looking handles (common names)
        personal_patterns = ['ybl', 'oksbi', 'okhdfc', 'okaxis', 'okicici', 'paytm', 'ibh',
                              'ybl', 'apl', 'axl', 'sbi', 'hdfcbank', 'icici', 'kotak']
        vpa = handle.split('@')[1] if '@' in handle else ''
        if vpa in personal_patterns:
            return handle  # Still return it — merchant DB can classify it as personal_transfer
        return handle

    # Format: UPI-DEBIT/handle/... or DEBIT-UPI/handle/...
    if 'UPI' in desc.upper():
        parts = desc.split('/')
        for i, part in enumerate(parts):
            if part.strip().upper().startswith('UPI') and i + 1 < len(parts):
                handle = parts[i + 1].strip()
                if handle:
                    return handle.lower()[:100]

    return None


# Heuristic merchant name extraction from UPI handle
def extract_display_name(upi_handle: str) -> str:
    """Extract a human-readable display name from a UPI handle."""
    # Take the part before @
    name = upi_handle.split('@')[0] if '@' in upi_handle else upi_handle
    # Remove common prefixes/suffixes
    name = re.sub(r'^(pay|p2p|p2m|merchant|txn|trn|order|bill)', '', name, flags=re.IGNORECASE)
    # Split on dots, hyphens, underscores and take meaningful parts
    parts = re.split(r'[.\-_\s]+', name)
    # Filter out short/empty parts and common noise
    meaningful = [p for p in parts if len(p) >= 2 and p.lower() not in ('upi', 'com', 'in', 'ltd')]
    if not meaningful:
        return name[:40].title()
    return ' '.join(meaningful[:3]).title()[:40]


# Heuristic category classification based on UPI handle keywords
MERCHANT_ALIASES = {
    # Handle pattern → (display_name, category, is_income, confidence)
    'apple': ('Apple', 'entertainment', False, 0.85),
    'appleservices': ('Apple Services', 'entertainment', False, 0.85),
    'amznlpa': ('Amazon', 'shopping', False, 0.85),
    'amazon': ('Amazon', 'shopping', False, 0.85),
    'discovery': ('Discovery+', 'entertainment', False, 0.85),
    'simpl': ('Simpl', 'credit_card', False, 0.85),
    'setu.simpl': ('Simpl', 'credit_card', False, 0.85),
    'dlf': ('DLF', 'bills', False, 0.75),
    'ambience': ('Ambience Mall', 'shopping', False, 0.75),
    'bistro': ('Bistro', 'food', False, 0.80),
    'bundl': ('Swiggy', 'food', False, 0.90),
    'eternal': ('Zomato', 'food', False, 0.90),
    'zepto': ('Zepto', 'grocery', False, 0.90),
    'blinkit': ('Blinkit', 'grocery', False, 0.90),
    'groww': ('Groww', 'investment', False, 0.85),
    'indmoney': ('IndMoney', 'investment', False, 0.85),
    'zerodha': ('Zerodha', 'trading_deposit', False, 0.85),
    'paytmqr': ('PayTM QR', 'bills', False, 0.70),
    'qutab': ('Qutab Plaza', 'bills', False, 0.70),
    'hsquare': ('H Square', 'bills', False, 0.70),
    'rumaani': ('Rumaani', 'food', False, 0.70),
    'laxman': ('Laxman Cafe', 'food', False, 0.70),
    'vinod': ('Vinod Mandi', 'grocery', False, 0.70),
    'idealprepa': ('Ideal Prep', 'education', False, 0.70),
}

CURATED_TRANSACTION_MARKERS = {
    'gpaytoll@icici': ('Google Pay FASTag', 'travel', False, 0.98),
    'blusmartmobilit': ('BluSmart', 'travel', False, 0.98),
    '1mg.payu@axisba': ('Tata 1mg', 'medical', False, 0.98),
    'artemis ho': ('Artemis Hospital', 'medical', False, 0.98),
    'artemishospita': ('Artemis Hospital', 'medical', False, 0.98),
    'the chemis': ('The Chemist', 'medical', False, 0.95),
    'the chemist': ('The Chemist', 'medical', False, 0.95),
    'zomatoindia@ic': ('Zomato', 'food', False, 0.98),
    'mgf mall m': ('MGF Mall Parking', 'bills', False, 0.95),
    'med point': ('Med Point', 'medical', False, 0.95),
}

HANDLE_CATEGORY_MAP = {
    # Food delivery
    'zomato': ('Zomato', 'food', False, 0.95),
    'swiggy': ('Swiggy', 'food', False, 0.95),
    'blinkit': ('Blinkit', 'grocery', False, 0.95),
    'zepto': ('Zepto', 'grocery', False, 0.95),
    'bigbasket': ('BigBasket', 'grocery', False, 0.95),
    'dominos': ('Dominos', 'food', False, 0.92),
    'pizzahut': ('Pizza Hut', 'food', False, 0.92),
    'kfc': ('KFC', 'food', False, 0.90),
    'mcdonald': ("McDonald's", 'food', False, 0.92),
    'eatfit': ('EatFit', 'food', False, 0.85),
    'box8': ('Box8', 'food', False, 0.85),
    # Shopping
    'amazon': ('Amazon', 'shopping', False, 0.90),
    'flipkart': ('Flipkart', 'shopping', False, 0.90),
    'myntra': ('Myntra', 'shopping', False, 0.90),
    'ajio': ('AJIO', 'shopping', False, 0.88),
    'meesho': ('Meesho', 'shopping', False, 0.85),
    'nykaa': ('Nykaa', 'shopping', False, 0.88),
    'tatacliq': ('Tata CLiQ', 'shopping', False, 0.85),
    'jiomart': ('JioMart', 'grocery', False, 0.88),
    'bigbazaar': ('Big Bazaar', 'grocery', False, 0.82),
    # Travel
    'uber': ('Uber', 'travel', False, 0.95),
    'ola': ('Ola', 'travel', False, 0.95),
    'blusmart': ('BluSmart', 'travel', False, 0.92),
    'rapido': ('Rapido', 'travel', False, 0.92),
    'irctc': ('IRCTC', 'travel', False, 0.95),
    'makemytrip': ('MakeMyTrip', 'travel', False, 0.90),
    'redbus': ('RedBus', 'travel', False, 0.90),
    'goibibo': ('Goibibo', 'travel', False, 0.88),
    'indigo': ('Indigo Airlines', 'travel', False, 0.92),
    'airindia': ('Air India', 'travel', False, 0.90),
    # Entertainment
    'netflix': ('Netflix', 'entertainment', False, 0.95),
    'spotify': ('Spotify', 'entertainment', False, 0.95),
    'hotstar': ('Disney+ Hotstar', 'entertainment', False, 0.92),
    'prime': ('Amazon Prime', 'entertainment', False, 0.90),
    'youtube': ('YouTube', 'entertainment', False, 0.95),
    'playstore': ('Google Play Store', 'entertainment', False, 0.92),
    'sonyliv': ('SonyLIV', 'entertainment', False, 0.88),
    'jiosaavn': ('JioSaavn', 'entertainment', False, 0.85),
    # Bills & utilities
    'gpay-utility': ('Google Pay Utility', 'bills', False, 0.80),
    'mygate': ('MyGate', 'bills', False, 0.90),
    'paytm-mygate': ('MyGate Society', 'bills', False, 0.90),
    'electricity': ('Electricity Bill', 'bills', False, 0.82),
    'water': ('Water Bill', 'bills', False, 0.80),
    'gas': ('Gas Bill', 'bills', False, 0.80),
    'broadband': ('Broadband Bill', 'bills', False, 0.82),
    'airtel': ('Airtel', 'bills', False, 0.85),
    'jio': ('Jio', 'bills', False, 0.82),
    'vodafone': ('Vodafone Idea', 'bills', False, 0.80),
    'bsnl': ('BSNL', 'bills', False, 0.80),
    # Insurance
    'nivabupa': ('Niva Bupa Insurance', 'insurance', False, 0.92),
    'hdfclife': ('HDFC Life', 'insurance', False, 0.90),
    'iciciprulife': ('ICICI Prudential Life', 'insurance', False, 0.90),
    'lic': ('LIC', 'insurance', False, 0.88),
    'starhealth': ('Star Health', 'insurance', False, 0.88),
    # Trading / investments
    'zerodha': ('Zerodha', 'trading_deposit', False, 0.98),
    'groww': ('Groww', 'trading_deposit', False, 0.92),
    'indmoney': ('INDmoney', 'investment', False, 0.90),
    'upstox': ('Upstox', 'trading_deposit', False, 0.90),
    'angelone': ('Angel One', 'trading_deposit', False, 0.90),
    '5paisa': ('5paisa', 'trading_deposit', False, 0.85),
    # Credit card payments via CRED — check before food/shopping (CRED intermediates for many merchants)
    'cred.club': ('CRED', 'credit_card', False, 0.95),
    'cred': ('CRED', 'credit_card', False, 0.95),
    'paytm-jiomobili': ('CRED Bill Pay', 'bills', False, 0.82),
    'payzomato@hdfcb': ('CRED Bill Pay', 'bills', False, 0.75),
    'paytm-credit': ('Paytm Credit Card', 'credit_card', False, 0.88),
    # Medical
    'pharmeasy': ('PharmEasy', 'medical', False, 0.90),
    'tata1mg': ('Tata 1mg', 'medical', False, 0.90),
    '1mg': ('Tata 1mg', 'medical', False, 0.90),
    'apollo': ('Apollo Pharmacy', 'medical', False, 0.82),
    'netmeds': ('Netmeds', 'medical', False, 0.85),
    'artemis': ('Artemis Hospital', 'medical', False, 0.88),
    # Education
    'udemy': ('Udemy', 'education', False, 0.92),
    'coursera': ('Coursera', 'education', False, 0.92),
    'unacademy': ('Unacademy', 'education', False, 0.90),
    'byjus': ("Byju's", 'education', False, 0.88),
    # Personal transfers (VPA patterns indicating P2P)
    'ybl': ('UPI Transfer', 'personal_transfer', False, 0.40),
    'oksbi': ('UPI Transfer', 'personal_transfer', False, 0.40),
    'okhdfc': ('UPI Transfer', 'personal_transfer', False, 0.40),
    'okaxis': ('UPI Transfer', 'personal_transfer', False, 0.40),
    'okicici': ('UPI Transfer', 'personal_transfer', False, 0.40),
    'apl': ('UPI Transfer', 'personal_transfer', False, 0.40),
    # --- GitHub-augmented: high-signal UPI handles from training data ---
    'cred.club': ('CRED', 'credit_card', False, 0.95),
    'payzomato': ('Zomato Pay (via CRED)', 'bills', False, 0.85),
    'setu.simpl': ('Simpl', 'credit_card', False, 0.90),
    'airindia.bdpg': ('Air India', 'travel', False, 0.90),
    'paytmqr': ('Paytm Merchant', 'bills', False, 0.70),
    # --- Training-data misclassification fixes ---
    'grofersindia': ('Blinkit (Grofers)', 'grocery', False, 0.85),
    'flightsmojoin': ('Flight Booking', 'travel', False, 0.80),
    'khargymkhana': ('Khar Gymkhana', 'health_fitness', False, 0.85),
    'getsimpl': ('Simpl', 'credit_card', False, 0.90),
    # --- Cash withdrawal ---
    'atm': ('ATM Withdrawal', 'cash_withdrawal', False, 0.85),
}


PERSONAL_TRANSFER_MARKERS = (
    'p2p',
    'personal transfer',
    'send money',
)

# Generic category words belong to transaction-purpose inference, not merchant identity.
GENERIC_NARRATION_KEYWORDS = {
    'electricity', 'water', 'gas', 'broadband', 'jio', 'lic', 'atm', 'prime',
}

# Conservative purpose/category evidence from the complete bank narration.
# These rules intentionally exclude generic words such as "payment" and "purchase".
NARRATION_CATEGORY_RULES = (
    ('credit_card', 'Credit Card Payment', 0.86, (
        'credit card bill', 'card bill payment', 'credit card payment',
    )),
    ('tax_payment', 'Tax Payment', 0.86, (
        'income tax', 'advance tax', 'tax challan', 'tax payment',
    )),
    ('insurance', 'Insurance Premium', 0.84, (
        'insurance premium', 'policy premium',
    )),
    ('medical', 'Medical', 0.80, (
        'pharmacy', 'hospital', 'medical store', 'clinic payment',
    )),
    ('education', 'Education', 0.80, (
        'school fee', 'college fee', 'tuition fee', 'course fee',
    )),
    ('trading_deposit', 'Trading Deposit', 0.82, (
        'trading account', 'broker deposit',
    )),
    ('investment', 'Investment', 0.82, (
        'mutual fund', 'sip investment', 'investment contribution',
    )),
    ('grocery', 'Grocery', 0.78, (
        'grocery', 'supermarket', 'kirana', 'provision store',
    )),
    ('food', 'Food', 0.76, (
        'restaurant', 'food order', 'cafe payment', 'meal payment',
    )),
    ('travel', 'Travel', 0.78, (
        'flight booking', 'hotel booking', 'cab ride', 'railway ticket',
        'travel booking',
    )),
    ('entertainment', 'Entertainment', 0.76, (
        'movie ticket', 'cinema', 'streaming subscription',
    )),
    ('bills', 'Utility Bill', 0.78, (
        'electricity bill', 'water bill', 'gas bill', 'mobile recharge',
        'broadband bill', 'utility bill',
    )),
    ('shopping', 'Shopping', 0.72, (
        'retail purchase', 'shopping order', 'apparel', 'electronics purchase',
    )),
    ('staff_salary', 'Staff Salary', 0.82, (
        'staff salary', 'maid salary', 'driver salary',
    )),
    ('donation', 'Donation', 0.78, ('donation', 'charity contribution')),
    ('cash_withdrawal', 'Cash Withdrawal', 0.85, (
        'cash withdrawal', 'upi atm withdrawal',
    )),
)


def _normalize_evidence(value: str) -> str:
    """Normalize narration text for conservative token/phrase matching."""
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', value.lower()).split())


def _contains_evidence(value: str, phrase: str) -> bool:
    """Match a normalized token or phrase without accidental substrings."""
    normalized_value = f" {_normalize_evidence(value)} "
    normalized_phrase = _normalize_evidence(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_value


def _handle_contains_keyword(handle: str, keyword: str) -> bool:
    """Match exact token phrases or brand-prefixed handle tokens without infixes."""
    if _contains_evidence(handle, keyword):
        return True
    compact_keyword = _normalize_evidence(keyword).replace(" ", "")
    if len(compact_keyword) <= 4:
        return False
    handle_tokens = re.findall(r"[a-z0-9]+", handle.lower())
    return any(token.startswith(compact_keyword) for token in handle_tokens)


def get_curated_transaction_override(
    upi_handle: str,
    sample_description: str = "",
) -> Optional[dict]:
    """Return only exact transaction markers that may outrank learned cache rows."""
    handle_lower = (upi_handle or "").lower().strip()
    description_lower = (sample_description or "").lower()
    for marker, (display, category, is_income, confidence) in CURATED_TRANSACTION_MARKERS.items():
        marker_pattern = rf"(?<![a-z0-9._@-]){re.escape(marker)}(?![a-z0-9._@-])"
        if handle_lower == marker or re.search(marker_pattern, description_lower):
            return {
                "display_name": display,
                "category": category,
                "is_income": is_income,
                "confidence": confidence,
                "rationale": f"Curated transaction marker: {display}",
            }
    return None


def get_curated_merchant_override(
    upi_handle: str,
    sample_description: str = "",
) -> Optional[dict]:
    """Return curated markers and handle aliases for heuristic classification."""
    curated = get_curated_transaction_override(upi_handle, sample_description)
    if curated:
        return curated
    handle_lower = (upi_handle or "").lower().strip()
    for alias_key, (display, category, is_income, confidence) in MERCHANT_ALIASES.items():
        if _handle_contains_keyword(handle_lower, alias_key):
            return {
                "display_name": display,
                "category": category,
                "is_income": is_income,
                "confidence": confidence,
                "rationale": f"Merchant alias: {display}",
            }
    return None


def classify_upi_merchant(
    upi_handle: str,
    sample_description: str,
    *,
    learn: bool = True,
) -> dict:
    """Infer a UPI category, optionally learning stable handle evidence."""
    handle_lower = (upi_handle or '').lower().strip()
    description = sample_description or ''

    curated = get_curated_merchant_override(upi_handle, description)
    if curated:
        return curated

    # Exact handle identity always outranks incidental merchant text in narration.
    sorted_map = sorted(HANDLE_CATEGORY_MAP.items(), key=lambda item: len(item[0]), reverse=True)
    merchant_match = next(
        (
            (keyword, merchant)
            for keyword, merchant in sorted_map
            if merchant[1] != 'personal_transfer'
            and _handle_contains_keyword(handle_lower, keyword)
        ),
        None,
    )

    # Only high-confidence, sufficiently specific merchant names may match narration.
    if merchant_match is None:
        merchant_match = next(
            (
                (keyword, merchant)
                for keyword, merchant in sorted_map
                if merchant[1] != 'personal_transfer'
                and merchant[3] >= 0.80
                and keyword not in GENERIC_NARRATION_KEYWORDS
                and len(_normalize_evidence(keyword).replace(' ', '')) >= 4
                and _contains_evidence(description, keyword)
            ),
            None,
        )

    if merchant_match is not None:
        keyword, (display, category, is_income, confidence) = merchant_match
        if category == 'credit_card' and 'cred' in keyword:
            for part in description.split('/'):
                part = part.strip()
                if any(bank in part.upper() for bank in [
                    'AXIS BANK', 'HDFC BANK', 'ICICI BANK', 'SBI', 'YES BANK',
                    'KOTAK', 'IDFC', 'INDUSIND', 'AMERICAN EXPRESS',
                    'STANDARD CHARTED', 'STANDARD CHARTERED', 'RBL', 'FEDERAL',
                    'BANDHAN', 'YES BANK LIMITE',
                ]):
                    display = f'CRED — {part.title()}'
                    break

        # Specific known-merchant evidence is stable enough to learn for this handle.
        if handle_lower and learn:
            try:
                store_merchant(
                    upi_handle,
                    display,
                    category,
                    is_income=is_income,
                    confidence=confidence,
                    sample_desc=description[:200],
                )
            except Exception:
                pass
        return {
            'display_name': display,
            'category': category,
            'is_income': is_income,
            'confidence': confidence,
            'rationale': f'Known UPI merchant evidence: {display}',
        }

    display = extract_display_name(handle_lower) or 'Unknown UPI counterparty'

    # Purpose/category evidence is transaction-specific, so do not cache it by handle.
    for category, generic_display, confidence, phrases in NARRATION_CATEGORY_RULES:
        matched_phrase = next(
            (phrase for phrase in phrases if _contains_evidence(description, phrase)),
            None,
        )
        if matched_phrase:
            return {
                'display_name': display if display != 'Unknown UPI counterparty' else generic_display,
                'category': category,
                'is_income': False,
                'confidence': confidence,
                'rationale': f'UPI narration evidence: {matched_phrase}',
            }

    local_part = handle_lower.split('@', 1)[0]
    compact_local = re.sub(r'[^a-z0-9]', '', local_part)
    mostly_numeric = bool(compact_local) and (
        compact_local.isdigit()
        or sum(character.isdigit() for character in compact_local) / len(compact_local) >= 0.8
    )
    explicit_personal = any(
        _contains_evidence(description, marker) for marker in PERSONAL_TRANSFER_MARKERS
    )

    # Indian person-name P2P detection
    local_part_fallback = handle_lower.split('@', 1)[0] if handle_lower else ''
    # Remove non-alpha chars to evaluate the name
    alpha_only = re.sub(r'[^a-z]', '', local_part_fallback)
    # Skip masked handles (xxxxxxxxxx), repeated-char handles, and handles
    # where the local part is mostly one repeated character — these are
    # privacy-masked VPAs, not person names
    unique_chars = set(alpha_only)
    is_masked = len(unique_chars) <= 2  # e.g. "xxxxxxxxxx" → {'x'} → masked
    # If handle local part is 5+ alphabetic chars, has no merchant keywords,
    # no digits, no known brand indicators, and doesn't match any narration
    # category → classify as personal_transfer at 0.40 confidence
    if (len(alpha_only) >= 5
        and not is_masked  # exclude masked/repeated-char handles
        and not any(kw in alpha_only for kw in (
            'paytm', 'phonepe', 'gpay', 'amazon', 'flipkart', 'zomato',
            'swiggy', 'blinkit', 'zepto', 'cred', 'bill', 'pay', 'tax',
            'loan', 'emi', 'insur', 'med', 'hospital', 'pharma', 'food',
            'mart', 'store', 'shop', 'bazar', 'mall', 'petrol', 'gas',
            'electric', 'water', 'broadband', 'recharge', 'netflix',
            'spotify', 'prime', 'hotstar', 'disney', 'apple', 'google',
            'flight', 'air', 'irctc', 'mmt', 'makemy', 'yatra', 'goibibo', 'cleartrip',
            'uber', 'ola', 'rapido', 'rent', 'pg', 'hostel',
        ))
        and not mostly_numeric  # already handled above
        and not explicit_personal  # already handled above
        and not merchant_match  # no merchant evidence found
        and not any(_contains_evidence(description, phrase)
                    for category, _, _, phrases in NARRATION_CATEGORY_RULES
                    for phrase in phrases)
    ):
        return {
            'display_name': 'UPI Transfer',
            'category': 'personal_transfer',
            'is_income': False,
            'confidence': 0.40,
            'rationale': 'Personal UPI transfer — no merchant evidence in handle or narration',
        }

    if mostly_numeric or explicit_personal:
        return {
            'display_name': 'UPI Transfer',
            'category': 'personal_transfer',
            'is_income': False,
            'confidence': 0.55 if mostly_numeric else 0.60,
            'rationale': 'Strong personal-transfer evidence in UPI transaction',
        }

    return {
        'display_name': display,
        'category': 'unclassified',
        'is_income': False,
        'confidence': 0.35,
        'rationale': 'No reliable merchant or purpose evidence in UPI transaction',
    }


def store_merchant(upi_handle: str, display_name: str, category: str,
                   is_income: bool = False, confidence: float = 0.85,
                   sample_desc: str = '') -> bool:
    """Store a classified merchant in the database."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """INSERT OR REPLACE INTO merchants
               (upi_handle, display_name, category, is_income, confidence, sample_desc)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (upi_handle.lower(), display_name, category,
             1 if is_income else 0, confidence, sample_desc[:200])
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error storing merchant {upi_handle}: {e}")
        return False
    finally:
        conn.close()


def batch_store(merchants: list[dict]) -> int:
    """Store multiple merchants at once. Each dict: {upi_handle, display_name, category, is_income, confidence, sample_desc}"""
    conn = sqlite3.connect(str(DB_PATH))
    count = 0
    for m in merchants:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO merchants
                   (upi_handle, display_name, category, is_income, confidence, sample_desc)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (m['upi_handle'].lower(), m['display_name'], m['category'],
                 1 if m.get('is_income') else 0, m.get('confidence', 0.85),
                 m.get('sample_desc', '')[:200])
            )
            count += 1
        except Exception:
            pass
    conn.commit()
    conn.close()
    return count


def get_db_stats() -> dict:
    """Get statistics about the merchant database."""
    conn = sqlite3.connect(str(DB_PATH))
    total = conn.execute("SELECT COUNT(*) FROM merchants").fetchone()[0]
    by_cat = conn.execute(
        "SELECT category, COUNT(*) as cnt FROM merchants GROUP BY category ORDER BY cnt DESC"
    ).fetchall()
    conn.close()
    return {
        'total_merchants': total,
        'categories': {cat: cnt for cat, cnt in by_cat}
    }


# ─── Seed Data: Known merchants from regex patterns ───

SEED_MERCHANTS = [
    # Trading / investments
    {'upi_handle': 'zerodhabroking@', 'display_name': 'Zerodha', 'category': 'trading_deposit', 'is_income': False, 'confidence': 0.98},
    {'upi_handle': 'indmoney@', 'display_name': 'INDmoney', 'category': 'investment', 'is_income': False, 'confidence': 0.90},

    # Food delivery
    {'upi_handle': 'zomato-order@pt', 'display_name': 'Zomato', 'category': 'food', 'is_income': False, 'confidence': 0.95},
    {'upi_handle': 'swiggy@', 'display_name': 'Swiggy', 'category': 'food', 'is_income': False, 'confidence': 0.95},

    # Shopping
    {'upi_handle': 'amazon-pod@rap', 'display_name': 'Amazon', 'category': 'shopping', 'is_income': False, 'confidence': 0.90},
    {'upi_handle': 'amazonsellerser', 'display_name': 'Amazon Seller Services', 'category': 'shopping', 'is_income': False, 'confidence': 0.85},
    {'upi_handle': 'flipkart@', 'display_name': 'Flipkart', 'category': 'shopping', 'is_income': False, 'confidence': 0.90},

    # Bills & utilities
    {'upi_handle': 'gpay-utility@ok', 'display_name': 'Google Pay Utility', 'category': 'bills', 'is_income': False, 'confidence': 0.80},
    {'upi_handle': 'youtube@axisba', 'display_name': 'YouTube Premium', 'category': 'entertainment', 'is_income': False, 'confidence': 0.95},
    {'upi_handle': 'playstore@axis', 'display_name': 'Google Play Store', 'category': 'entertainment', 'is_income': False, 'confidence': 0.95},
    {'upi_handle': 'netflix@', 'display_name': 'Netflix', 'category': 'entertainment', 'is_income': False, 'confidence': 0.95},

    # Insurance (known providers)
    {'upi_handle': 'nivabupa@', 'display_name': 'Niva Bupa Insurance', 'category': 'insurance', 'is_income': False, 'confidence': 0.92},

    # Society / maintenance
    {'upi_handle': 'paytm-mygate@pt', 'display_name': 'MyGate Society', 'category': 'bills', 'is_income': False, 'confidence': 0.90},
    {'upi_handle': 'mygate.razorpa', 'display_name': 'MyGate', 'category': 'bills', 'is_income': False, 'confidence': 0.90},

    # Travel
    {'upi_handle': 'uber@', 'display_name': 'Uber', 'category': 'travel', 'is_income': False, 'confidence': 0.95},
    {'upi_handle': 'ola@', 'display_name': 'Ola', 'category': 'travel', 'is_income': False, 'confidence': 0.95},
    {'upi_handle': 'irctc@', 'display_name': 'IRCTC', 'category': 'travel', 'is_income': False, 'confidence': 0.95},
    {'upi_handle': 'airindiaexpress', 'display_name': 'Air India Express', 'category': 'travel', 'is_income': False, 'confidence': 0.90},

    # Credit card payments
    {'upi_handle': 'cred@', 'display_name': 'CRED', 'category': 'credit_card', 'is_income': False, 'confidence': 0.95},

    # Grocery
    {'upi_handle': 'blinkit@', 'display_name': 'Blinkit', 'category': 'grocery', 'is_income': False, 'confidence': 0.95},
    {'upi_handle': 'zepto@', 'display_name': 'Zepto', 'category': 'grocery', 'is_income': False, 'confidence': 0.95},
    {'upi_handle': 'bigbasket@', 'display_name': 'BigBasket', 'category': 'grocery', 'is_income': False, 'confidence': 0.95},
]


def seed_database():
    """Initialize the merchant database with known merchants."""
    count = batch_store(SEED_MERCHANTS)
    print(f"Seeded {count} known merchants")
    return count
