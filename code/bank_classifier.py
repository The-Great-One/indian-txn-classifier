"""
Bank Statement Classifier — Multi-bank parser + AI classification pipeline.

Parses PDF/CSV statements from ICICI, SBI, HDFC and generic formats.
Classifies every credit transaction as income category using:
1. Rule engine (70-80% coverage)
2. Recurring pattern detector (10% more)
3. LLM fallback for remaining uncertain transactions
"""
import re, json, csv, io, logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import re as _re_module

logger = logging.getLogger(__name__)

BANK_PATTERNS = [
    ("ICICI", ["ICICI", "icici"], "🏦"),
    ("HDFC", ["HDFC", "hdfc"], "🏦"),
    ("SBI", ["SBI", "State Bank", "STATE BANK"], "🏛️"),
    ("Axis", ["AXIS", "axis"], "🏦"),
    ("Kotak", ["KOTAK", "kotak"], "🏦"),
    ("Yes Bank", ["YES BANK", "YESBANK", "yes bank"], "🏦"),
    ("Federal Bank", ["FEDERAL", "FDRL", "federal"], "🏦"),
    ("IDFC First", ["IDFC", "idfc"], "🏦"),
    ("IndusInd", ["INDUSIND", "indusind"], "🏦"),
    ("Bank of Baroda", ["BARODA", "BOB", "baroda"], "🏦"),
    ("Punjab National", ["PNB", "PUNJAB NATIONAL", "pnb"], "🏦"),
    ("Canara", ["CANARA", "canara"], "🏦"),
    ("Union Bank", ["UNION BANK", "UNION"], "🏦"),
    ("Unity SFB", ["UNITY", "unity"], "🏦"),
]


def detect_bank(filepath: str) -> dict:
    """Detect bank name from a statement file. Returns {name, icon}."""
    filepath = str(filepath)
    # Check filename first
    fname_lower = Path(filepath).name.lower()
    
    for bank_name, patterns, icon in BANK_PATTERNS:
        for pat in patterns:
            if pat.lower() in fname_lower:
                return {"name": bank_name, "icon": icon}
    
    # If not in filename, try reading the file content.
    try:
        suffix = Path(filepath).suffix.lower()
        if suffix in ('.xls', '.xlsx'):
            import pandas as pd
            df = pd.read_excel(filepath, header=None)
            # Check first 20 rows for bank name
            for i in range(min(20, len(df))):
                for j in range(min(8, len(df.columns))):
                    val = str(df.iloc[i, j])
                    for bank_name, patterns, icon in BANK_PATTERNS:
                        for pat in patterns:
                            if _re_module.search(pat, val, _re_module.IGNORECASE):
                                return {"name": bank_name, "icon": icon}
        elif suffix == '.pdf':
            import pymupdf
            with pymupdf.open(filepath) as doc:
                content = "\n".join(str(doc[index].get_text())
                                    for index in range(min(3, len(doc))))
            for bank_name, patterns, icon in BANK_PATTERNS:
                if any(re.search(re.escape(pattern), content, re.IGNORECASE)
                       for pattern in patterns):
                    return {"name": bank_name, "icon": icon}
    except Exception:
        pass
    
    return {"name": "Unknown Bank", "icon": "🏦"}
from typing import Optional
from collections import defaultdict

# Merchant DB lookup for UPI transactions
try:
    from .merchant_classifier import get_merchant, extract_upi_handle
except ImportError:
    from pipeline.merchant_classifier import get_merchant, extract_upi_handle


@dataclass
class RawTransaction:
    """Bank-agnostic normalized transaction."""
    date: date
    description: str
    type: str          # 'credit' | 'debit'
    amount: float
    balance: Optional[float] = None
    ref_number: Optional[str] = None

@dataclass
class ClassifiedTransaction:
    """Transaction with AI classification."""
    raw: RawTransaction
    category: str = 'unclassified'
    income_type: Optional[str] = None
    confidence: float = 0.0
    rationale: str = ''
    is_income: bool = False
    is_expense: bool = False
    recurring: bool = False
    counterparty: str = ''
    tags: list = field(default_factory=list)

@dataclass  
class ClassificationReport:
    """Summary of classification results."""
    total: int = 0
    credits: int = 0
    debits: int = 0
    rule_classified: int = 0
    recurring_detected: int = 0
    llm_classified: int = 0
    unclassified: int = 0
    income_detected: int = 0
    classified: list = field(default_factory=list)


# ─── Rule Engine ───────────────────────────────────────────

RULES = [
    # (name, category, patterns, confidence, is_income)
    ('salary', 'salary', [
        r'\bSALARY\b', r'\bSAL\s', r'\bMASTERCARD\b',
        r'\bPAYROLL\b', r'\bSALARIED\b',
    ], 0.98, True),
    
    ('dividend', 'dividend', [
        r'\bDIVIDEND\b', r'\bDIV\s', r'\bDIVIDEND\sWARRANT\b',
        r'\bACH/.*?(?:DIV|FINAL|INTERIM|INTDIV)\b',  # ACH dividend payments
        r'\bFINAL\s*(?:DIVIDEND|DIV)\b',
        r'\bINTERIM\s*DIVIDEND\b',
        # CMS dividend payments (various companies)
        r'\bCMS/.*(?:DIV|LIMITED|LIMITED\s-\s\\d)',  # CMS/CDSL, CMS/CESC, CMS/TECHNO
        r'\bCMS/CDSL\b', r'\bCMS/CESC\b', r'\bCMS/TECHNO\b',
        r'\bCMS/Deep\sIndustries\b', r'\bCMS/COMPUTER\sAGE\b',
        # NEFT dividends
        r'\bNEFT.*?(?:FINAL\sDIV|INTERIM\sDIV|\sDIV\s)',
        r'\bBAJAJ\sHEALTHCARE\b.*\bDIV\b', r'\bVENKYS\sINDIA\b',
        # Known dividend-paying companies via ACH/CMS
        r'\bVARUN\sBEVERAGES\b', r'\bADANI\s?ENT', r'\bZYDUS\b',
        r'\bSHANTHI\sGEARS\b', r'\bSIGACHI\b', r'\bMAITHAN\sALLOYS\b',
        r'\bRPG\s?LIFE\b', r'\bHPCL\b.*\bINTDIV\b',
        r'\bBRITANNIA\sINDUSTRIES\b', r'\bTATAELXSI\b',
        r'\bTVS\sMOTOR\b', r'\bRAILTEL\b', r'\bREC\sLIMITED\b',
    ], 0.90, True),
    
    ('business_trust_distribution', 'other_income', [
        r'\bPOWERGRID\sINFRA', r'\bINVIT\b', r'\bBUSINESS\sTRUST\b',
        r'\bEMBASSY\sOFFICE\b', r'\bINDIA\sGRID\b',
    ], 0.90, True),
    
    ('interest_savings', 'interest', [
        r'\bINTEREST\b', r'\bINT\sPAID\b', r'\bINT\sCR\b',
        r'\bFD\sINTEREST\b', r'\bSAVINGS\sINTEREST\b', r'\bINT\.?\s?(PD|CR)\b',
    ], 0.95, True),
    
    ('rental', 'rental', [
        r'\bRENT\b', r'\bHIRE\sCHARGES\b', r'\bLEASE\sRENT\b',
    ], 0.85, True),
    
    ('trading_credit', 'trading_credit', [
        r'\bZERODHA', r'\bANGEL', r'\bGROWW',
        r'\bUPSTOX', r'\bFYERS', r'\b5PAISA', r'\bICICI\sDIRECT',
        r'\bKOTAK\sSECURITIES', r'\bMOTILAL\sOSWAL', r'\bSHAREKHAN',
        r'\bBROKING', r'\bSECURITIES\sINDIA', r'\bPAYTM\sMONEY',
        r'\bINDIAN\sCLEARING\sCORPORATION\b', r'\bICCL\b',
        r'\bMUTUAL\sFUND.*REDEMPTION\b', r'\bCOMMON\sREDEMPTION\b',
    ], 0.80, False),  # NOT income — can't determine from bank statement alone
    
    ('tax_refund', 'tax_refund', [
        r'\bITD\b', r'\bINCOME\sTAX\b', r'\bIT\sREFUND\b',
        r'\bCPC\b', r'\bTAX\sREFUND\b',
    ], 0.95, False),  # Refund, not income
    
    ('loan_repayment', 'loan_repayment', [
        r'\bLOAN\sREPAY\b', r'\bLOAN\sRETURN\b', r'\bLENDING\b',
    ], 0.75, False),  # Not taxable income
    
    # ─── Additional Income Rules ───
    ('salary_fdrl', 'salary', [
        r'\bNEFT-FDRL.*(?:VK\sECOTRADE|CURRENT\sACCOUNT\sGENERAL)', r'\bVK\sECOTRADE\sLLP\b',
        r'\bFDRL.*SALARY\b',
    ], 0.95, True),
    
    ('rental_income_known', 'rental', [
        r'\bHENNYS\sWAFFLE\b', r'\bWAFFLE\sENTERPRISES\b',
        r'\bsubbu526@',  # Monica's tenant
        r'\bRAHUL\sPANDEY\b', r'\bSAMEER\sSHAIKH\b',
        r'Rent/',  # ICICI BIL/INFT/...Rent/ pattern
        r'\brent\b.*@[a-z]',  # UPI rent payments
    ], 0.88, True),
    
    ('mf_redemption', 'trading_credit', [
        r'\bMUTUAL\sFUND.*REDEMPTION\b', r'\bCOMMON\sREDEMPTION\b',
        r'\bKMMF\sREDEMPTIONS\b', r'\bCANARA\sROBECO.*REDEMPTION\b',
        r'\bELSS\sTAXSAVER.*INCOME\sDISTRIBUT\b',
        r'\bICICI\sPRUDENTIAL.*REDEMPTION\b',  # ICICI MF redemptions
    ], 0.90, True),
    
    ('interest_icici_format', 'interest', [
        r':Int\.Pd:',  # ICICI quarterly interest: "000501538878:Int.Pd:29-03-2025 to 29-06-2025"
    ], 0.98, True),
    
    ('family_transfer_in', 'personal_transfer', [
        r'\bVINOD\sKUMAR\sGUPTA\b',  # Monica's father
    ], 0.70, False),  # Not taxable, just flagged
    
    # ─── Additional Expense Rules ───
    ('society_maintenance', 'bills', [
        r'paytm-mygate@pt', r'@PT\b.*\bMAINT',  # Society maintenance
    ], 0.85, False),
    
    ('esanchala_bill', 'bills', [
        r'\bE\sSANCHALA\b', r'\bESANCHALAKSOLUT\b',  # Electricity/maintenance
    ], 0.85, False),
    
    ('maxbupa_insurance', 'insurance', [
        r'\bMAX\sBUPA\b', r'\bMAX\sBUPA\sH\b',  # Max Bupa health insurance
    ], 0.92, False),
    
    ('icici_securities_invest', 'investment', [
        r'\bEBA/MFP-',  # ICICI securities/insurance recurring investment
    ], 0.70, False),
    
    ('amazon_subscription', 'entertainment', [
        r'\bPUR_PRIME900\b',  # Amazon Prime subscription
    ], 0.90, False),
    
    ('airindia_flight', 'travel', [
        r'\bairindiaexpress\b',  # Air India Express flights
    ], 0.80, False),
    
    ('hospital_expense', 'medical', [
        r'\bBALABHAI\sNANAVATI\b',  # Hospital payments
    ], 0.85, False),
    ('mutual_fund_sip', 'investment', [
        r'\bSIP\b', r'\bMUTUAL\sFUND\b', r'\bMUTF\b', r'\bMF\sINVEST\b',
        r'\bELSS\b', r'\bNFO\b', r'\bFOLIO\b',
        r'\bINDMONEY', r'\bPAYU\b.*\bMONEY\b',  # No trailing \b — matches "indmoney3"
        r'\bZERODHAMF\b', r'\bBSESTAR', r'\bBSE\sSTAR',  # MF platforms
        r'\bICCLZR@YESPAY\b', r'\bICCLZERODHA\b', r'\bZERODHA\.ICCL',  # All ICCL channels = MF
    ], 0.85, False),
    
    ('staff_salary', 'staff_salary', [
        r'\bRAJ\sKUMARI\b', r'\bBIHARI\sSAH\b',  # Domestic staff
    ], 0.85, False),
    
    ('trading_transfer', 'trading_deposit', [
        r'\bZERODHA', r'\bANGEL', r'\bGROWW',
        r'\bUPSTOX', r'\bFYERS', r'\b5PAISA',
    ], 0.90, False),  # Money sent to trading account
    
    ('vehicle_purchase', 'vehicle_purchase', [
        r'\bASB\sAUTOMO', r'\bCAR\sDEALER\b', r'\bVEHICLE\b',
    ], 0.90, False),
    
    ('tax_payment_out', 'tax_payment', [
        r'\bINCOME\sTAX\b', r'\bADVANCE\sTAX\b', r'\bSELF\sASSESSMENT\b',
        r'\bCHALLAN\b', r'\bITNS\b', r'\bTDS\sPAYMENT\b',
        r'\bDTAX\b', r'\bGIB/',  # Tax payment patterns
    ], 0.92, False),
    
    ('toll_payment', 'travel', [  # FASTag, NHAI, toll — must be before credit_card
        r'\bNHAI\b', r'\bFAST\s?TAG\b', r'\bFASTAG\b', r'\bTOLL\b',
        r'\bIHMCL\b', r'\bGPTOLL\b', r'\bGP-TOLL\b', r'\bGP\.TOLL\b',
        r'\bPAYTOLL\b', r'\bNETC\s?FASTAG\b',
    ], 0.88, False),

    ('credit_card_payment', 'credit_card', [  # conf raised to 0.95 — CRED/Unipay are unambiguous
        r'\bCREDIT\sCARD\b', r'\bCC\sPAYMENT\b', r'\bUNIPAY\b.*\bCARD\b',
        r'\bCARD\sPAYMENT\b', r'\bCREDITCARD\b', r'\bSIMPL\b.*\bPAY\b',
        r'\bBIL/.*CREDIT\sC[A-Z]\b',
        r'\bCRED\b', r'\bCRED\.', r'\bCRED\sCLUB\b',  # CRED credit card payments
    ], 0.90, False),
    
    ('bill_payment', 'bills', [
        r'\bBILL\b', r'\bRECHARGE\b', r'\bELECTRICITY\b', r'\bELECTRIC\b',
        r'\bBROADBAND\b', r'\bWIFI\b', r'\bMOBILE\b', r'\bDTH\b',
        r'\bGAS\b', r'\bWATER\b', r'\bMAINTENANCE\b',
        r'\bGOOGLE\sIND', r'\bGOOGLE\b.*\bINDIA\b',  # Google services
        r'\bIDEALPREPA\b',  # Prepaid recharge
    ], 0.80, False),
    
    ('insurance', 'insurance', [
        r'\bINSURANCE\b', r'\bPREMIUM\b', r'\bLIC\b', r'\bPOLICY\b',
        r'\bICICI\sPRU\b', r'\bHDFC\sLIFE\b', r'\bMAX\sLIFE\b',
        r'\bTERM\sPLAN\b', r'\bHEALTH\sINSUR\b',
        r'\bNivaBupa', r'\bMAX\sBUPA\b', r'\bMAX\sBUPA\sH\b',  # Health insurance providers
        r'BIL/ONL.*NivaBupa', r'BIL/ONL.*MAX\sBUPA',  # BIL/ONL format insurance payments
    ], 0.92, False),
    
    ('grocery_delivery', 'grocery', [
        r'\bBLINKIT\b', r'\bZEPTO\b', r'\bINSTAMART\b',
        r'\bBIGBASKET\b', r'\bDMART\b', r'\bGROFERS\b',
    ], 0.85, False),
    
    ('loan_emi', 'loan_emi', [
        r'\bEMI\b', r'\bLOAN\sREPAYMENT\b', r'\bHOME\sLOAN\b',
        r'\bCAR\sLOAN\b', r'\bPERSONAL\sLOAN\b', r'\bEDUCATION\sLOAN\b',
        r'\bCMS/.*SMSOTP',  # Recurring CMS payments — typically loan EMI
    ], 0.85, False),
    
    ('cash_withdrawal', 'cash_withdrawal', [
        r'\bATM\b', r'\bCASH\sWDL\b', r'\bCASH\sWITHDRAWAL\b',
        r'\bCASH\sWDL\sRVSL\b',
    ], 0.95, False),
    
    ('gym_fitness', 'health_fitness', [
        r'\bEQUANIMITY\b', r'\bINNOVANAFI\b', r'\bGYMKHANA\b',
        r'\bGYM\b', r'\bFITNESS\b', r'\bKHAR\sGYM\b',
    ], 0.85, False),
    
    ('personal_transfer_out', 'personal_transfer', [
        r'\bMONICA\sGOE', r'\bRUHI\sTARUN', r'\bGOELMONICA',
        r'\bRUHIGOEL', r'\bTARUNKUMAR', r'\bJAI\sGUPTA',
        r'\bROHIT\sAROR', r'\bBHAVISHYA', r'\bGUPTARASHI',
        r'\bABHISHEK', r'\bPRIYA\sMANI', r'\bPRATEEK\sSI',
        r'\bVIPUL\sCHOU', r'\bARCHITA\sBA', r'\bAARSHIN\sBA',
    ], 0.80, False),
    
    ('rent_or_property', 'rent', [
        r'\bRAJ\sPHULLA\b',
    ], 0.70, False),
    
    ('paytm_merchant', 'misc_daily', [
        r'PAYTM', r'@PTY', r'@PTAX',  # Paytm merchant payments
    ], 0.60, False),
    
    ('rent_payment', 'rent', [
        r'\bRENT\sPAY\b', r'\bRENT\sTO\b', r'\bMAINTENANCE\sCHARGE\b',
    ], 0.80, False),
    
    ('food_dining', 'food', [
        r'\bSWIGGY\b', r'\bZOMATO\b', r'\bFOOD\b', r'\bRESTAURANT\b',
        r'\bDOMINOS\b', r'\bMCDONALD\b', r'\bEAT\b',
    ], 0.75, False),
    
    ('shopping', 'shopping', [
        r'\bAMAZON\b', r'\bFLIPKART\b', r'\bMYNTRA\b', r'\bAJIO\b',
        r'\bSHOP\b', r'\bRETAIL\b', r'\bMART\b', r'\bGROCERY\b',
        r'\bAPPLE\b.*\bONLIN\b', r'\bVIN/Apple\b',  # Apple online store
    ], 0.75, False),
    
    ('travel', 'travel', [
        r'\bUBER\b', r'\bOLA\b', r'\bRAPIDO\b', r'\bIRCTC\b',
        r'\bMAKEMYTRIP\b', r'\bFLIGHT\b', r'\bAIRLINE\b', r'\bBUS\b',
    ], 0.75, False),
    
    ('entertainment', 'entertainment', [
        r'\bNETFLIX\b', r'\bPRIME\b', r'\bHOTSTAR\b', r'\bSPOTIFY\b',
        r'\bYOUTUBE\b', r'\bSUBSCRIPTION\b', r'\bGAME\b',
    ], 0.70, False),
    
    ('neft_transfer', 'transfer', [
        r'\bNEFT-', r'\bIMPS/', r'\bRTGS',
        r'\bBIL/NEFT/',  # Bill payment via NEFT
    ], 0.40, False),  # Very low confidence — generic
    
    ('trading_fees', 'trading_fees', [
        r'\bDPCHG\b', r'\bDP\sCHGS\b', r'\bDP\sCHARGES\b',
        r'\bDMC/',  # Demat charges
        r'\bANNUAL\sMAINTENANCE\b.*\bDP\b',
    ], 0.92, False),
    
    ('credit_card_refund', 'credit_card_refund', [
        r'\bCREDIT\sCARD\b', r'\bCC\sPAYMENT\b', r'\bPAYMENT\sREVERSAL\b',
        r'\bCARD\sREFUND\b',
    ], 0.85, False),
    
    ('self_transfer', 'self_transfer', [
        r'\bSELF\b', r'\bOWN\sACCOUNT\b', r'\bTRANSFER\sTO\sSELF\b',
    ], 0.99, False),
    
    ('cash_deposit', 'cash_deposit', [
        r'\bCASH\sDEPOSIT\b', r'\bCASH\sDEP\b', r'\bCDM\b', r'\bBY\sCASH\b',
    ], 0.90, False),  # Flag for review
    
    # --- Training-data-augmented rules (NEFT/IMPS patterns) ---
    ('insurance_claim', 'insurance', [
        r'\bTHE\s+ORIENTAL\s+INSURANCE\b', r'\bORIENTAL\s+INS\b',
        r'\bINSURANCE\s+CO\b.*\bFHP\b',
    ], 0.85, True),
    
    ('nse_settlement', 'trading_credit', [
        r'\bNSE\s+CLEARING\b', r'\bMFSS\s+SETTLEMENT\b',
        r'\bNSE\s+CLR\b', r'\bNSCCL\b',
    ], 0.90, True),
    
    ('gift_from_family', 'family', [
        r'\bGIFT\s+TO\b', r'\bUSHA\s+GUPTA\b',
    ], 0.80, True),
    
    ('gift_in_self', 'personal_transfer', [
        r'\bSAHIL\s+TARUNKUMAR\s+GOEL\b', r'\bSAHIL\s+TARU\b',
    ], 0.80, False),
    
    ('mmt_hotel', 'travel', [
        r'\bMMT/IMPS.*HOUSR\b', r'\bHOUSR\s+TECH\b',
    ], 0.80, False),
    
    # --- Original catch-all rules ---
    ('upi_collect', 'unclassified_credit', [
        r'@[a-z]',  # UPI ID pattern
    ], 0.50, True),  # Low confidence — needs LLM
    
    ('neft_imps', 'unclassified_credit', [
        r'\bNEFT\b', r'\bIMPS\b', r'\bRTGS\b', r'\bUPI\b',
    ], 0.30, True),  # Very low confidence — generic transfer
]

# Non-income keywords that should suppress income classification
NON_INCOME_PATTERNS = [
    r'\bPAYMENT\b', r'\bPURCHASE\b', r'\bFEE\b', r'\bCHARGE\b',
    r'\bBILL\b', r'\bEMI\b', r'\bINSURANCE\b', r'\bPREMIUM\b',
    r'\bTAX\sPAID\b', r'\bCHALLAN\b',
]


_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from .classifier import ClassificationPipeline
        from .classifier.stages import MerchantDBStage, UPIHeuristicStage, DescriptionRuleStage, RegexRuleStage, LLMFallbackStage, CatchAllStage
        _pipeline = ClassificationPipeline([
            MerchantDBStage(),
            UPIHeuristicStage(),
            DescriptionRuleStage(),
            RegexRuleStage(),
            LLMFallbackStage(),
            CatchAllStage(),
        ])
    return _pipeline

def _derive_tags(category: str, counterparty: str, description: str) -> list:
    """Derive faceted tags from category, counterparty, and description.

    Tags enable multi-dimensional filtering: a Zomato transaction gets
    ["food", "delivery", "zomato"] in addition to its primary category.
    """
    tags = []
    desc_lower = description.lower()

    # Category is always the primary tag
    tags.append(category)

    # Channel tag (how the payment was made)
    if desc_lower.startswith('upi/') or '@' in desc_lower:
        tags.append('upi')
    elif desc_lower.startswith('imps'):
        tags.append('imps')
    elif desc_lower.startswith('neft'):
        tags.append('neft')
    elif desc_lower.startswith('rtgs'):
        tags.append('rtgs')
    elif desc_lower.startswith('nach'):
        tags.append('nach')
    elif 'atm' in desc_lower:
        tags.append('atm')
    elif 'card' in desc_lower or 'pos' in desc_lower:
        tags.append('card')

    # Merchant tag (normalized counterparty)
    if counterparty:
        merchant_tag = re.sub(r'[^a-z0-9]', '', counterparty.lower())[:20]
        if merchant_tag and merchant_tag != category:
            tags.append(merchant_tag)

    # Purpose tags (semantic facets)
    purpose_map = {
        'food': ['delivery', 'restaurant'],
        'grocery': ['essential'],
        'medical': ['healthcare'],
        'travel': ['transport'],
        'shopping': ['online'],
        'entertainment': ['subscription'],
        'investment': ['sip', 'mutual_fund'],
        'trading_deposit': ['stock'],
        'credit_card': ['bill_payment'],
        'insurance': ['premium'],
        'education': ['tuition'],
        'loan_emi': ['loan'],
    }
    if category in purpose_map:
        tags.extend(purpose_map[category])

    # Income tag
    from pipeline.training_schema import INCOME_CATEGORIES
    if category in INCOME_CATEGORIES:
        tags.append('income')

    # Deduplicate while preserving order
    seen = set()
    return [t for t in tags if not (t in seen or seen.add(t))]


CARD_ISSUER_PATTERNS: list[tuple] = [
    (re.compile(r"(?i)ICICI\\s*BANK\\s*CREDIT\\s*CA|icici\\s*bank\\s*card"), "ICICI Credit Card"),
    (re.compile(r"(?i)HDFC\\s*BANK\\s*CREDIT|hdfc\\s*bank\\s*card"), "HDFC Credit Card"),
    (re.compile(r"(?i)SBI\\s*CARD|sbicard|sbi\\s*credit\\s*card"), "SBI Credit Card"),
    (re.compile(r"(?i)AXIS\\s*BANK\\s*CREDIT|axis\\s*bank\\s*card|AXIS.*?CARD"), "Axis Credit Card"),
    (re.compile(r"(?i)AMEX|AMERICAN\\s*EXPRESS"), "Amex"),
    (re.compile(r"(?i)KOTAK\\s*MAHINDRA.*CARD|kotak.*credit"), "Kotak Credit Card"),
    (re.compile(r"(?i)RBL\\s*CARD|rbl.*credit"), "RBL Credit Card"),
    (re.compile(r"(?i)YES\\s*BANK.*CARD|yes.*credit.*card"), "Yes Bank Credit Card"),
    (re.compile(r"(?i)INDUSIND.*CREDIT.*CARD|indusind.*card"), "IndusInd Credit Card"),
    (re.compile(r"(?i)STANDARD\\s*CHARTERED.*CARD|SCB.*CREDIT"), "StanChart Credit Card"),
    (re.compile(r"(?i)HSBC.*CREDIT.*CARD"), "HSBC Credit Card"),
    (re.compile(r"(?i)CITI.*CREDIT.*CARD|CITIBANK.*CARD"), "Citi Credit Card"),
    (re.compile(r"(?i)AU\\s*BANK.*CARD|AU.*CREDIT"), "AU Credit Card"),
    (re.compile(r"(?i)IDFC.*CREDIT|IDFC.*CARD"), "IDFC Credit Card"),
    (re.compile(r"(?i)BOB\\s*CARD|BANK\\s*OF\\s*BARODA.*CARD|bobcard|onecard"), "BOB/OneCard"),
]


def _extract_card_issuer(description: str) -> str:
    """Extract credit card issuer name from transaction description."""
    for pattern, issuer in CARD_ISSUER_PATTERNS:
        if pattern.search(description):
            return issuer
    return ""


def classify_with_rules(
    txn: RawTransaction,
    *,
    learn_merchants: bool = True,
) -> Optional[ClassifiedTransaction]:
    """Apply rule engine. Income rules match credits, expense rules match debits.
    
    Order: Merchant DB lookup → regex rules → catch-all.
    """
    result = _get_pipeline().classify(txn, learn=learn_merchants)
    if result is None:
        return None
    
    counterparty = result.counterparty
    
    # Extract credit card issuer if not already set
    if result.category == 'credit_card' and not counterparty:
        counterparty = _extract_card_issuer(txn.description)
    
    return ClassifiedTransaction(
        raw=txn,
        category=result.category,
        confidence=result.confidence,
        is_income=result.is_income,
        is_expense=result.is_expense,
        counterparty=counterparty,
        rationale=result.rationale,
        income_type=result.income_type or None,
        tags=list(getattr(result, 'tags', [])) or _derive_tags(result.category, result.counterparty, txn.description),
    )


# ─── Recurring Detector ────────────────────────────────────

class RecurringDetector:
    """Detects recurring income patterns across transactions."""
    
    def detect(self, transactions: list[ClassifiedTransaction]) -> list[ClassifiedTransaction]:
        """Find recurring patterns in unclassified credits."""
        # Group by counterparty (sender extracted from description)
        groups = defaultdict(list)
        for ctxn in transactions:
            if ctxn.category == 'unclassified' and ctxn.raw.type == 'credit':
                cp = self._extract_counterparty(ctxn.raw.description)
                groups[cp].append(ctxn)
        
        for cp, group in groups.items():
            if len(group) < 2:
                continue
            
            dates = sorted(tx.raw.date for tx in group)
            amounts = [tx.raw.amount for tx in group]
            
            # Check for monthly pattern
            if self._is_monthly(dates) and self._amount_stable(amounts):
                for tx in group:
                    tx.category = 'rental' if 'rent' in tx.raw.description.lower() else 'recurring_income'
                    tx.confidence = 0.82
                    tx.is_income = True
                    tx.rationale = f'Recurring monthly: {cp} — {len(group)} occurrences'
                    tx.recurring = True
                    tx.counterparty = cp
        
        return transactions
    
    def _extract_counterparty(self, desc: str) -> str:
        """Extract sender name from transaction description."""
        # Credit card issuer detection
        for pattern, issuer in CARD_ISSUER_PATTERNS:
            if pattern.search(desc):
                return issuer
        
        # NEFT: NEFT-SENDER_NAME-BANK
        m = re.search(r'NEFT[-\s]+([A-Za-z0-9\s]+?)[-\s]+', desc)
        if m: return m.group(1).strip()[:40]
        # IMPS: IMPS/SENDER/...
        m = re.search(r'IMPS[-\s/]+([A-Za-z0-9\s]+?)[-\s/]', desc)
        if m: return m.group(1).strip()[:40]
        # UPI: sender@bank
        m = re.search(r'([a-zA-Z0-9_.]+@[a-zA-Z]+)', desc)
        if m: return m.group(1)
        # Fallback: first word
        return desc.split()[0] if desc else 'unknown'

    def _is_monthly(self, dates: list[date], tolerance: int = 5) -> bool:
        """Check if dates are approximately monthly."""
        if len(dates) < 2:
            return False
        for i in range(1, len(dates)):
            delta = abs((dates[i] - dates[i-1]).days)
            if not (25 <= delta <= 35):
                return False
        return True
    
    def _amount_stable(self, amounts: list[float], tolerance: float = 0.05) -> bool:
        """Check if amounts are within tolerance percentage of each other."""
        if not amounts:
            return False
        avg = sum(amounts) / len(amounts)
        return all(abs(a - avg) / avg <= tolerance for a in amounts if avg > 0)


# ─── LLM Classifier (stub) ─────────────────────────────────

class LLMClassifier:
    """Classifies uncertain transactions using the fine-tuned Qwen 0.5B model."""

    def __init__(self):
        self._local = None

    def _get_model(self):
        if self._local is None:
            try:
                from pipeline.llm_classifier import get_llm_classifier
                self._local = get_llm_classifier()
                if not self._local.available:
                    logger.warning("Qwen model not available — transactions will need manual review")
            except Exception as exc:
                logger.warning("Failed to import LLM classifier: %s", exc)
                self._local = False
        return self._local if self._local and self._local is not False else None

    def classify_batch(
        self,
        transactions: list[ClassifiedTransaction],
    ) -> list[ClassifiedTransaction]:
        """Re-classify uncertain transactions using the local Qwen model."""
        model = self._get_model()
        if model is None:
            for tx in transactions:
                if tx.category == "unclassified":
                    tx.rationale = "Needs manual review (LLM not available)"
            return transactions

        # Only classify unclassified or low-confidence transactions
        uncertain = [
            tx for tx in transactions
            if tx.category == "unclassified" or tx.confidence < 0.70
        ]

        if not uncertain:
            return transactions

        for tx in uncertain:
            result = model.classify(
                description=tx.raw.description,
                txn_type=tx.raw.type,
            )
            if result and result.confidence >= 0.50:
                tx.category = result.category
                tx.confidence = result.confidence
                tx.counterparty = result.company_name or tx.counterparty
                tx.rationale = result.rationale
                tx.is_income = result.is_income
            else:
                tx.rationale = "LLM uncertain — needs manual review"

        return transactions


# ─── Pipeline Orchestrator ─────────────────────────────────

def classify_bank_statement(filepath: str) -> ClassificationReport:
    """
    Full classification pipeline:
    1. Parse statement → RawTransaction[]
    2. Rule classifier
    3. Recurring detector  
    4. LLM fallback
    Returns ClassificationReport with stats and classified transactions.
    """
    report = ClassificationReport()
    
    # Step 1: Parse
    raw_txns = _parse_statement(filepath)
    report.total = len(raw_txns)
    report.credits = sum(1 for t in raw_txns if t.type == 'credit')
    report.debits = sum(1 for t in raw_txns if t.type == 'debit')
    
    # Step 2: Rules
    classified = []
    for raw in raw_txns:
        result = classify_with_rules(raw)
        if result:
            classified.append(result)
        else:
            classified.append(ClassifiedTransaction(raw=raw))
    
    report.rule_classified = sum(1 for c in classified if c.category != 'unclassified')
    
    # Step 3: Recurring
    classified = RecurringDetector().detect(classified)
    report.recurring_detected = sum(1 for c in classified if c.recurring)
    
    # Step 4: LLM
    uncertain = [c for c in classified if c.category == 'unclassified' and c.raw.type == 'credit']
    if uncertain:
        classified = LLMClassifier().classify_batch(classified)
    
    report.classified = classified
    report.unclassified = sum(1 for c in classified if c.category == 'unclassified')
    report.income_detected = sum(1 for c in classified if c.is_income)
    
    return report


# ─── Statement Parsers ─────────────────────────────────────

def _parse_statement(filepath: str) -> list[RawTransaction]:
    """Route to appropriate parser based on file extension and content."""
    path = Path(filepath)
    ext = path.suffix.lower()
    
    if ext == '.csv':
        return _parse_csv(path)
    elif ext == '.pdf':
        return _parse_pdf(path)
    elif ext in ('.xls', '.xlsx'):
        return _parse_icici_excel(str(path))
    else:
        # Try CSV first, then PDF
        try:
            return _parse_csv(path)
        except:
            return _parse_pdf(path)

def _parse_icici_excel(filepath: str) -> list[RawTransaction]:
    """Parse ICICI Bank XLS/XLSX statement (JasperReports format).
    
    Auto-detects the column layout — some files have an extra NaN column at index 0.
    """
    import pandas as pd
    df = pd.read_excel(filepath, header=None)
    
    # Detect if there's an extra NaN column at index 0 (Monica/user format)
    col0_is_nan = True
    for i in range(min(15, len(df))):
        if pd.notna(df.iloc[i, 0]):
            col0_is_nan = False
            break
    
    col_offset = 1 if col0_is_nan else 0
    
    # Find header row
    data_start = 8  # default
    for i in range(20):
        v = str(df.iloc[i, col_offset]) if pd.notna(df.iloc[i, col_offset]) else ''
        if v == 'S No.':
            data_start = i + 1
            break
    
    transactions = []
    for i in range(data_start, len(df)):
        # Skip rows without a valid S.No. (continuation lines, footers)
        sno = str(df.iloc[i, col_offset]) if pd.notna(df.iloc[i, col_offset]) else ''
        if not sno.isdigit():
            continue

        desc_col = col_offset + 4
        withdrawal_col = col_offset + 5
        deposit_col = col_offset + 6
        desc = str(df.iloc[i, desc_col]) if pd.notna(df.iloc[i, desc_col]) else ''
        withdrawal = df.iloc[i, withdrawal_col] if pd.notna(df.iloc[i, withdrawal_col]) else 0
        deposit = df.iloc[i, deposit_col] if pd.notna(df.iloc[i, deposit_col]) else 0
        
        if not desc or desc == 'nan':
            continue
        
        try:
            w = float(withdrawal) if withdrawal and str(withdrawal) != 'nan' else 0.0
            d = float(deposit) if deposit and str(deposit) != 'nan' else 0.0
        except (TypeError, ValueError):
            continue
        txn_type = 'credit' if d > 0 else 'debit'
        amount = d if d > 0 else w
        
        date_col = col_offset + 2
        date_str = str(df.iloc[i, date_col]) if pd.notna(df.iloc[i, date_col]) else ''
        try:
            txn_date = pd.to_datetime(date_str, dayfirst=True).date()
        except:
            txn_date = date.today()
        
        transactions.append(RawTransaction(
            date=txn_date, description=desc.strip(),
            type=txn_type, amount=amount
        ))
    
    return sorted(transactions, key=lambda t: t.date)


def _parse_credit_card(filepath: str) -> list[RawTransaction]:
    """Parse a credit card statement (CSV/PDF/XLSX).
    
    Credit card CSVs typically have: Date, Description, Amount (all debits).
    Merging these with bank statements fills expense gaps — the bank only shows
    one bulk payment to the card company, but the card statement has every purchase.
    """
    path = Path(filepath)
    ext = path.suffix.lower()
    
    if ext == '.csv':
        return _parse_credit_card_csv(path)
    elif ext in ('.xls', '.xlsx'):
        return _parse_credit_card_excel(filepath)
    else:
        # Try CSV first
        try:
            return _parse_credit_card_csv(path)
        except:
            return []


def _parse_credit_card_csv(path: Path) -> list[RawTransaction]:
    """Parse credit card CSV. Auto-detects columns."""
    transactions = []
    with open(path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return transactions
        
        cols = [c.lower().strip() for c in reader.fieldnames]
        
        # Detect date column
        date_col = next((c for c in cols if 'date' in c and 'post' not in c), cols[0] if len(cols) > 0 else None)
        # Detect description column
        desc_col = next((c for c in cols if c in ('description','narration','particulars','transaction details','details')), None)
        if not desc_col:
            desc_col = next((c for c in cols if 'desc' in c), cols[1] if len(cols) > 1 else None)
        # Detect amount column
        amt_col = next((c for c in cols if c in ('amount','transaction amount','inr','rs.')), None)
        if not amt_col:
            amt_col = next((c for c in cols if 'amount' in c), cols[2] if len(cols) > 2 else None)
        
        for row in reader:
            try:
                desc = str(row.get(desc_col, '')).strip()
                amt_str = str(row.get(amt_col, '0')).replace(',', '').replace('₹', '').replace('Rs.', '').strip()
                amt = abs(float(amt_str)) if amt_str else 0
                if not desc or amt <= 0:
                    continue
                
                date_str = str(row.get(date_col, ''))
                try:
                    from dateutil.parser import parse as dateparse
                    txn_date = dateparse(date_str).date()
                except:
                    txn_date = date.today()
                
                transactions.append(RawTransaction(date=txn_date, description=desc, type='debit', amount=amt))
            except (ValueError, KeyError):
                continue
    
    return sorted(transactions, key=lambda t: t.date)


def _parse_credit_card_excel(filepath: str) -> list[RawTransaction]:
    """Parse credit card XLS/XLSX. Reads first sheet, auto-detects columns."""
    import pandas as pd
    try:
        df = pd.read_excel(filepath)
    except:
        return []
    
    if df.empty:
        return []
    
    # Normalize column names
    df.columns = [str(c).lower().strip() for c in df.columns]
    cols = list(df.columns)
    
    date_col = next((c for c in cols if 'date' in c and 'post' not in c), cols[0] if cols else None)
    desc_col = next((c for c in cols if c in ('description','narration','particulars')), cols[1] if len(cols) > 1 else None)
    amt_col = next((c for c in cols if 'amount' in c or c in ('inr','rs.')), cols[2] if len(cols) > 2 else None)
    
    if not all([date_col, desc_col, amt_col]):
        return []
    
    transactions = []
    for _, row in df.iterrows():
        try:
            desc = str(row[desc_col]).strip()
            amt = abs(float(str(row[amt_col]).replace(',', '').replace('₹', ''))) if pd.notna(row[amt_col]) else 0
            if not desc or amt <= 0 or desc == 'nan':
                continue
            txn_date = pd.to_datetime(row[date_col]).date() if pd.notna(row[date_col]) else date.today()
            transactions.append(RawTransaction(date=txn_date, description=desc, type='debit', amount=amt))
        except (ValueError, KeyError):
            continue
    
    return sorted(transactions, key=lambda t: t.date)


def _parse_csv(path: Path) -> list[RawTransaction]:
    """Parse a CSV bank statement. Columns: Date, Description, Debit, Credit, Balance."""
    transactions = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                txn_date = _parse_date(row.get('Date', row.get('date', '')))
                desc = row.get('Description', row.get('description', row.get('Narration', '')))
                debit = float(str(row.get('Debit', row.get('debit', '0')).replace(',', '')))
                credit = float(str(row.get('Credit', row.get('credit', '0')).replace(',', '')))
                balance = row.get('Balance', row.get('balance', ''))
                bal = float(str(balance).replace(',', '')) if balance else None
                
                if debit > 0:
                    transactions.append(RawTransaction(date=txn_date, description=desc, 
                                                        type='debit', amount=debit, balance=bal))
                elif credit > 0:
                    transactions.append(RawTransaction(date=txn_date, description=desc,
                                                        type='credit', amount=credit, balance=bal))
            except (ValueError, KeyError):
                continue
    
    return sorted(transactions, key=lambda t: t.date)

def _parse_pdf(path: Path) -> list[RawTransaction]:
    """Parse PDF bank statement using pymupdf. Handles ICICI, SBI, HDFC formats."""
    try:
        import pymupdf
    except ImportError:
        raise ImportError("pymupdf required for PDF parsing. Run: pip install pymupdf")

    with pymupdf.open(str(path)) as doc:
        text = "\n".join(str(page.get_text()) for page in doc)

        # Indie exports lose blank debit/credit cells in plain-text order.
        # Parse visual amount columns while PDF coordinates are available.
        if _is_indie_icici_text(text):
            return _parse_indie_icici_pdf(doc)

    # Detect bank and parse accordingly
    if 'ICICI Bank' in text:
        return _parse_icici_text(text)
    elif 'State Bank of India' in text or 'SBI' in text:
        return _parse_sbi_text(text)
    elif 'HDFC Bank' in text:
        return _parse_hdfc_text(text)
    else:
        return _parse_generic_text(text)


_INDIE_DATE_RE = re.compile(r'^\d{2}\.\d{2}\.\d{4}$')
_INDIE_AMOUNT_RE = re.compile(r'^-?[\d,]+\.\d{2}$')
_INDIE_SNO_RE = re.compile(r'^\d{1,4}$')


def _is_indie_icici_text(text: str) -> bool:
    upper = text.upper()
    return ('STATEMENT OF TRANSACTIONS IN SAVING ACCOUNT' in upper
            or ('ICICI BANK LIMITED' in upper
                and 'WITHDRAWAL' in upper
                and 'DEPOSIT' in upper
                and bool(re.search(r'\d{2}\.\d{2}\.\d{4}', text))))


def _parse_indie_icici_text(text: str) -> list[RawTransaction]:
    """Parse synthetic Indie-style text only when direction is explicit.

    Plain extraction omits blank withdrawal/deposit cells, so a positive amount
    is not evidence of direction. Real PDFs must use `_parse_indie_icici_pdf`.
    """
    lines = [line.strip() for line in text.splitlines()]
    transactions = []
    i = 0
    while i + 1 < len(lines):
        if not (_INDIE_SNO_RE.fullmatch(lines[i])
                and _INDIE_DATE_RE.fullmatch(lines[i + 1])):
            i += 1
            continue

        txn_date = _parse_date(lines[i + 1])
        description_lines = []
        j = i + 2
        while j < len(lines) and not _INDIE_AMOUNT_RE.fullmatch(lines[j]):
            if (_INDIE_SNO_RE.fullmatch(lines[j]) and j + 1 < len(lines)
                    and _INDIE_DATE_RE.fullmatch(lines[j + 1])):
                break
            if lines[j]:
                description_lines.append(lines[j])
            j += 1

        amounts = []
        while j < len(lines) and _INDIE_AMOUNT_RE.fullmatch(lines[j]):
            amounts.append(float(lines[j].replace(',', '')))
            j += 1

        description = ' '.join(description_lines).strip()
        has_credit = bool(re.search(r'\bCREDIT\b', description, re.IGNORECASE))
        has_debit = bool(re.search(r'\bDEBIT\b', description, re.IGNORECASE))
        if len(amounts) >= 2 and description and has_credit != has_debit:
            transactions.append(RawTransaction(
                date=txn_date,
                description=description,
                type='credit' if has_credit else 'debit',
                amount=amounts[0],
                balance=amounts[-1],
            ))
        i = max(j, i + 1)

    return sorted(transactions, key=lambda txn: txn.date)


def _page_text_lines(page) -> list[tuple[float, float, float, float, str]]:
    """Return visual PDF lines as `(x0, y0, x1, y1, text)` tuples."""
    result = []
    for block in page.get_text('dict').get('blocks', []):
        for line in block.get('lines', []):
            text = ''.join(span.get('text', '') for span in line.get('spans', [])).strip()
            if text:
                x0, y0, x1, y1 = line['bbox']
                result.append((x0, y0, x1, y1, text))
    return sorted(result, key=lambda item: (item[1], item[0]))


def _parse_indie_icici_pdf(doc) -> list[RawTransaction]:
    """Parse Indie exports using withdrawal/deposit/balance x-coordinates."""
    transactions = []
    for page in doc:
        lines = _page_text_lines(page)
        date_rows = [line for line in lines if _INDIE_DATE_RE.fullmatch(line[4])]
        width = float(page.rect.width)

        for index, date_line in enumerate(date_rows):
            row_top = date_line[1] - 1.0
            row_bottom = (date_rows[index + 1][1] - 1.0
                          if index + 1 < len(date_rows) else float(page.rect.height))
            row = [line for line in lines if row_top <= line[1] < row_bottom]

            withdrawal = []
            deposit = []
            balances = []
            for x0, _y0, x1, _y1, value in row:
                if not _INDIE_AMOUNT_RE.fullmatch(value) or x0 < width * 0.65:
                    continue
                parsed = float(value.replace(',', ''))
                if x1 <= width * 0.78:
                    withdrawal.append(parsed)
                elif x1 <= width * 0.89:
                    deposit.append(parsed)
                else:
                    balances.append(parsed)

            # Direction comes exclusively from the populated visual amount column.
            has_withdrawal = len(withdrawal) == 1 and withdrawal[0] != 0
            has_deposit = len(deposit) == 1 and deposit[0] != 0
            if has_withdrawal == has_deposit or len(balances) != 1:
                continue

            description_parts = [
                value for x0, _y0, x1, _y1, value in row
                if width * 0.30 <= x0 and x1 < width * 0.67
                and not _INDIE_AMOUNT_RE.fullmatch(value)
            ]
            description = ' '.join(description_parts).strip()
            if not description:
                continue

            transactions.append(RawTransaction(
                date=_parse_date(date_line[4]),
                description=description,
                type='debit' if has_withdrawal else 'credit',
                amount=withdrawal[0] if has_withdrawal else deposit[0],
                balance=balances[0],
            ))

    return sorted(transactions, key=lambda txn: txn.date)


def _parse_icici_text(text: str) -> list[RawTransaction]:
    """Parse ICICI Bank statement text."""
    transactions = []
    # ICICI format: Date | Description | Cheque No | Debit | Credit | Balance
    lines = text.split('\n')
    in_txn_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect transaction section start
        if re.search(r'Date\s+Description\s+.*(?:Debit|Credit)', line):
            in_txn_section = True
            continue
        
        if in_txn_section:
            # Try to match: DD/MM/YYYY Description ... Amount Amount Amount
            match = re.match(r'(\d{2}/\d{2}/\d{4})\s+(.+?)\s+([\d,]+\.?\d*)\s*$', line)
            if not match:
                match = re.match(r'(\d{2}/\d{2}/\d{4})\s+(.+?)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)', line)
            
            if match:
                txn_date = _parse_date(match.group(1))
                desc = match.group(2).strip()
                amounts = [float(g.replace(',', '')) for g in match.groups()[2:] if g]
                
                if len(amounts) >= 2:
                    if amounts[-2] > 0:  # Debit
                        transactions.append(RawTransaction(
                            date=txn_date, description=desc, type='debit', 
                            amount=amounts[-2], balance=amounts[-1] if len(amounts) > 2 else None))
                    elif amounts[-1] > 0:  # Credit
                        transactions.append(RawTransaction(
                            date=txn_date, description=desc, type='credit',
                            amount=amounts[-1], balance=amounts[-2] if len(amounts) > 2 else None))
                elif len(amounts) == 1:
                    txn_type = 'credit' if 'CR' in desc.upper() or 'credit' in desc.lower() else 'debit'
                    transactions.append(RawTransaction(
                        date=txn_date, description=desc, type=txn_type, amount=amounts[0]))
    
    return sorted(transactions, key=lambda t: t.date)

def _parse_sbi_text(text: str) -> list[RawTransaction]:
    """Parse SBI statement. Falls back to generic parser."""
    return _parse_generic_text(text)

def _parse_hdfc_text(text: str) -> list[RawTransaction]:
    """Parse HDFC statement. Falls back to generic parser."""
    return _parse_generic_text(text)

def _parse_generic_text(text: str) -> list[RawTransaction]:
    """Generic parser — looks for date + amount patterns."""
    transactions = []
    for line in text.split('\n'):
        line = line.strip()
        # Look for DD/MM/YYYY or DD-MM-YYYY followed by amount
        match = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4}).*?([\d,]+\.?\d{2})', line)
        if match:
            try:
                txn_date = _parse_date(match.group(1))
                amount = float(match.group(2).replace(',', ''))
            except ValueError:
                continue
            desc = line[:match.start(2)].strip()
            txn_type = 'credit' if ('CR' in line.upper() or amount > 10000) else 'debit'
            transactions.append(RawTransaction(date=txn_date, description=desc, type=txn_type, amount=amount))
    return sorted(transactions, key=lambda t: t.date)


def _parse_date(s: str) -> date:
    """Parse date from various formats."""
    s = s.strip()
    for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%Y-%m-%d', '%d/%m/%y', '%m/%d/%Y']:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s}")


# ─── CLI ───────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bank_classifier.py <statement.pdf|csv>")
        print("  Classifies bank transactions into income categories.")
        sys.exit(1)
    
    report = classify_bank_statement(sys.argv[1])
    
    print(f"Total transactions: {report.total}")
    print(f"Credits: {report.credits} | Debits: {report.debits}")
    print(f"Rule-classified: {report.rule_classified}")
    print(f"Recurring detected: {report.recurring_detected}")
    print(f"Unclassified: {report.unclassified}")
    print(f"Income detected: {report.income_detected}")
    print()
    
    # Show income transactions
    print("=== Income Transactions ===")
    for txn in report.classified:
        if txn.is_income:
            print(f"  {txn.raw.date} | ₹{txn.raw.amount:>10,.2f} | {txn.category:25s} | {txn.confidence:.0%} | {txn.raw.description[:60]}")
    
    # Show unclassified credits
    unclassified = [t for t in report.classified if t.category == 'unclassified' and t.raw.type == 'credit']
    if unclassified:
        print(f"\n=== Unclassified Credits ({len(unclassified)}) — Needs Review ===")
        for txn in unclassified[:20]:
            print(f"  {txn.raw.date} | ₹{txn.raw.amount:>10,.2f} | {txn.raw.description[:80]}")
