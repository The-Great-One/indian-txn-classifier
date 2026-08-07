"""Shared output schema for classifier training and inference."""

CATEGORIES = (
    "salary", "dividend", "interest", "rental", "capital_gains", "other_income",
    "food", "grocery", "shopping", "bills", "medical", "insurance", "tax_payment",
    "credit_card", "personal_transfer", "investment", "trading_deposit",
    "trading_credit", "education", "travel", "entertainment", "donation",
    "loan_emi", "loan_repayment", "cash_withdrawal", "unclassified",
    "friends", "family", "flat_deposit", "trading_fees", "vehicle_purchase",
    "staff_salary", "unclassified_credit", "health_fitness", "transfer",
)

SUPPORTED_CATEGORIES = frozenset(CATEGORIES)

INCOME_CATEGORIES = frozenset({
    "capital_gains", "dividend", "interest", "other_income", "rental", "salary",
})

NON_INCOME_CATEGORIES = frozenset({
    "cash_withdrawal", "family", "friends", "loan_repayment", "personal_transfer",
    "staff_salary", "transfer",
})
