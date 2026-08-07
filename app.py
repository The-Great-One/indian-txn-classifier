import gradio as gr
import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "SahilGoel/indian-txn-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model.eval()

SYSTEM_PROMPT = (
    "You are a bank transaction classifier for Indian bank statements. "
    "Given a raw transaction description, infer both its category and the actual company when evidence exists. "
    'Respond with ONLY a JSON object: {"category": "<category>", "company_name": "<company_or_null>", '
    '"is_income": false, "confidence": 0.0}.'
)

CATEGORIES = [
    "salary", "dividend", "interest", "rental", "capital_gains", "other_income",
    "food", "grocery", "shopping", "bills", "medical", "insurance", "tax_payment",
    "credit_card", "personal_transfer", "investment", "trading_deposit",
    "trading_credit", "education", "travel", "entertainment", "donation",
    "loan_emi", "loan_repayment", "cash_withdrawal", "friends", "family",
    "flat_deposit", "trading_fees", "vehicle_purchase", "staff_salary",
    "health_fitness", "transfer", "unclassified",
]

def classify(description: str) -> str:
    if not description.strip():
        return "Enter a transaction description above."
    prompt = f"### System:\n{SYSTEM_PROMPT}\n\n### Input:\n{description.strip()}\n\n### Output:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, temperature=0.1)
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract JSON from output
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            category = result.get("category", "unclassified")
            company = result.get("company_name") or "—"
            is_income = result.get("is_income", False)
            confidence = result.get("confidence", 0.0)
            return f"**Category:** {category}\n**Company:** {company}\n**Is Income:** {'Yes' if is_income else 'No'}\n**Confidence:** {confidence:.0%}"
        except json.JSONDecodeError:
            pass
    return raw.strip()

examples = [
    "UPI/zerodhabroking@/HDFC BANK LTD",
    "NEFT/SALARY/ACME CORP/INR 50000",
    "UPI/swiggybengaluru@/HDFC BANK LTD",
    "IMPS/AJIO/MYNTRA DESIGNS",
    "UPI/Amazon Pay/Amazon Retail India",
    "RTGS/EDUCATION FEE/DELHI PUBLIC SCHOOL",
]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Transaction Description", placeholder="Paste a bank transaction description..."),
    outputs=gr.Markdown(label="Classification"),
    title="Indian Transaction Classifier",
    description="Fine-tuned Qwen2.5-0.5B that classifies Indian bank transactions (UPI/NEFT/IMPS/RTGS) into 30+ categories with merchant identification.",
    examples=examples,
)

if __name__ == "__main__":
    demo.launch()