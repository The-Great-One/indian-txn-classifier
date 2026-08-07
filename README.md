# Indian Transaction Classifier

Fine-tuned **Qwen2.5-0.5B** model for classifying Indian bank transactions (UPI, NEFT, IMPS, RTGS) into 30+ categories with merchant identification.

## Model on Hugging Face

🤗 [SahilGoel/indian-txn-classifier](https://huggingface.co/SahilGoel/indian-txn-classifier)

The model weights (957MB safetensors), tokenizer, and LoRA adapter are hosted on Hugging Face Hub.

## Categories

The classifier supports 30+ categories:

**Income:** salary, dividend, interest, rental, capital_gains, other_income

**Expenses:** food, grocery, shopping, bills, medical, insurance, tax_payment, credit_card, personal_transfer, investment, trading_deposit, trading_credit, education, travel, entertainment, donation, loan_emi, loan_repayment, cash_withdrawal

**Special:** friends, family, flat_deposit, trading_fees, vehicle_purchase, staff_salary, health_fitness, transfer, unclassified

## How It Works

1. **Rule engine** (70-80% coverage) — regex patterns for known merchants, UPI handles, and bank reference formats
2. **Recurring pattern detector** (10% more) — identifies repeating transactions
3. **Fine-tuned Qwen 0.5B** (remaining) — LLM fallback for uncertain transactions

## Training Data

1,700 privacy-sanitized transaction descriptions with category labels. PII (phone numbers, personal UPI handles, reference numbers) has been redacted.

## Run on Hugging Face Spaces

Try the model in your browser without any setup:
[🤗 Spaces Demo](https://huggingface.co/spaces/SahilGoel/indian-txn-classifier-demo)

Paste any UPI/NEFT/IMPS/RTGS transaction description and get the category, company, income flag, and confidence score.

## Usage (Local)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("SahilGoel/indian-txn-classifier")
tokenizer = AutoTokenizer.from_pretrained("SahilGoel/indian-txn-classifier")

system_prompt = "You are a bank transaction classifier for Indian bank statements. Given a raw transaction description, infer both its category and the actual company when evidence exists. Respond with ONLY a JSON object: {\"category\": \"<category>\", \"company_name\": \"<company_or_null>\", \"is_income\": false, \"confidence\": 0.0}."

input_text = "UPI/zerodhabroking@/HDFC BANK LTD"
prompt = f"### System:\n{system_prompt}\n\n### Input:\n{input_text}\n\n### Output:\n"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Files

- `app.py` — Gradio app for Hugging Face Spaces (run in browser)
- `requirements.txt` — Python deps for the Space
- `code/` — Classifier pipeline (bank parser, rule engine, LLM classifier, merchant DB)
- `model.safetensors` — Fine-tuned model weights (on Hugging Face Hub)
- `lora-adapter/` — LoRA adapter checkpoints (on Hugging Face Hub)

## Supported Banks

ICICI, HDFC, SBI, Axis, Kotak, Yes Bank, Federal Bank, IDFC First, IndusInd, Bank of Baroda, Punjab National, Canara, Union Bank, Unity SFB

## License

MIT