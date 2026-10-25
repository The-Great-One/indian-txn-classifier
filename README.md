<div align="center">

<img src="icon.png" alt="indian-txn-classifier logo" width="120" height="120" />

# indian-txn-classifier

**Fine-tuned Qwen2.5-0.5B for classifying Indian bank transactions.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/The-Great-One/indian-txn-classifier)](https://github.com/The-Great-One/indian-txn-classifier/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/The-Great-One/indian-txn-classifier)](https://github.com/The-Great-One/indian-txn-classifier/pulls)
[![GitHub stars](https://img.shields.io/github/stars/The-Great-One/indian-txn-classifier?style=social)](https://github.com/The-Great-One/indian-txn-classifier/stargazers)

[Report Bug](https://github.com/The-Great-One/indian-txn-classifier/issues/new?template=bug_report.md) · [Request Feature](https://github.com/The-Great-One/indian-txn-classifier/issues/new?template=feature_request.md)

</div>

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## About

`indian-txn-classifier` is a fine-tuned **Qwen2.5-0.5B** model for classifying
Indian bank transactions (UPI, NEFT, IMPS, RTGS) into 30+ categories with
merchant identification. Model weights (957MB safetensors), tokenizer, and LoRA
adapter are hosted on Hugging Face Hub.

## Features

- ✅ 30+ categories (income, expenses, special)
- ✅ Rule engine (70-80% coverage) + recurring detector + LLM fallback
- ✅ Privacy-sanitized training data (PII redacted)
- ✅ Gradio app for Hugging Face Spaces
- ✅ Supports 14 Indian banks
- 🚧 Expanded category set

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone https://github.com/The-Great-One/indian-txn-classifier.git
cd indian-txn-classifier
pip install -r requirements.txt
```

## Usage

### Run on Hugging Face Spaces

Try the model in your browser:
[🤗 Spaces Demo](https://huggingface.co/spaces/SahilGoel/indian-txn-classifier-demo)

### Usage (Local)

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

### Categories

**Income:** salary, dividend, interest, rental, capital_gains, other_income

**Expenses:** food, grocery, shopping, bills, medical, insurance, tax_payment,
credit_card, personal_transfer, investment, trading_deposit, trading_credit,
education, travel, entertainment, donation, loan_emi, loan_repayment,
cash_withdrawal

**Special:** friends, family, flat_deposit, trading_fees, vehicle_purchase,
staff_salary, health_fitness, transfer, unclassified

### Supported Banks

ICICI, HDFC, SBI, Axis, Kotak, Yes Bank, Federal Bank, IDFC First, IndusInd,
Bank of Baroda, Punjab National, Canara, Union Bank, Unity SFB

### Files

- `app.py` — Gradio app for Hugging Face Spaces
- `requirements.txt` — Python deps for the Space
- `code/` — Classifier pipeline (bank parser, rule engine, LLM classifier, merchant DB)
- `model.safetensors` — Fine-tuned model weights (on Hugging Face Hub)
- `lora-adapter/` — LoRA adapter checkpoints (on Hugging Face Hub)

### Training Data

1,700 privacy-sanitized transaction descriptions with category labels. PII
(phone numbers, personal UPI handles, reference numbers) has been redacted.

### How It Works

1. **Rule engine** (70-80% coverage) — regex patterns for known merchants, UPI handles, bank reference formats
2. **Recurring pattern detector** (10% more) — identifies repeating transactions
3. **Fine-tuned Qwen 0.5B** (remaining) — LLM fallback for uncertain transactions

## Roadmap

- [x] v0.1 – Model + Gradio Space
- [ ] v0.2 – Expanded category set
- [ ] v1.0 – Stable release

See the open issues for the full list of proposed features and known bugs.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

Sahil Goel – [@The-Great-One](https://github.com/The-Great-One) –
sahilgoel187@gmail.com

Project link: [https://github.com/The-Great-One/indian-txn-classifier](https://github.com/The-Great-One/indian-txn-classifier)

## Acknowledgments

- [Hugging Face](https://huggingface.co)
- [Img Shields](https://shields.io)
- [Contributor Covenant](https://www.contributor-covenant.org)
