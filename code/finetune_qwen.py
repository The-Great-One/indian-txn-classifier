#!/usr/bin/env python3
"""Continue fine-tuning Qwen2.5-0.5B for category and company inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pipeline.augment_training_data import sanitize_training_description
from pipeline.company_inference import infer_company_name
from pipeline.training_schema import CATEGORIES, INCOME_CATEGORIES, NON_INCOME_CATEGORIES

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = PACKAGE_ROOT / "data" / "training_data.json"
OUTPUT_DIR = PACKAGE_ROOT / "data" / "qwen-lora-adapter-0.5b"

SYSTEM_PROMPT = (
    "You are a bank transaction classifier for Indian bank statements. "
    "Given a raw transaction description, infer both its category and the actual company when evidence exists. "
    "Respond with ONLY a JSON object: "
    '{"category": "<category>", "company_name": "<company_or_null>", "is_income": false, "confidence": 0.0}. '
    f"Categories: {', '.join(CATEGORIES)}. "
    "Use company_name=null for personal transfers or when the company is not supported by the description. "
    "Credits to known employers = salary. UPI to person names = personal_transfer. "
    "Refunds/reversals = original category. If truly unknown, category=unclassified, confidence=0.30."
)


def format_training_example(item: dict) -> dict[str, str]:
    """Create one category + company prompt/completion training pair."""
    description = item["description"]
    category = item["category"]
    if "is_income" in item:
        is_income = bool(item["is_income"])
    elif category in NON_INCOME_CATEGORIES:
        is_income = False
    else:
        is_income = item.get("type") == "credit" or category in INCOME_CATEGORIES
    company_name = infer_company_name(
        description,
        category=category,
        explicit_name=item.get("company_name") or item.get("merchant") or item.get("counterparty"),
    )
    sanitized_description = sanitize_training_description(
        description,
        category=category,
        company_name=company_name,
    )
    prompt = f"### System:\n{SYSTEM_PROMPT}\n\n### Input:\n{sanitized_description}\n\n### Output:\n"
    completion = json.dumps({
        "category": category,
        "company_name": company_name,
        "is_income": is_income,
        "confidence": 0.90,
    })
    return {"prompt": prompt, "completion": completion}


def prepare_training_examples(data: list[dict]) -> list[dict[str, str]]:
    """Deduplicate sanitized prompts and reject contradictory completions."""
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for item in data:
        example = format_training_example(item)
        grouped.setdefault(example["prompt"], {})[example["completion"]] = example
    return [
        next(iter(grouped[prompt].values()))
        for prompt in sorted(grouped)
        if len(grouped[prompt]) == 1
    ]


def balance_training_examples(
    examples: list[dict[str, str]],
    *,
    income_target: int = 20,
) -> list[dict[str, str]]:
    """Oversample represented income classes after conflict-safe deduplication."""
    by_category: dict[str, list[dict[str, str]]] = {}
    for example in examples:
        category = json.loads(example["completion"])["category"]
        by_category.setdefault(category, []).append(example)

    balanced = list(examples)
    for category in sorted(INCOME_CATEGORIES):
        category_examples = by_category.get(category, [])
        if not category_examples or len(category_examples) >= income_target:
            continue
        balanced.extend(
            category_examples[index % len(category_examples)]
            for index in range(income_target - len(category_examples))
        )
    return balanced


def load_training_data():
    """Load, sanitize, deduplicate, balance, and format labeled transactions."""
    from datasets import Dataset

    with open(DATA_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    return Dataset.from_list(balance_training_examples(prepare_training_examples(data)))


def _load_trainable_model(*, fresh: bool):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="mps",
        trust_remote_code=True,
    )
    adapter_file = OUTPUT_DIR / "adapter_model.safetensors"
    if adapter_file.exists() and not fresh:
        print(f"Continuing from adapter: {OUTPUT_DIR}")
        return PeftModel.from_pretrained(base_model, str(OUTPUT_DIR), is_trainable=True)

    print("Starting a fresh LoRA adapter")
    return get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )


def main(*, epochs: float = 2.0, fresh: bool = False) -> None:
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = _load_trainable_model(fresh=fresh)
    model.print_trainable_parameters()

    print("Loading training data...")
    dataset = load_training_data()
    company_labels = sum(
        json.loads(completion)["company_name"] is not None
        for completion in dataset["completion"]
    )
    print(f"Training samples: {len(dataset)}; company labels: {company_labels}")

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-4 if not fresh else 2e-4,
            warmup_ratio=0.05,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=False,
            fp16=False,
            optim="adamw_torch",
            report_to="none",
            max_length=512,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting continued training..." if not fresh else "Starting training...")
    trainer.train()
    print(f"Saving LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Done! LoRA adapter saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args()
    main(epochs=arguments.epochs, fresh=arguments.fresh)
