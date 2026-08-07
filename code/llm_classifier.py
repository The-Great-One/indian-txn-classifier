"""
Local LLM classifier using fine-tuned Qwen 0.5B model.

Acts as a targeted fallback — only invoked for transactions the
regex pipeline marks as unclassified or low-confidence (<0.70).
The model runs on CPU and is loaded once at module import time.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PACKAGE_ROOT / "data" / "qwen-merged-0.5b"

SYSTEM_PROMPT = (
    "You are a bank transaction classifier for Indian bank statements. "
    "Given a raw transaction description, infer both its category and the actual company when evidence exists. "
    "Respond with ONLY a JSON object: "
    '{"category": "<category>", "company_name": "<company_or_null>", "is_income": false, "confidence": 0.0}. '
    "Categories: salary, dividend, interest, rental, capital_gains, other_income, "
    "food, grocery, shopping, bills, medical, insurance, tax_payment, credit_card, "
    "personal_transfer, investment, trading_deposit, trading_credit, education, "
    "travel, entertainment, donation, loan_emi, loan_repayment, cash_withdrawal, unclassified. "
    "Use company_name=null for personal transfers or when the company is not supported by the description. "
    "Credits to known employers = salary. UPI to person names = personal_transfer. "
    "Toll/FASTag/NHAI/IHMCL payments = travel. "
    "Refunds/reversals = original category. If truly unknown, category=unclassified, confidence=0.30."
)


@dataclass
class LLMClassification:
    category: str
    company_name: Optional[str]
    is_income: bool
    confidence: float
    rationale: str = ""


class LocalQwenClassifier:
    """Classifies transactions using the fine-tuned Qwen model."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self._model = None
        self._tokenizer = None
        self._model_path = model_path
        self._available = model_path.is_dir()

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("Loading Qwen model from %s", self._model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_path), trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self._model_path),
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="cpu",
            )
            self._model.eval()
            logger.info("Qwen model loaded successfully")
        except Exception as exc:
            logger.warning("Failed to load Qwen model: %s", exc)
            self._available = False

    def classify(
        self,
        description: str,
        txn_type: str = "debit",
    ) -> Optional[LLMClassification]:
        """Classify a single transaction description."""
        if not self._available:
            return None

        self._ensure_loaded()
        if self._model is None:
            return None

        prompt = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Input:\n{description} (type: {txn_type})\n\n"
            f"### Output:\n"
        )

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract JSON from the response
            json_str = response.split("### Output:\n")[-1].strip()
            # Remove any markdown code fences
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            parsed = json.loads(json_str)
            return LLMClassification(
                category=parsed.get("category", "unclassified"),
                company_name=parsed.get("company_name"),
                is_income=bool(parsed.get("is_income", False)),
                confidence=float(parsed.get("confidence", 0.5)),
                rationale=f"Qwen-0.5B fine-tuned",
            )
        except Exception as exc:
            logger.debug("LLM classification failed for '%s': %s", description[:60], exc)
            return None

    def classify_batch(
        self,
        transactions: list[dict],
    ) -> list[Optional[LLMClassification]]:
        """Classify multiple transactions. Each dict must have 'description' and 'type'."""
        results = []
        for txn in transactions:
            results.append(
                self.classify(
                    description=str(txn.get("description", "")),
                    txn_type=str(txn.get("type", "debit")),
                )
            )
        return results


# Singleton
_classifier: Optional[LocalQwenClassifier] = None


def get_llm_classifier() -> LocalQwenClassifier:
    global _classifier
    if _classifier is None:
        _classifier = LocalQwenClassifier()
    return _classifier


def classify_with_llm(description: str, txn_type: str = "debit") -> Optional[LLMClassification]:
    """Convenience function for single classification."""
    return get_llm_classifier().classify(description, txn_type)
