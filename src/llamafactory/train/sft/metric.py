# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available

# Try to import math_verify, fallback to basic matching if not available
try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()


def extract_answer(text: str) -> str:
    """
    Extract the final answer from the model output.
    This function handles common mathematical answer formats.
    """
    import re
    
    # Remove whitespace
    text = text.strip()
    
    # Try to find answer patterns (customize based on your data format)
    patterns = [
        r"答案是[:：]?\s*([^\n]*)",  # Chinese: 答案是
        r"Answer[:：]?\s*([^\n]*)",  # English: Answer
        r"因此[:：]?\s*([^\n]*)",    # Chinese: 因此
        r"Therefore[:：]?\s*([^\n]*)",  # English: Therefore
        r"所以[:：]?\s*([^\n]*)",    # Chinese: 所以
        r"So[:：]?\s*([^\n]*)",     # English: So
        r"\$([^$]*)\$",            # LaTeX format: $...$
        r"\\boxed\{([^}]*)\}",     # LaTeX boxed format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no pattern found, return the last line or the whole text
    lines = text.split('\n')
    return lines[-1].strip() if lines else text


def math_verify_compare(pred_answer: str, true_answer: str) -> bool:
    """
    Compare predicted answer with true answer using math_verify library.
    Falls back to string comparison if math_verify is not available.
    
    Args:
        pred_answer: The predicted answer string
        true_answer: The ground truth answer string
        
    Returns:
        bool: True if answers are mathematically equivalent, False otherwise
    """
    import re
    
    if not HAS_MATH_VERIFY:
        # Basic fallback comparison
        pred_clean = re.sub(r'[^\w\.\-\+]', '', pred_answer.lower())
        true_clean = re.sub(r'[^\w\.\-\+]', '', true_answer.lower())
        return pred_clean == true_clean
    
    try:
        # Use math_verify for precise mathematical comparison
        # Parse both gold (true answer) and predicted answer
        gold_parsed = parse(true_answer)
        pred_parsed = parse(pred_answer)
        
        # Order is important: verify(gold, answer)
        return verify(gold_parsed, pred_parsed)
    except Exception as e:
        print(f"Math verify error for '{pred_answer}' vs '{true_answer}': {e}")
        print("Falling back to string comparison")
        # Fallback to basic comparison
        pred_clean = re.sub(r'[^\w\.\-\+]', '', pred_answer.lower())
        true_clean = re.sub(r'[^\w\.\-\+]', '', true_answer.lower())
        return pred_clean == true_clean


@dataclass
class ComputeMathVerifyAccuracy:
    r"""
    Compute mathematical accuracy using math_verify library for precise answer comparison.
    Supports batch_eval_metrics and handles various mathematical answer formats.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"math_accuracy": [], "exact_match": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        # Decode predictions and labels
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            # Extract answers from both prediction and label
            pred_answer = extract_answer(pred)
            true_answer = extract_answer(label)
            
            # Math verify comparison
            is_math_correct = math_verify_compare(pred_answer, true_answer)
            
            # Exact string match for comparison
            is_exact_match = pred_answer.strip().lower() == true_answer.strip().lower()

            # Store individual results for batch evaluation
            self.score_dict["math_accuracy"].append(float(is_math_correct))
            self.score_dict["exact_match"].append(float(is_exact_match))

        if compute_result:
            return self._dump()
