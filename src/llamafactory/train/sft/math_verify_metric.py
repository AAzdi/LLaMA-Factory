# Copyright 2025 the LlamaFactory team.
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

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer

# Try to import math_verify, fallback to basic matching if not available
try:
    import math_verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False
    print("Warning: math_verify library not found. Using basic string matching instead.")


def extract_answer(text: str) -> str:
    """
    Extract the final answer from the model output.
    This function handles common mathematical answer formats.
    """
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
    """
    if not HAS_MATH_VERIFY:
        # Basic fallback comparison
        pred_clean = re.sub(r'[^\w\.\-\+]', '', pred_answer.lower())
        true_clean = re.sub(r'[^\w\.\-\+]', '', true_answer.lower())
        return pred_clean == true_clean
    
    try:
        # Use math_verify for precise mathematical comparison
        return math_verify.compare_answers(pred_answer, true_answer)
    except Exception as e:
        print(f"Math verify error: {e}, falling back to string comparison")
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

        math_correct = 0
        exact_match = 0
        total = len(decoded_preds)

        for pred, label in zip(decoded_preds, decoded_labels):
            # Extract answers from both prediction and label
            pred_answer = extract_answer(pred)
            true_answer = extract_answer(label)
            
            # Math verify comparison
            is_math_correct = math_verify_compare(pred_answer, true_answer)
            if is_math_correct:
                math_correct += 1

            # Exact string match for comparison
            if pred_answer.strip().lower() == true_answer.strip().lower():
                exact_match += 1

            # Store individual results for batch evaluation
            self.score_dict["math_accuracy"].append(float(is_math_correct))
            self.score_dict["exact_match"].append(float(pred_answer.strip().lower() == true_answer.strip().lower()))

        if compute_result:
            return self._dump()


@dataclass 
class ComputeTokenMathVerifyAccuracy:
    r"""
    Compute token-level mathematical accuracy using math_verify for precise comparison.
    This version works with raw token predictions similar to the original ComputeAccuracy.
    """

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"math_token_accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            
            # Calculate token-level accuracy
            token_accuracy = np.mean(pred[label_mask] == label[label_mask])
            self.score_dict["math_token_accuracy"].append(token_accuracy)

        if compute_result:
            return self._dump()
