from __future__ import annotations

import re
import string
from typing import Any


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def extract_answer(solution_str: str) -> str | None:
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def normalize_expected_answers(value: Any) -> list[str]:
    if isinstance(value, dict) and "target" in value:
        value = value["target"]
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def em_check(prediction: str | None, golden_answers: Any) -> bool:
    if prediction is None:
        return False
    candidates = normalize_expected_answers(golden_answers)
    normalized_prediction = normalize_answer(prediction)
    for answer in candidates:
        if normalize_answer(answer) == normalized_prediction:
            return True
    return False


def subem_check(prediction: str | None, golden_answers: Any) -> bool:
    if prediction is None:
        return False
    candidates = normalize_expected_answers(golden_answers)
    normalized_prediction = normalize_answer(prediction)
    for answer in candidates:
        if normalize_answer(answer) in normalized_prediction:
            return True
    return False


def is_search_answer_correct(prediction: str | None, golden_answers: Any, use_substring: bool = False) -> bool:
    if use_substring:
        return subem_check(prediction, golden_answers)
    return em_check(prediction, golden_answers)
