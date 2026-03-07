from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluator import normalize_expected_answers


@dataclass
class SearchSample:
    sample_id: str
    data_source: str
    question: str
    expected_answers: list[str]
    raw: dict[str, Any]


def load_records(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path).expanduser().resolve()
    suffix = file_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(file_path).to_dict(orient="records")
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        raise ValueError(f"JSON file must contain a list, got: {type(obj)}")
    if suffix == ".csv":
        return pd.read_csv(file_path).to_dict(orient="records")
    raise ValueError(f"Unsupported file extension: {suffix}")


def record_to_sample(record: dict[str, Any], idx: int) -> SearchSample:
    question = _extract_question(record)
    expected_answers = normalize_expected_answers(_extract_expected(record))
    sample_id = str(record.get("id", f"sample-{idx}"))
    data_source = str(record.get("data_source", "unknown"))
    return SearchSample(
        sample_id=sample_id,
        data_source=data_source,
        question=question,
        expected_answers=expected_answers,
        raw=record,
    )


def _extract_question(record: dict[str, Any]) -> str:
    if "question" in record:
        return str(record["question"])
    if "prompt" in record:
        prompt = record["prompt"]
        if isinstance(prompt, list):
            for turn in prompt:
                if isinstance(turn, dict) and turn.get("role") == "user":
                    return str(turn.get("content", ""))
            if prompt and isinstance(prompt[0], dict):
                return str(prompt[-1].get("content", ""))
        return str(prompt)
    if "extra_info" in record and isinstance(record["extra_info"], dict):
        if "question" in record["extra_info"]:
            return str(record["extra_info"]["question"])
    raise ValueError("Cannot extract question from record")


def _extract_expected(record: dict[str, Any]) -> Any:
    if "expected_answers" in record:
        return record["expected_answers"]
    if "expected_answer" in record:
        return record["expected_answer"]
    if "golden_answers" in record:
        return record["golden_answers"]
    if "reward_model" in record and isinstance(record["reward_model"], dict):
        if "ground_truth" in record["reward_model"]:
            return record["reward_model"]["ground_truth"]
    if "env_kwargs" in record and isinstance(record["env_kwargs"], dict):
        if "ground_truth" in record["env_kwargs"]:
            return record["env_kwargs"]["ground_truth"]
    return None
