from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _prepare_import_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Black-box validation for extracted Search MAS application.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--input_file", default=None, help="Override config.data.input_path")
    parser.add_argument("--prediction_file", default=None, help="Override config.output.prediction_path")
    parser.add_argument("--report_file", default=None, help="Override config.output.report_path")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N samples only.")
    args = parser.parse_args()

    _prepare_import_path()

    from tqdm import tqdm

    from search_mas.apps.factory import build_application
    from search_mas.apps.search.data_utils import load_records, record_to_sample
    from search_mas.apps.search.evaluator import (
        is_search_answer_correct,
        normalize_answer,
    )
    from search_mas.core.config import load_yaml_config

    config = load_yaml_config(args.config)
    app = build_application(config)

    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    val_cfg = config.get("validation", {})
    use_substring_em = bool(val_cfg.get("use_substring_em", False))

    input_file = args.input_file or data_cfg.get("input_path")
    if not input_file:
        raise ValueError("Missing input file. Set --input_file or config.data.input_path")

    prediction_file = Path(
        args.prediction_file or output_cfg.get("prediction_path", "./outputs/search_predictions.jsonl")
    ).expanduser().resolve()
    report_file = Path(
        args.report_file or output_cfg.get("report_path", "./outputs/search_report.json")
    ).expanduser().resolve()
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    include_trace = bool(output_cfg.get("include_trace", False))
    records = load_records(input_file)
    if args.limit is not None:
        records = records[: args.limit]

    total = 0
    correct = 0
    source_totals: dict[str, int] = defaultdict(int)
    source_corrects: dict[str, int] = defaultdict(int)
    mismatch_types: dict[str, int] = defaultdict(int)

    with prediction_file.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(tqdm(records, desc="Validating")):
            sample = record_to_sample(record, idx)
            result = app.solve(sample.question)

            is_correct = is_search_answer_correct(
                result.final_answer,
                sample.expected_answers,
                use_substring=use_substring_em,
            )
            score = 1.0 if is_correct else 0.0

            total += 1
            correct += int(is_correct)
            source_totals[sample.data_source] += 1
            source_corrects[sample.data_source] += int(is_correct)

            mismatch_type = _mismatch_type(result.final_answer, is_correct)
            mismatch_types[mismatch_type] += 1

            row = {
                "sample_id": sample.sample_id,
                "data_source": sample.data_source,
                "question": sample.question,
                "expected_answers": sample.expected_answers,
                "predicted_answer": result.final_answer,
                "final_response": result.final_response,
                "approved": result.approved,
                "score": score,
                "is_correct": is_correct,
                "mismatch_type": mismatch_type,
                "normalized_prediction": normalize_answer(result.final_answer or ""),
                "normalized_expected_answers": [normalize_answer(ans) for ans in sample.expected_answers],
            }
            if include_trace:
                row["trace"] = [step.__dict__ for step in result.trace]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    accuracy = (correct / total) if total > 0 else 0.0
    source_accuracy = {
        k: {
            "correct": source_corrects[k],
            "total": source_totals[k],
            "accuracy": source_corrects[k] / source_totals[k],
        }
        for k in sorted(source_totals.keys())
    }

    report = {
        "input_file": str(Path(input_file).expanduser().resolve()),
        "prediction_file": str(prediction_file),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "use_substring_em": use_substring_em,
        "mismatch_breakdown": dict(sorted(mismatch_types.items())),
        "source_accuracy": source_accuracy,
    }
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


def _mismatch_type(predicted_answer: str | None, is_correct: bool) -> str:
    if is_correct:
        return "correct"
    if predicted_answer is None:
        return "missing_answer_tag"
    return "wrong_value"


if __name__ == "__main__":
    main()
