from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _prepare_import_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _build_output_payload(result: Any, include_trace: bool) -> dict[str, Any]:
    payload = result.to_dict()
    if not include_trace:
        payload.pop("trace", None)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extracted Search MAS application for inference.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--question", default=None, help="Single question input.")
    parser.add_argument("--input_file", default=None, help="Dataset file path (.parquet/.jsonl/.json/.csv).")
    parser.add_argument("--limit", type=int, default=None, help="Max sample count when --input_file is set.")
    parser.add_argument("--output_file", default=None, help="Save outputs to jsonl.")
    args = parser.parse_args()

    _prepare_import_path()

    from search_mas.apps.factory import build_application
    from search_mas.apps.search.data_utils import load_records, record_to_sample
    from search_mas.core.config import load_yaml_config

    cfg = load_yaml_config(args.config)
    app = build_application(cfg)
    data_cfg = cfg.get("data", {})
    output_cfg = cfg.get("output", {})

    input_file = args.input_file or data_cfg.get("input_path")
    output_file_value = args.output_file or output_cfg.get("prediction_path")
    include_trace = bool(output_cfg.get("include_trace", True))

    if not args.question and not input_file:
        raise ValueError("Provide either --question or --input_file, or set config.data.input_path")

    output_file = Path(output_file_value).expanduser().resolve() if output_file_value else None
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.question:
        result = app.solve(args.question)
        payload = _build_output_payload(result, include_trace=include_trace)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if output_file:
            with output_file.open("w", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(f"saved outputs: {output_file}")
        return

    records = load_records(input_file)
    if args.limit is not None:
        records = records[: args.limit]

    total = 0
    with output_file.open("w", encoding="utf-8") if output_file else _null_context() as f:
        for idx, record in enumerate(records):
            sample = record_to_sample(record, idx)
            result = app.solve(sample.question)
            payload = _build_output_payload(result, include_trace=include_trace)
            payload["sample_id"] = sample.sample_id
            payload["data_source"] = sample.data_source
            payload["expected_answers"] = sample.expected_answers
            print(json.dumps(payload, ensure_ascii=False))
            if f is not None:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            total += 1

    print(f"completed: {total} samples")
    if output_file:
        print(f"saved outputs: {output_file}")


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    main()
