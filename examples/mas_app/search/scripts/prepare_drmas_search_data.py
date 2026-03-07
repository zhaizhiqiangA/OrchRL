from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


def _normalize_expected_answers(ground_truth: Any) -> list[str]:
    if isinstance(ground_truth, dict) and "target" in ground_truth:
        ground_truth = ground_truth["target"]
    if ground_truth is None:
        return []
    if isinstance(ground_truth, (list, tuple)):
        return [str(v) for v in ground_truth]
    return [str(ground_truth)]


def _extract_ground_truth(row: pd.Series) -> Any:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return reward_model.get("ground_truth")
    if "golden_answers" in row:
        return row.get("golden_answers")
    return None


def _build_record(row: pd.Series, split: str, idx: int) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    ground_truth = _extract_ground_truth(row)
    expected_answers = _normalize_expected_answers(ground_truth)
    data_source = str(row.get("data_source", "unknown"))
    return {
        "id": f"{data_source}-{split}-{idx}",
        "data_source": data_source,
        "split": split,
        "index": idx,
        "ability": str(row.get("ability", "search")),
        "question": question,
        "expected_answer": expected_answers[0] if expected_answers else "",
        "expected_answers": expected_answers,
        "ground_truth_raw": ground_truth,
    }


def _sample_by_source(df: pd.DataFrame, samples_per_source: int) -> pd.DataFrame:
    if samples_per_source <= 0:
        return df
    sampled: list[pd.DataFrame] = []
    for data_source in df["data_source"].astype(str).unique():
        source_df = df[df["data_source"].astype(str) == data_source]
        sampled.append(source_df.head(min(samples_per_source, len(source_df))))
    return pd.concat(sampled, ignore_index=True)


def _save_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    def _json_default(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)

    with output_path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _resolve_dataset_root(dataset_root_arg: str | None) -> Path | None:
    dataset_root = dataset_root_arg or os.environ.get("HF_HOME")
    if not dataset_root:
        return None
    root = Path(dataset_root).expanduser().resolve()
    if root.exists():
        return root
    print(f"Warning: dataset_root does not exist and will be ignored: {root}")
    return None


def _local_dataset_candidates(dataset_root: Path, repo_id: str) -> list[Path]:
    repo_basename = repo_id.split("/")[-1]
    repo_token = repo_id.replace("/", "--")
    candidates = [
        dataset_root / repo_id,
        dataset_root / repo_basename,
        dataset_root / repo_token,
        dataset_root / "datasets" / repo_id,
        dataset_root / "datasets" / repo_basename,
        dataset_root / "datasets" / repo_token,
        dataset_root / f"datasets--{repo_token}",
        dataset_root / "datasets" / f"datasets--{repo_token}",
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _local_file_candidates(dataset_root: Path, repo_id: str, filename: str) -> list[Path]:
    repo_token = repo_id.replace("/", "--")
    candidates = [path / filename for path in _local_dataset_candidates(dataset_root, repo_id)]
    snapshot_roots = [
        dataset_root / f"datasets--{repo_token}" / "snapshots",
        dataset_root / "datasets" / f"datasets--{repo_token}" / "snapshots",
    ]
    for snapshot_root in snapshot_roots:
        if not snapshot_root.exists():
            continue
        for snapshot_dir in sorted(snapshot_root.glob("*")):
            candidates.append(snapshot_dir / filename)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _resolve_explicit_local_file(path_arg: str | None, split: str) -> Path | None:
    if not path_arg:
        return None
    path = Path(path_arg).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Explicit {split} file does not exist: {path}")
    print(f"Using explicit local {split} file: {path}")
    return path


def _resolve_parquet_file(
    *,
    repo_id: str,
    filename: str,
    dataset_root: Path | None,
    local_only: bool,
) -> Path:
    attempted_local: list[Path] = []
    if dataset_root is not None:
        for local_path in _local_file_candidates(dataset_root, repo_id, filename):
            attempted_local.append(local_path)
            if local_path.exists():
                print(f"Using local dataset file: repo_id={repo_id}, filename={filename}, path={local_path}")
                return local_path

    if local_only:
        attempted = ", ".join(str(path) for path in attempted_local) if attempted_local else "none"
        raise RuntimeError(
            f"Failed to find local dataset file: repo_id={repo_id}, filename={filename}. "
            f"Checked local candidates: {attempted}"
        )

    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    print(f"Downloaded remote dataset file: repo_id={repo_id}, filename={filename}, local_path={downloaded}")
    return Path(downloaded)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DrMAS Search-R1 dataset for standalone MAS inference/validation."
    )
    parser.add_argument("--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train", help="HuggingFace dataset repo id.")
    parser.add_argument("--output_dir", default="~/data/drmas_search_mas", help="Output directory.")
    parser.add_argument("--samples_per_source", type=int, default=30, help="Rows per data_source for test_sampled.")
    parser.add_argument("--save_jsonl", action="store_true", help="Also save jsonl files.")
    parser.add_argument("--use_mirror", action="store_true", help="Use HF mirror endpoint.")
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Local dataset root. If omitted, use $HF_HOME. Local files are preferred when found.",
    )
    parser.add_argument("--local_only", action="store_true", help="Only load parquet files from local paths.")
    parser.add_argument("--train_file", default=None, help="Optional explicit local train.parquet path.")
    parser.add_argument("--test_file", default=None, help="Optional explicit local test.parquet path.")
    args = parser.parse_args()

    dataset_root = _resolve_dataset_root(args.dataset_root)
    if dataset_root is not None:
        print(f"Using dataset_root={dataset_root}")
    else:
        print("No local dataset_root found. Will use remote Hugging Face download if needed.")

    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print("Enabled offline mode: HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1")

    if args.use_mirror:
        if args.local_only:
            print("Warning: --use_mirror is ignored because --local_only is enabled.")
        else:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print("Using HF mirror: https://hf-mirror.com")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = _resolve_explicit_local_file(args.train_file, "train") or _resolve_parquet_file(
        repo_id=args.hf_repo_id,
        filename="train.parquet",
        dataset_root=dataset_root,
        local_only=args.local_only,
    )
    test_file = _resolve_explicit_local_file(args.test_file, "test") or _resolve_parquet_file(
        repo_id=args.hf_repo_id,
        filename="test.parquet",
        dataset_root=dataset_root,
        local_only=args.local_only,
    )

    train_raw = pd.read_parquet(train_file)
    test_raw = pd.read_parquet(test_file)

    train_processed = pd.DataFrame(
        [_build_record(row, split="train", idx=i) for i, row in train_raw.iterrows()]
    )
    test_processed = pd.DataFrame(
        [_build_record(row, split="test", idx=i) for i, row in test_raw.iterrows()]
    )
    test_sampled = _sample_by_source(test_processed, samples_per_source=args.samples_per_source)

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    sampled_path = output_dir / "test_sampled.parquet"

    train_processed.to_parquet(train_path, index=False)
    test_processed.to_parquet(test_path, index=False)
    test_sampled.to_parquet(sampled_path, index=False)

    print(f"train size: {len(train_processed)} -> {train_path}")
    print(f"test size: {len(test_processed)} -> {test_path}")
    print(f"test_sampled size: {len(test_sampled)} -> {sampled_path}")

    if args.save_jsonl:
        train_jsonl = output_dir / "train.jsonl"
        test_jsonl = output_dir / "test.jsonl"
        sampled_jsonl = output_dir / "test_sampled.jsonl"
        _save_jsonl(train_processed, train_jsonl)
        _save_jsonl(test_processed, test_jsonl)
        _save_jsonl(test_sampled, sampled_jsonl)
        print(f"saved jsonl files: {train_jsonl}, {test_jsonl}, {sampled_jsonl}")


if __name__ == "__main__":
    main()
