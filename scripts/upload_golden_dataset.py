"""CLI: upload a golden Q&A dataset to LangSmith.

Reads tests/golden_qa*.json (or a specific file) and uploads to LangSmith
as a named dataset so run_eval.py can evaluate against it over time.

Usage:
    python scripts/upload_golden_dataset.py \\
        --playlist-id PLUiw3naVGEDzPs3rTQhDcH8GgPLYzhq6f

    # Use a specific file:
    python scripts/upload_golden_dataset.py \\
        --path tests/golden_qa.json --name yt-rag-golden
"""
from __future__ import annotations

import argparse
import json
import pathlib

from dotenv import load_dotenv

load_dotenv()

from yt_rag.evaluation.datasets import upload_golden_dataset


def _load_examples(path: pathlib.Path, playlist_id: str | None) -> list[dict]:
    data: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    if playlist_id:
        data = [ex for ex in data if ex.get("playlist_id", "") == playlist_id
                or ex.get("playlist_id", "") == ""]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload golden dataset to LangSmith")
    parser.add_argument("--playlist-id", default=None,
                        help="Filter examples to this playlist and set dataset name to "
                             "yt-rag-{playlist_id}")
    parser.add_argument("--path", default=None,
                        help="Explicit JSON file path (default: auto-discover tests/golden_qa*.json)")
    parser.add_argument("--name", default=None,
                        help="LangSmith dataset name (default: yt-rag-{playlist_id} or yt-rag-golden)")
    args = parser.parse_args()

    # Resolve the source file(s)
    if args.path:
        paths = [pathlib.Path(args.path)]
    elif args.playlist_id:
        # Prefer playlist-specific file, fall back to generic
        specific = pathlib.Path(f"tests/golden_qa_{args.playlist_id}.json")
        generic = pathlib.Path("tests/golden_qa.json")
        paths = [specific] if specific.exists() else [generic]
    else:
        paths = sorted(pathlib.Path("tests").glob("golden_qa*.json"))

    examples: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"  Warning: {p} not found, skipping.")
            continue
        examples.extend(_load_examples(p, args.playlist_id))

    if not examples:
        print("No examples found. Generate a golden dataset first:")
        print("  python scripts/generate_golden_qa.py --playlist-id PLxxx --user-id alice")
        return

    dataset_name = args.name or (
        f"yt-rag-{args.playlist_id}" if args.playlist_id else "yt-rag-golden"
    )

    dataset_id = upload_golden_dataset(examples, dataset_name=dataset_name)
    print(f"Dataset ID: {dataset_id}")
    print(f"\nRun evaluation against it:")
    print(f"  python -m yt_rag.evaluation.run_eval --dataset {dataset_name}")


if __name__ == "__main__":
    main()
