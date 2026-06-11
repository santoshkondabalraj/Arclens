"""CLI: upload the golden Q&A test set to LangSmith.

Usage:
    python scripts/upload_golden_dataset.py [--path tests/golden_qa.json] [--name yt-rag-golden]
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from yt_rag.evaluation.datasets import upload_golden_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload golden Q&A dataset to LangSmith")
    parser.add_argument("--path", default="tests/golden_qa.json", help="Path to JSON file")
    parser.add_argument("--name", default="yt-rag-golden", help="LangSmith dataset name")
    args = parser.parse_args()

    dataset_id = upload_golden_dataset(args.path, dataset_name=args.name)
    print(f"Dataset ID: {dataset_id}")


if __name__ == "__main__":
    main()
