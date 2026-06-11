"""CLI: generate a golden retrieval test dataset for a playlist using Claude Opus.

Reads body chunks from the playlist's BM25 index and prompts Claude Opus to produce
16 diverse examples: direct-content, semantic-synonym, meta-framing, and off-topic.
Synonym examples deliberately use different vocabulary from the source chunks —
they are the core regression tests for vocabulary-gap retrieval.

Usage:
    python scripts/generate_golden_qa.py \\
        --playlist-id PLUiw3naVGEDzPs3rTQhDcH8GgPLYzhq6f \\
        --user-id santosh244ster

    # Write to a custom path instead of tests/golden_qa_{playlist_id}.json:
    python scripts/generate_golden_qa.py \\
        --playlist-id PLxxx --user-id alice \\
        --output tests/golden_qa.json

Requires ANTHROPIC_API_KEY in the environment (or in .env).
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

from dotenv import load_dotenv

load_dotenv()

from yt_rag.evaluation.golden_generator import _golden_path, generate_golden_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate golden retrieval dataset via Claude Opus"
    )
    parser.add_argument("--playlist-id", required=True, help="Playlist ID")
    parser.add_argument("--user-id", required=True, help="User who ingested the playlist")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: tests/golden_qa_{playlist_id}.json)",
    )
    args = parser.parse_args()

    examples = generate_golden_dataset(args.playlist_id, args.user_id)

    if args.output:
        dest = pathlib.Path(args.output)
        src = _golden_path(args.playlist_id)
        if dest != src:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)
            print(f"Also copied to: {dest}")

    print(f"\nGenerated {len(examples)} examples.")
    print("Review vocabulary_note fields on semantic_synonym entries before using as ground truth.")


if __name__ == "__main__":
    main()
