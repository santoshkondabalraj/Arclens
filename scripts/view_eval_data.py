"""View generated eval datasets — golden (Tier 1) and smoke probes (Tier 2).

Usage:
    python scripts/view_eval_data.py
    python scripts/view_eval_data.py --playlist-id PLxxx
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _bar(char: str = "-", width: int = 72) -> str:
    return char * width


def show_golden(playlist_id: str | None) -> None:
    paths = sorted(pathlib.Path("tests").glob("golden_qa_*.json"))
    if not paths:
        print("  No golden datasets found.")
        print("  Generate one with:")
        print("    python scripts/generate_golden_qa.py --playlist-id PLxxx --user-id <you>")
        return

    for path in paths:
        data: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            continue

        pid = data[0].get("playlist_id", path.stem.replace("golden_qa_", ""))
        if playlist_id and pid != playlist_id:
            continue

        print(f"\n  File : {path}")
        print(f"  Playlist : {pid}  ({len(data)} examples)")
        print(f"  {_bar()}")
        print(f"  {'#':<4} {'Type':<22} {'Expect':<8} Question")
        print(f"  {_bar()}")

        by_type: dict[str, list[dict]] = {}
        for ex in data:
            by_type.setdefault(ex.get("query_type", "unknown"), []).append(ex)

        i = 1
        for qtype in ("direct_content", "semantic_synonym", "meta_framing", "off_topic"):
            for ex in by_type.get(qtype, []):
                expect = "REFUSE" if ex.get("expected_refusal") else "answer"
                q = ex["question"]
                if len(q) > 62:
                    q = q[:59] + "..."
                print(f"  {i:<4} {qtype:<22} {expect:<8} {q}")
                if ex.get("vocabulary_note"):
                    print(f"  {'':4} {'':22} {'':8} ^ {ex['vocabulary_note']}")
                i += 1

        # Summary counts
        print(f"  {_bar()}")
        for qtype, exs in sorted(by_type.items()):
            print(f"  {'':4} {qtype:<22} {len(exs)} examples")


def show_smoke(playlist_id: str | None) -> None:
    probe_dir = pathlib.Path("data/eval_probes")
    if not probe_dir.exists() or not list(probe_dir.glob("*.json")):
        print("  No smoke probes found.")
        print("  They are generated automatically during ingestion.")
        return

    for path in sorted(probe_dir.glob("*.json")):
        pid = path.stem
        if playlist_id and pid != playlist_id:
            continue

        data: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n  File : {path}")
        print(f"  Playlist : {pid}  ({len(data)} probes)")
        print(f"  {_bar()}")
        print(f"  {'#':<4} {'Type':<26} {'Expect':<8} Question")
        print(f"  {_bar()}")
        for i, ex in enumerate(data, 1):
            expect = "REFUSE" if ex.get("expected_refusal") else "answer"
            q = ex["question"]
            if len(q) > 56:
                q = q[:53] + "..."
            print(f"  {i:<4} {ex.get('query_type', ''):<26} {expect:<8} {q}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View generated eval datasets")
    parser.add_argument("--playlist-id", default=None, help="Filter to a specific playlist")
    args = parser.parse_args()

    print("\n" + "=" * 74)
    print("  TIER 1 — Golden Retrieval Regression Dataset (Claude Opus generated)")
    print("=" * 74)
    show_golden(args.playlist_id)

    print("\n" + "=" * 74)
    print("  TIER 2 — Structural Smoke Probes (Gemini Flash generated, pipeline health)")
    print("=" * 74)
    show_smoke(args.playlist_id)
    print()


if __name__ == "__main__":
    main()
