"""Local two-mode eval runner — uploads results to LangSmith.

Mode A (default): Tier 1 retrieval regression.
  Loads tests/golden_qa*.json, runs against live pipeline, checks
  refusal accuracy + citation presence + faithfulness.

  python scripts/run_eval_local.py \\
      --playlist-id PLxxx --user-id alice

Mode B (--smoke): Tier 2 structural health check.
  Loads cached smoke probes (generated at ingestion), checks pipeline
  mechanics only — NOT vocabulary-gap retrieval.

  python scripts/run_eval_local.py \\
      --playlist-id PLxxx --user-id alice --smoke

  SMOKE TEST — pipeline health check only.
  Does not test vocabulary-gap retrieval.
  Run without --smoke to verify retrieval quality.

Options:
  --local-only   Print results to stdout; skip LangSmith upload.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()

from yt_rag.config import settings
from yt_rag.evaluation.judges import (
    citation_precision_evaluator,
    citation_presence_evaluator,
    faithfulness_evaluator,
    refusal_accuracy_evaluator,
)
from yt_rag.graph.rag_graph import rag_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_predict():
    def predict(inputs: dict) -> dict:
        state = {
            "question":         inputs["question"],
            "playlist_id":      inputs.get("playlist_id", "default"),
            "user_id":          inputs.get("user_id", "eval_user"),
            "metadata_filter":  None,
            "retrieved_docs":   [],
            "reranked_docs":    [],
            "min_rerank_score": settings.min_rerank_score,
            "threshold_passed": False,
            "refusal_reason":   None,
            "answer":           None,
            "citations":        [],
            "final_response":   None,
        }
        return rag_graph.invoke(state)

    return predict


def _load_regression_examples(playlist_id: str) -> list[dict]:
    """Discover all tests/golden_qa*.json files and filter to the given playlist."""
    examples = []
    for path in sorted(pathlib.Path("tests").glob("golden_qa*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  Warning: could not read {path}: {exc}")
            continue
        for ex in data:
            if ex.get("playlist_id", "") == playlist_id or ex.get("playlist_id", "") == "":
                examples.append(ex)
    return examples


def _load_smoke_examples(playlist_id: str, user_id: str) -> list[dict]:
    from yt_rag.evaluation.probe_generator import generate_smoke_probes, load_smoke_probes

    probes = load_smoke_probes(playlist_id)
    if probes is None:
        print(f"No cached smoke probes found for {playlist_id!r} — generating now…")
        probes = generate_smoke_probes(playlist_id, user_id)
    return probes


def _print_table(results, label: str) -> dict[str, float]:
    """Print a summary table and return aggregate scores."""
    scores: dict[str, list[float]] = {}
    for r in results:
        for key, val in (r.get("evaluation_results") or {}).items():
            scores.setdefault(key, []).append(float(val.get("score", 0)))

    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    agg = {}
    for key, vals in scores.items():
        avg = sum(vals) / len(vals) if vals else 0.0
        agg[key] = avg
        status = "✓" if avg >= 0.9 else ("~" if avg >= 0.7 else "✗")
        print(f"  {status}  {key:<28}  {avg:.2f}  ({len(vals)} examples)")
    return agg


def _run_with_langsmith(examples: list[dict], prefix: str, predict_fn):
    from langsmith.evaluation import evaluate

    inputs = [
        {
            "question":    ex["question"],
            "playlist_id": ex.get("playlist_id", ""),
            "user_id":     ex.get("user_id", ""),
            "query_type":  ex.get("query_type", ""),
        }
        for ex in examples
    ]
    outputs = [
        {"expected_refusal": ex.get("expected_refusal", False)}
        for ex in examples
    ]
    dataset = [{"inputs": i, "outputs": o} for i, o in zip(inputs, outputs)]

    return evaluate(
        predict_fn,
        data=dataset,
        evaluators=[
            faithfulness_evaluator(),
            citation_precision_evaluator(),
            citation_presence_evaluator(),
            refusal_accuracy_evaluator(),
        ],
        experiment_prefix=prefix,
        metadata={"min_rerank_score": settings.min_rerank_score},
    )


def _run_local_only(examples: list[dict], predict_fn) -> dict[str, float]:
    """Run without LangSmith — evaluate locally and print results."""
    rows = []
    for ex in examples:
        inp = {
            "question":    ex["question"],
            "playlist_id": ex.get("playlist_id", ""),
            "user_id":     ex.get("user_id", ""),
        }
        out = predict_fn(inp)
        final = (out.get("final_response") or {})
        refused = bool(final.get("refusal"))
        expected_refuse = ex.get("expected_refusal", False)
        refusal_ok = refused == expected_refuse
        has_citation = len(final.get("citations") or []) >= 1 if not refused else True
        rows.append({
            "question": ex["question"][:55],
            "type": ex.get("query_type", "")[:18],
            "refusal_ok": refusal_ok,
            "citation_ok": has_citation,
        })

    print(f"\n{'─' * 80}")
    print(f"  {'#':<4}  {'Type':<20}  {'Question':<45}  Refusal  Citation")
    print(f"{'─' * 80}")
    for i, r in enumerate(rows, 1):
        ref = "✓" if r["refusal_ok"] else "✗"
        cit = "✓" if r["citation_ok"] else "✗"
        print(f"  {i:<4}  {r['type']:<20}  {r['question']:<45}  {ref}        {cit}")

    total = len(rows)
    refusal_pass = sum(1 for r in rows if r["refusal_ok"])
    citation_pass = sum(1 for r in rows if r["citation_ok"])
    return {
        "refusal_accuracy": refusal_pass / total if total else 0.0,
        "citation_presence": citation_pass / total if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Arclens RAG evaluation")
    parser.add_argument("--playlist-id", required=True, help="Playlist ID to evaluate")
    parser.add_argument("--user-id", required=True, help="User ID for namespace resolution")
    parser.add_argument("--smoke", action="store_true",
                        help="Run Tier 2 structural health check instead of Tier 1 regression")
    parser.add_argument("--local-only", action="store_true",
                        help="Print results locally; do not upload to LangSmith")
    parser.add_argument("--type", dest="query_type", default=None,
                        help="Filter to a specific query_type (Tier 1 only)")
    args = parser.parse_args()

    if args.smoke:
        print("\n=== Arclens RAG — Structural Health Check (Tier 2) ===")
        print("SMOKE TEST — pipeline health check only.")
        print("These probes verify the pipeline runs for this domain.")
        print("They do NOT test vocabulary-gap retrieval.\n")
        examples = _load_smoke_examples(args.playlist_id, args.user_id)
        prefix = f"smoke-{args.playlist_id}"
        pass_thresholds = {"refusal_accuracy": 1.0, "citation_presence": 0.8}
    else:
        print("\n=== Arclens RAG — Retrieval Regression Suite (Tier 1) ===")
        examples = _load_regression_examples(args.playlist_id)
        if not examples:
            print(f"No golden examples found for playlist_id={args.playlist_id!r}.")
            print("Generate one first:")
            print(f"  python scripts/generate_golden_qa.py "
                  f"--playlist-id {args.playlist_id} --user-id {args.user_id}")
            sys.exit(1)
        if args.query_type:
            examples = [e for e in examples if e.get("query_type") == args.query_type]
        prefix = f"regression-{args.playlist_id}"
        pass_thresholds = {"refusal_accuracy": 1.0, "citation_presence": 0.9}

    print(f"Examples: {len(examples)}  |  playlist: {args.playlist_id}")

    predict_fn = _build_predict()

    if args.local_only:
        agg = _run_local_only(examples, predict_fn)
    else:
        results = _run_with_langsmith(examples, prefix, predict_fn)
        # langsmith evaluate returns an EvaluationResults object; summarise scores
        agg = {}
        for r in results._results:  # type: ignore[attr-defined]
            for ev in (r.get("evaluation_results") or {}).values():
                key = ev.get("key", "")
                score = float(ev.get("score", 0))
                agg.setdefault(key, []).append(score)
        agg = {k: sum(v) / len(v) for k, v in agg.items()}

    print(f"\n{'─' * 50}")
    print("  Aggregate scores:")
    overall_pass = True
    for key, threshold in pass_thresholds.items():
        score = agg.get(key, 0.0)
        ok = score >= threshold
        if not ok:
            overall_pass = False
        status = "✓" if ok else "✗"
        print(f"  {status}  {key:<28}  {score:.2f}  (threshold: {threshold:.2f})")
    for key, score in agg.items():
        if key not in pass_thresholds:
            print(f"     {key:<28}  {score:.2f}")

    print(f"\n  Overall: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    if args.smoke:
        print("\n  Run without --smoke to verify vocabulary-gap retrieval quality.")
    if not args.local_only:
        print(f"  Results uploaded to LangSmith under experiment prefix: {prefix}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
