"""Offline evaluation script for QA dataset metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from rag.offline_evaluation import run_offline_evaluation


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    project_root = resolve_project_root()
    default_dataset = project_root / "data" / "evaluation" / "qa_dataset.json"
    default_processed = project_root / "data" / "processed"

    parser = argparse.ArgumentParser(description="Evaluate QA performance on a dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Path to the QA dataset JSON file. (Path)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed,
        help="Directory containing chunks.pkl and faiss.index. (Path)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top chunks to retrieve for each question. (int)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the embedder, e.g. cpu or cuda. (str)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate answers with the LLM instead of retrieval-only evaluation. (flag)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for model generation when --generate is set. (str)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model name to use for generation. (str)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for model generation. (float)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to save the evaluation report as JSON. (Path)",
    )
    parser.add_argument(
        "--debug-failures",
        type=int,
        default=0,
        help="Print the N worst-performing examples by F1 and retrieval overlap. (int)",
    )
    return parser.parse_args()


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Evaluation summary ===\n")
    print(f"Examples: {summary.get('count')}")
    for key in ["exact_match", "average_f1", "negative_accuracy", "retrieval_hit_rate", "average_retrieval_overlap"]:
        value = summary.get(key)
        if value is None:
            print(f"{key}: n/a")
        else:
            print(f"{key}: {value:.3f}")

    by_type = summary.get("by_type", {}) or {}
    if by_type:
        print("\nBy type:")
        for example_type, metrics in by_type.items():
            print(f"  {example_type}:")
            for metric_name in ["count", "exact_match", "average_f1", "retrieval_hit_rate", "average_retrieval_overlap", "negative_accuracy"]:
                metric_value = metrics.get(metric_name)
                if metric_value is None:
                    metric_text = "n/a"
                elif metric_name == "count":
                    metric_text = str(metric_value)
                else:
                    metric_text = f"{metric_value:.3f}"
                print(f"    {metric_name}: {metric_text}")


def print_debug_examples(details: List[Dict[str, Any]], limit: int) -> None:
    """Print a small set of the worst-performing examples for debugging."""
    if limit <= 0:
        return

    ranked = sorted(
        details,
        key=lambda ex: (
            ex.get("f1") if ex.get("f1") is not None else 1.0,
            ex.get("retrieval_overlap") if ex.get("retrieval_overlap") is not None else 1.0,
        ),
    )

    print(f"\n=== Debug examples (worst {min(limit, len(ranked))} by F1) ===\n")
    for index, example in enumerate(ranked[:limit], start=1):
        question = example.get("question", "")
        reference = example.get("reference", "")
        prediction = example.get("prediction") or ""
        print(f"[{index}] Question: {question}")
        print(f"    Ground truth: {reference}")
        print(f"    Prediction:   {prediction or '(not generated)'}")
        print(
            "    Metrics: "
            f"exact_match={example.get('exact_match')}, "
            f"f1={example.get('f1')}, "
            f"retrieval_overlap={example.get('retrieval_overlap')}, "
            f"retrieval_hit={example.get('retrieval_hit')}"
        )

        context = example.get("retrieved_text", "")
        if context:
            preview = " ".join(context.split())
            if len(preview) > 1200:
                preview = preview[:1200] + "..."
            print(f"    Retrieved context preview:\n{preview}\n")
        else:
            print("    Retrieved context preview: <empty>\n")


def main() -> None:
    args = parse_args()
    if args.generate and not args.api_key:
        raise ValueError("--api-key is required when --generate is set.")

    summary, details = run_offline_evaluation(
        dataset_path=args.dataset,
        processed_dir=args.processed_dir,
        k=args.k,
        device=args.device,
        generate_answers=args.generate,
        api_key=args.api_key,
        model_name=args.model,
        temperature=args.temperature,
    )

    print_summary(summary)
    print_debug_examples(details, args.debug_failures)

    # Save detailed results to a timestamped file inside results/ by default
    if args.output_json is None:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"offline_evaluation-{__import__('datetime').datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    else:
        output_path = args.output_json
        if output_path.is_dir():
            output_path = output_path / "offline_evaluation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "details": details}, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved evaluation report to {output_path}")


if __name__ == "__main__":
    main()
