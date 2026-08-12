"""Profile a representative query run and save cProfile output."""

from __future__ import annotations

import argparse
import cProfile
from datetime import datetime
import pstats
import re
from pathlib import Path
import sys

DEFAULT_QUESTION = "What is the title of the thesis"
DEFAULT_DEVICE = "cpu"
DEFAULT_MODEL = "gpt-4.1-mini"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.rag_chat as rag_chat
import rag.engine as rag_engine


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_profiling_dir(project_root: Path) -> Path:
    return project_root / "results" / "profiling"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug or "query"


class _DummyChoice:
    def __init__(self, content: str):
        self.message = type("Message", (), {"content": content})()


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [_DummyChoice(content)]


class _DummyClient:
    class chat:
        class completions:
            @staticmethod
            def create(model, messages, temperature):
                return _DummyResponse(
                    "Image quality assessment and image fusion for electron tomography"
                )


def _fake_load_openai_client(api_key=None):
    return _DummyClient()


def _fake_generate_answer(client, prompt, model_name="gpt-4.1-mini", temperature=0.0):
    return "Image quality assessment and image fusion for electron tomography"


query_index.load_openai_client = _fake_load_openai_client
query_index.generate_answer = _fake_generate_answer
rag_engine.load_openai_client = _fake_load_openai_client
rag_engine.generate_answer = _fake_generate_answer


def parse_args() -> argparse.Namespace:
    project_root = resolve_project_root()
    parser = argparse.ArgumentParser(
        description="Profile a deterministic query run and save the output under results/profiling."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Question to profile. (str)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=project_root / "data" / "processed",
        help="Directory with chunks.pkl and faiss.index. (Path)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_profiling_dir(project_root),
        help="Directory where profiling reports are written. (Path)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Embedder device, usually cpu for profiling. (str)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top chunks to retrieve. (int)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name used by the query pipeline. (str)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used by the query pipeline. (float)",
    )
    parser.add_argument(
        "--use-metadata-reranking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable metadata-based reranking. (flag)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(args.question)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    profile_stem = f"{slug}-{timestamp}"
    profile_path = output_dir / f"{profile_stem}.prof"
    report_path = output_dir / f"{profile_stem}.txt"

    profiler = cProfile.Profile()
    profiler.enable()
    rag_chat.run(
        question=args.question,
        processed_dir=args.processed_dir,
        k=args.k,
        device=args.device,
        model_name=args.model,
        temperature=args.temperature,
        api_key=None,
        show_top_k=False,
        evaluate_answer_flag=False,
        eval_model=None,
        output_json=None,
        use_metadata_reranking=args.use_metadata_reranking,
    )
    profiler.disable()

    profiler.dump_stats(str(profile_path))
    with report_path.open("w", encoding="utf-8") as fh:
        # Section 1: full view, library and all, sorted by cumulative time.
        stats = pstats.Stats(profiler, stream=fh)
        stats.strip_dirs().sort_stats("cumulative")

        fh.write("=" * 70 + "\n")
        fh.write("SECTION 1 — Top 200 functions by cumulative time (full view)\n")
        fh.write("=" * 70 + "\n\n")
        stats.print_stats(200)

        # Section 2: only our own code, by cumulative time.
        # strip_dirs() removes the path prefix, so match bare module filenames.
        _OWN_MODULES = (
            "embedding.py",
            "engine.py",
            "retriever.py",
            "faiss_index.py",
            "llm.py",
            "prompting.py",
            "utils.py",
            "runtime_evaluation.py",
            "offline_evaluation.py",
            "text_splitter.py",
            "query_index.py",
            "evaluate.py",
            "build_index.py",
            "profile_query.py",
        )
        own_pattern = "(" + "|".join(_OWN_MODULES) + ")"
        fh.write("\n" + "=" * 70 + "\n")
        fh.write("SECTION 2 — Project code only (rag/ and scripts/ functions)\n")
        fh.write("Sorted by cumulative time. Focus here for per-query latency.\n")
        fh.write("=" * 70 + "\n\n")
        stats.sort_stats("cumulative").print_stats(own_pattern)

        fh.write("\n" + "=" * 70 + "\n")
        fh.write("SECTION 3 — Project code only, sorted by self time (tottime)\n")
        fh.write("High tottime = the function body itself is slow (not its callees).\n")
        fh.write("=" * 70 + "\n\n")
        stats.sort_stats("tottime").print_stats(own_pattern)

    print(f"Saved profiling data to {profile_path}")
    print(f"Saved profiling report to {report_path}")


if __name__ == "__main__":
    main()
