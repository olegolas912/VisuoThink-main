"""Summarise results for geometry solver runs.

Usage:
    scripts/analyze_geometry.py outputs/geometry/llama3.2-vision

The script looks for per-task folders containing an ``output.json``
transcript and prints aggregate metrics: success rate, average turns and
error counts, plus a per-task table with the extracted final answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import Iterable, List, Tuple

from geometry.metrics import extract_metrics


def load_transcripts(root: Path) -> Iterable[Tuple[str, Path, List[dict]]]:
    for output_json in sorted(root.rglob("output.json")):
        task_dir = output_json.parent
        task_name = task_dir.name
        with output_json.open("r", encoding="utf-8") as fh:
            transcript = json.load(fh)
        yield task_name, task_dir, transcript


def summarise(root: Path) -> None:
    runs = list(load_transcripts(root))
    if not runs:
        raise SystemExit(f"No transcripts found under {root}")

    metrics = [(task, extract_metrics(task_dir, transcript)) for task, task_dir, transcript in runs]

    total = len(metrics)
    successes = sum(1 for _, m in metrics if m["success"])
    avg_turns = sum(m["turns"] for _, m in metrics) / total
    avg_exec_errors = sum(m["execution_errors"] for _, m in metrics) / total
    avg_parse_errors = sum(m["parsing_errors"] for _, m in metrics) / total

    print(f"Evaluated tasks: {total}")
    print(f"Successful tasks: {successes} ({successes / total * 100:.1f}%)")
    print(f"Average assistant turns: {avg_turns:.2f}")
    print(f"Average execution errors: {avg_exec_errors:.2f}")
    print(f"Average parsing errors: {avg_parse_errors:.2f}")
    graded = [m for _, m in metrics if m["correct"] is not None]
    if graded:
        accuracy = sum(1 for m in graded if m["correct"]) / len(graded) * 100
        print(f"Answer accuracy (|pred-ref|<=1e-2): {accuracy:.1f}% ({len(graded)} graded)")
    print()
    print("Detailed results:")
    for task, data in metrics:
        status = "success" if data["success"] else "failure"
        answer_display = data["final_answer"] or "-"
        ref = data.get("reference")
        num_ans = data.get("numeric_answer")
        if ref is not None and num_ans is not None:
            correctness = "correct" if data["correct"] else "WRONG"
            detail = f"ref={ref}, pred={num_ans} ({correctness})"
        elif ref is not None:
            detail = f"ref={ref}, pred=n/a"
        else:
            detail = "ref=n/a"
        print(
            f"  {task:>40}: {status}, turns={data['turns']}, "
            f"parse_errors={data['parsing_errors']}, exec_errors={data['execution_errors']}, "
            f"answer={answer_display} | {detail}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise geometry solver outputs")
    parser.add_argument("outputs", type=Path, help="Path to outputs/geometry/... directory")
    args = parser.parse_args()
    summarise(args.outputs)


if __name__ == "__main__":
    main()
