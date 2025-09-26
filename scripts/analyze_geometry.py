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
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Optional


def load_transcripts(root: Path) -> Iterable[Tuple[str, Path, List[dict]]]:
    for output_json in sorted(root.rglob("output.json")):
        task_dir = output_json.parent
        task_name = task_dir.name
        with output_json.open("r", encoding="utf-8") as fh:
            transcript = json.load(fh)
        yield task_name, task_dir, transcript


def _load_reference_answer(task_dir: Path) -> Optional[float]:
    ex_json = task_dir / "ex.json"
    if not ex_json.is_file():
        return None
    try:
        data = json.loads(ex_json.read_text(encoding="utf-8"))
        label = data.get("ext_info", {}).get("label")
        if label is None:
            return None
        return float(label)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _extract_numeric_answer(answer: str) -> Optional[float]:
    # find first float-looking number
    match = re.search(r"-?\d+(?:\.\d+)?", answer)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_metrics(task_dir: Path, transcript: List[dict], atol: float = 1e-2) -> dict:
    assistant_turns = sum(1 for msg in transcript if msg.get("role") == "assistant")
    parsing_errors = sum(
        1 for msg in transcript if "Parsing error" in msg.get("content", "")
    )
    execution_errors = sum(
        1 for msg in transcript if "Execution error" in msg.get("content", "")
    )

    final_answer = ""
    for msg in reversed(transcript):
        if msg.get("role") == "assistant" and "ANSWER:" in msg.get("content", ""):
            final_answer = msg["content"].split("ANSWER:", 1)[-1].strip()
            break

    reference = _load_reference_answer(task_dir)
    extracted = _extract_numeric_answer(final_answer) if final_answer else None
    if reference is not None and extracted is not None:
        correct = abs(reference - extracted) <= atol
    else:
        correct = None

    # favour ground-truth correctness when available; otherwise fall back to
    # checking whether the agent produced a terminating answer message
    terminated = any(
        msg.get("role") == "assistant" and "TERMINATE" in msg.get("content", "")
        for msg in transcript
    )

    if correct is True:
        success = True
    elif correct is False:
        success = False
    else:
        success = terminated

    return {
        "turns": assistant_turns,
        "parsing_errors": parsing_errors,
        "execution_errors": execution_errors,
        "success": success,
        "final_answer": final_answer,
        "reference": reference,
        "numeric_answer": extracted,
        "correct": correct,
    }


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
