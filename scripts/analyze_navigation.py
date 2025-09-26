#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load_runs(outputs_dir: Path):
    runs = []
    for child in sorted(outputs_dir.iterdir(), key=lambda p: p.name):
        output_json = child / "output.json"
        if not output_json.is_file():
            continue
        with output_json.open("r", encoding="utf-8") as fh:
            transcript = json.load(fh)
        runs.append((child.name, transcript))
    return runs


def extract_metrics(transcript):
    success = False
    steps_to_goal = None
    execution_errors = 0
    assistant_turns = 0

    for msg in transcript:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            assistant_turns += 1
        if role == "user" and "Execution error" in content:
            execution_errors += 1
        if "TERMINATE" in content:
            success = True
            match = re.search(r"moving (\d+) steps", content)
            if match:
                steps_to_goal = int(match.group(1))

    return {
        "success": success,
        "assistant_turns": assistant_turns,
        "execution_errors": execution_errors,
        "steps_to_goal": steps_to_goal,
    }


def summarise(outputs_dir: Path):
    runs = load_runs(outputs_dir)
    if not runs:
        raise SystemExit(f"No transcripts found in {outputs_dir}")

    summary_rows = []
    for run_id, transcript in runs:
        metrics = extract_metrics(transcript)
        summary_rows.append((run_id, metrics))

    total = len(summary_rows)
    successes = sum(1 for _, m in summary_rows if m["success"])
    success_rate = successes / total * 100.0

    avg_turns = sum(m["assistant_turns"] for _, m in summary_rows) / total
    avg_errors = sum(m["execution_errors"] for _, m in summary_rows) / total
    successful_steps = [m["steps_to_goal"] for _, m in summary_rows if m["steps_to_goal"] is not None]

    print(f"Evaluated tasks: {total}")
    print(f"Successful tasks: {successes} ({success_rate:.1f}%)")
    print(f"Average assistant turns: {avg_turns:.2f}")
    print(f"Average execution errors: {avg_errors:.2f}")
    if successful_steps:
        avg_steps = sum(successful_steps) / len(successful_steps)
        print(f"Average steps on successful runs: {avg_steps:.2f}")
    else:
        print("Average steps on successful runs: n/a")

    print("\nDetailed per-task metrics:")
    for run_id, metrics in summary_rows:
        status = "success" if metrics["success"] else "failure"
        steps = metrics["steps_to_goal"] if metrics["steps_to_goal"] is not None else "-"
        print(
            f"  Task {run_id:>3}: {status}, turns={metrics['assistant_turns']}, "
            f"errors={metrics['execution_errors']}, steps_to_goal={steps}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate visual navigation transcripts.")
    parser.add_argument(
        "outputs",
        type=Path,
        help="Path to the directory containing per-task output folders.",
    )
    args = parser.parse_args()
    summarise(args.outputs)
