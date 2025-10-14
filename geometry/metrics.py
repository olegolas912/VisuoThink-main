"""Utilities for extracting and persisting geometry solver metrics."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_STATS_DIR = Path(os.environ.get("GEOMETRY_STATS_DIR", "outputs/geometry/stats"))
_HISTORY_FILENAME = "history.jsonl"
_LATEST_FILENAME = "latest.json"


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


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numeric_answer(answer: str) -> Optional[float]:
    match = _NUMERIC_RE.search(answer)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_metrics(task_dir: Path, transcript: List[dict], atol: float = 1e-2) -> Dict[str, Any]:
    assistant_messages = [msg for msg in transcript if msg.get("role") == "assistant"]
    assistant_turns = len(assistant_messages)
    parsing_errors = sum(1 for msg in transcript if "Parsing error" in msg.get("content", ""))
    execution_errors = sum(1 for msg in transcript if "Execution error" in msg.get("content", ""))

    final_answer = ""
    for msg in reversed(assistant_messages):
        content = msg.get("content", "")
        if "ANSWER:" in content:
            final_answer = content.split("ANSWER:", 1)[-1].strip()
            break

    reference = _load_reference_answer(task_dir)
    numeric_answer = _extract_numeric_answer(final_answer) if final_answer else None
    if reference is not None and numeric_answer is not None:
        correct = abs(reference - numeric_answer) <= atol
    else:
        correct = None

    terminated = any("TERMINATE" in (msg.get("content") or "") for msg in assistant_messages)
    success: bool
    if correct is True:
        success = True
    elif correct is False:
        success = False
    else:
        success = terminated

    total_chars = sum(len(msg.get("content") or "") for msg in assistant_messages)
    avg_chars = total_chars / assistant_turns if assistant_turns else 0.0
    thought_messages = sum("THOUGHT" in (msg.get("content") or "") for msg in assistant_messages)
    action_messages = sum("ACTION" in (msg.get("content") or "") for msg in assistant_messages)

    return {
        "turns": assistant_turns,
        "parsing_errors": parsing_errors,
        "execution_errors": execution_errors,
        "success": success,
        "final_answer": final_answer,
        "reference": reference,
        "numeric_answer": numeric_answer,
        "correct": correct,
        "terminated": terminated,
        "assistant_total_chars": total_chars,
        "assistant_avg_chars": avg_chars,
        "thought_messages": thought_messages,
        "action_messages": action_messages,
    }


def _normalise_path_component(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    return slug.strip("_") or "unknown"


def record_task_metrics(
    task_dir: Path,
    transcript: List[dict],
    model_config: Optional[Dict[str, Any]] = None,
    stats_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    metrics = extract_metrics(task_dir, transcript)
    timestamp = datetime.now(timezone.utc).isoformat()

    model_config = model_config or {}
    model_name = model_config.get("model") or model_config.get("model_path")
    model_api = model_config.get("api_type")
    config_snapshot = {
        key: model_config[key]
        for key in (
            "temperature",
            "max_new_tokens",
            "device",
            "dtype",
            "do_sample",
            "top_p",
            "top_k",
            "repetition_penalty",
            "load_in_8bit",
            "load_in_4bit",
            "chat_template",
        )
        if key in model_config and model_config[key] is not None
    }

    record = {
        "timestamp": timestamp,
        "task_name": task_dir.name,
        "task_parent": task_dir.parent.name if task_dir.parent != task_dir else None,
        "task_path": str(task_dir),
        "model": model_name,
        "model_api": model_api,
        "config": config_snapshot,
        "metrics": metrics,
    }

    stats_root = Path(stats_dir) if stats_dir else DEFAULT_STATS_DIR
    stats_root.mkdir(parents=True, exist_ok=True)

    metrics_path = task_dir / "metrics.json"
    metrics_payload = {
        "timestamp": timestamp,
        "model": model_name,
        "model_api": model_api,
        "config": config_snapshot,
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    history_path = stats_root / _HISTORY_FILENAME
    model_slug = f"{_normalise_path_component(model_api)}__{_normalise_path_component(model_name)}"
    model_history_path = stats_root / f"{model_slug}.jsonl"
    latest_path = stats_root / _LATEST_FILENAME

    line = json.dumps(record, ensure_ascii=False)
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    with model_history_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    latest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "record": record,
        "metrics_path": metrics_path,
        "history_path": history_path,
        "model_history_path": model_history_path,
        "latest_path": latest_path,
    }
