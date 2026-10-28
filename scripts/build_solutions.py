#!/usr/bin/env python3
"""Build solutions.json from torch_judge task definitions for the frontend."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "web" / "src" / "lib" / "solutions.json"

sys.path.insert(0, str(ROOT))

from torch_judge.algo import spec as algo_spec
from torch_judge.tasks import HOT100_TASKS, TASKS, list_tasks


def build_solution_entry(task: dict) -> dict:
    cells = [{"type": "code", "source": task["solution"].strip(), "role": "solution"}]
    if "explanation" in task:
        cells.append({"type": "markdown", "source": task["explanation"].strip(), "role": "explanation"})
    if "demo" in task:
        cells.append({"type": "code", "source": task["demo"].strip(), "role": "demo"})
    return {"cells": cells}


def build_hot100_solution_entry(task: dict) -> dict:
    """Core solution, the generated ACM program, and the explanation in both languages."""
    entry = build_solution_entry({k: task[k] for k in ("solution", "explanation") if k in task})
    entry["cells"].insert(1, {"type": "code", "source": algo_spec.acm_solution(task).strip(), "role": "solution_acm"})
    if "explanation_zh" in task:
        entry["cells"].append({"type": "markdown", "source": task["explanation_zh"].strip(), "role": "explanation_zh"})
    return entry


def main():
    result = {}
    for task_id, task in list_tasks():
        if "solution" not in task:
            print(f"WARNING: {task_id} has no solution field, skipping")
            continue
        result[task_id] = build_solution_entry(task)
    for task_id in sorted(HOT100_TASKS):
        result[task_id] = build_hot100_solution_entry(HOT100_TASKS[task_id])

    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written {len(result)} solutions to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
