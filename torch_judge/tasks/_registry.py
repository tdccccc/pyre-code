"""Auto-discovery registry for task definitions."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}

TASKS: dict[str, dict[str, Any]] = {}
# LeetCode Hot 100 algorithm problems (tasks/hot100/). Kept out of TASKS so the notebook
# judge and AI-track listings never see them; get_task() covers both.
HOT100_TASKS: dict[str, dict[str, Any]] = {}

_pkg_dir = str(Path(__file__).parent)
for _info in pkgutil.iter_modules([_pkg_dir]):
    if _info.name.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__package__}.{_info.name}")
    if hasattr(_mod, "TASK"):
        TASKS[_info.name] = _mod.TASK

for _info in pkgutil.iter_modules([str(Path(_pkg_dir) / "hot100")]):
    if _info.name.startswith("_"):
        continue
    if _info.name in TASKS:
        raise ValueError(f"Hot 100 task id '{_info.name}' collides with an existing task")
    _mod = importlib.import_module(f"{__package__}.hot100.{_info.name}")
    HOT100_TASKS[_info.name] = {**_mod.TASK, "track": "hot100"}


def get_task(task_id: str) -> dict[str, Any] | None:
    return TASKS.get(task_id) or HOT100_TASKS.get(task_id)


def list_tasks() -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        TASKS.items(),
        key=lambda t: DIFFICULTY_ORDER.get(t[1]["difficulty"], 9),
    )
