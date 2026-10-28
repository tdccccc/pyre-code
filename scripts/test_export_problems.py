"""Tests for export_problems.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from export_problems import OUTPUT, build_problem_catalog
from torch_judge.tasks import HOT100_TASKS, TASKS


def test_build_problem_catalog_matches_committed_json():
    expected = build_problem_catalog()
    actual = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert actual == expected


def test_build_problem_catalog_includes_all_registered_tasks():
    data = build_problem_catalog()
    exported_ids = [problem["id"] for problem in data["problems"]]
    assert exported_ids == list(dict.fromkeys(exported_ids))
    assert set(exported_ids) == set(TASKS) | set(HOT100_TASKS)
    assert len(exported_ids) == len(TASKS) + len(HOT100_TASKS)


def test_exported_problem_shape():
    problem = build_problem_catalog()["problems"][0]
    assert set(problem) == {
        "id",
        "title",
        "titleZh",
        "difficulty",
        "functionName",
        "hint",
        "hintZh",
        "descriptionEn",
        "descriptionZh",
        "tests",
    }


def test_build_problem_catalog_recovers_from_malformed_existing_json(tmp_path):
    output = tmp_path / "problems.json"
    output.write_text("<<<<<<< HEAD\nnot json\n", encoding="utf-8")

    data = build_problem_catalog(output)

    exported_ids = [problem["id"] for problem in data["problems"]]
    assert set(exported_ids) == set(TASKS) | set(HOT100_TASKS)
    assert len(exported_ids) == len(TASKS) + len(HOT100_TASKS)


def test_hot100_entries_come_last_in_study_order():
    problems = build_problem_catalog()["problems"]
    hot100 = [p for p in problems if p.get("track") == "hot100"]
    assert problems[-len(hot100):] == hot100
    assert [p["order"] for p in hot100] == list(range(1, len(hot100) + 1))


def test_hot100_entry_has_both_modes_and_hides_large_inputs():
    problem = next(p for p in build_problem_catalog()["problems"] if p["id"] == "two_sum")
    assert set(problem["cases"]) == {"core", "acm"}
    assert set(problem["starters"]) == {"core", "acm"}
    hidden = [c for c in problem["cases"]["core"] if c.get("hidden")]
    assert hidden and all(set(c) == {"name", "hidden"} for c in hidden)
