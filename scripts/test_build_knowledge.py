"""Tests for build_knowledge.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from build_knowledge import OUTPUT, build_knowledge


def _write(root: Path, locale: str, task_id: str, text: str) -> None:
    (root / locale).mkdir(parents=True, exist_ok=True)
    (root / locale / f"{task_id}.md").write_text(text, encoding="utf-8")


def test_build_knowledge_matches_committed_json():
    assert json.loads(OUTPUT.read_text(encoding="utf-8")) == build_knowledge()


def test_collects_locales_and_strips(tmp_path):
    _write(tmp_path, "zh", "relu", "\n## 背景")
    _write(tmp_path, "en", "relu", "## Background\n")
    _write(tmp_path, "zh", "gelu", "## 背景")
    assert build_knowledge(tmp_path) == {
        "gelu": {"zh": "## 背景"},
        "relu": {"zh": "## 背景", "en": "## Background"},
    }


@pytest.mark.parametrize("task_id", ["not_a_task", "two_sum"])
def test_rejects_unknown_and_hot100_tasks(tmp_path, task_id):
    _write(tmp_path, "zh", task_id, "x")
    with pytest.raises(ValueError, match=task_id):
        build_knowledge(tmp_path)


def test_rejects_note_without_zh(tmp_path):
    _write(tmp_path, "en", "relu", "x")
    with pytest.raises(ValueError, match="no zh version"):
        build_knowledge(tmp_path)
