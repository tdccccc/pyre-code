"""Tests for build_knowledge.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from build_knowledge import OUTPUT, build_knowledge


def _write(root: Path, locale: str, doc_id: str, text: str) -> None:
    (root / locale / doc_id).parent.mkdir(parents=True, exist_ok=True)
    (root / locale / f"{doc_id}.md").write_text(text, encoding="utf-8")


def _topics(root: Path, *topics: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "topics.json").write_text(json.dumps(list(topics)), encoding="utf-8")


def _topic(topic_id: str, problems: list[str]) -> dict:
    return {"id": topic_id, "titleZh": "归一化", "titleEn": "Normalization", "problems": problems}


def test_build_knowledge_matches_committed_json():
    assert json.loads(OUTPUT.read_text(encoding="utf-8")) == build_knowledge()


def test_collects_locales_and_strips(tmp_path):
    _write(tmp_path, "zh", "relu", "\n## 背景")
    _write(tmp_path, "en", "relu", "## Background\n")
    _write(tmp_path, "zh", "gelu", "## 背景")
    assert build_knowledge(tmp_path) == {
        "notes": {
            "gelu": {"zh": "## 背景"},
            "relu": {"zh": "## 背景", "en": "## Background"},
        },
        "topics": {},
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


def test_builds_topic_in_listed_order(tmp_path):
    for task_id in ("layernorm", "batchnorm"):
        _write(tmp_path, "zh", task_id, "x")
    _write(tmp_path, "zh", "topics/normalization", "\n## 总览\n")
    _topics(tmp_path, _topic("normalization", ["layernorm", "batchnorm"]))
    assert build_knowledge(tmp_path)["topics"] == {
        "normalization": {
            "titleZh": "归一化",
            "titleEn": "Normalization",
            "problems": ["layernorm", "batchnorm"],
            "overview": {"zh": "## 总览"},
        }
    }


def test_rejects_topic_without_overview(tmp_path):
    _write(tmp_path, "zh", "layernorm", "x")
    _topics(tmp_path, _topic("normalization", ["layernorm"]))
    with pytest.raises(ValueError, match="no overview"):
        build_knowledge(tmp_path)


def test_rejects_overview_without_topic(tmp_path):
    _write(tmp_path, "zh", "topics/normalization", "x")
    with pytest.raises(ValueError, match="no topic named 'normalization'"):
        build_knowledge(tmp_path)


def test_rejects_topic_problem_without_note(tmp_path):
    _write(tmp_path, "zh", "topics/normalization", "x")
    _topics(tmp_path, _topic("normalization", ["layernorm"]))
    with pytest.raises(ValueError, match="'layernorm' has no note"):
        build_knowledge(tmp_path)


def test_rejects_problem_in_two_topics(tmp_path):
    _write(tmp_path, "zh", "layernorm", "x")
    _write(tmp_path, "zh", "topics/normalization", "x")
    _write(tmp_path, "zh", "topics/basics", "x")
    _topics(tmp_path, _topic("normalization", ["layernorm"]), _topic("basics", ["layernorm"]))
    with pytest.raises(ValueError, match="in both 'normalization' and 'basics'"):
        build_knowledge(tmp_path)
