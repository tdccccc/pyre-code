#!/usr/bin/env python3
"""Build knowledge.json from the concept notes and topic overviews under knowledge/.

    knowledge/{zh,en}/<task_id>.md          one note per AI problem
    knowledge/topics.json                   topics: id, titles, member problems in reading order
    knowledge/{zh,en}/topics/<topic_id>.md  one overview per topic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
OUTPUT = ROOT / "web" / "src" / "lib" / "knowledge.json"
# Chinese is the source language; an English version is optional
LOCALES = ("zh", "en")

sys.path.insert(0, str(ROOT))

from torch_judge.tasks import TASKS


def _collect(knowledge_dir: Path, subdir: str, valid_ids, kind: str) -> dict[str, dict[str, str]]:
    """Id -> {locale: markdown} for the .md files in knowledge/<locale>/<subdir>, each needing zh."""
    docs: dict[str, dict[str, str]] = {}
    for locale in LOCALES:
        for path in sorted((knowledge_dir / locale / subdir).glob("*.md")):
            if path.stem not in valid_ids:
                rel = path.relative_to(knowledge_dir).as_posix()
                raise ValueError(f"knowledge/{rel}: no {kind} named '{path.stem}'")
            docs.setdefault(path.stem, {})[locale] = path.read_text(encoding="utf-8").strip()
    for doc_id, entry in docs.items():
        if "zh" not in entry:
            raise ValueError(f"The {kind} '{doc_id}' has no zh version")
    return dict(sorted(docs.items()))


def _build_topics(knowledge_dir: Path, notes: dict[str, dict[str, str]]) -> dict[str, dict]:
    topics_file = knowledge_dir / "topics.json"
    topics = json.loads(topics_file.read_text(encoding="utf-8")) if topics_file.exists() else []
    overviews = _collect(knowledge_dir, "topics", {t["id"] for t in topics}, "topic")

    built: dict[str, dict] = {}
    owner: dict[str, str] = {}
    for topic in topics:
        topic_id = topic["id"]
        if topic_id in built:
            raise ValueError(f"Topic '{topic_id}' is listed twice")
        if topic_id not in overviews:
            raise ValueError(f"Topic '{topic_id}' has no overview")
        for problem in topic["problems"]:
            if problem not in notes:
                raise ValueError(f"Topic '{topic_id}': problem '{problem}' has no note")
            if problem in owner:
                raise ValueError(f"Problem '{problem}' is in both '{owner[problem]}' and '{topic_id}'")
            owner[problem] = topic_id
        built[topic_id] = {
            "titleZh": topic["titleZh"],
            "titleEn": topic["titleEn"],
            "problems": topic["problems"],
            "overview": overviews[topic_id],
        }
    return built


def build_knowledge(knowledge_dir: Path = KNOWLEDGE_DIR) -> dict[str, dict]:
    """{"notes": task id -> {locale: md}, "topics": topic id -> titles, problems, overview}.

    Notes are for AI problems only; topics keep the order of topics.json.
    """
    notes = _collect(knowledge_dir, "", TASKS, "AI problem")
    return {"notes": notes, "topics": _build_topics(knowledge_dir, notes)}


def main() -> None:
    knowledge = build_knowledge()
    OUTPUT.write_text(json.dumps(knowledge, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Written {len(knowledge['notes'])} knowledge notes and "
        f"{len(knowledge['topics'])} topics to {OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
