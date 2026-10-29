#!/usr/bin/env python3
"""Build knowledge.json from the concept notes in knowledge/{zh,en}/<task_id>.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
OUTPUT = ROOT / "web" / "src" / "lib" / "knowledge.json"
# Chinese is the source language; an English note is optional
LOCALES = ("zh", "en")

sys.path.insert(0, str(ROOT))

from torch_judge.tasks import TASKS


def build_knowledge(knowledge_dir: Path = KNOWLEDGE_DIR) -> dict[str, dict[str, str]]:
    """Task id -> {locale: markdown}. Notes are for AI problems only, and each needs a zh version."""
    notes: dict[str, dict[str, str]] = {}
    for locale in LOCALES:
        for path in sorted((knowledge_dir / locale).glob("*.md")):
            if path.stem not in TASKS:
                raise ValueError(f"knowledge/{locale}/{path.name}: no AI problem named '{path.stem}'")
            notes.setdefault(path.stem, {})[locale] = path.read_text(encoding="utf-8").strip()
    for task_id, entry in notes.items():
        if "zh" not in entry:
            raise ValueError(f"Knowledge note '{task_id}' has no zh version")
    return dict(sorted(notes.items()))


def main() -> None:
    notes = build_knowledge()
    OUTPUT.write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Written {len(notes)} knowledge notes to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
