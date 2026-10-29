# Concept notes (知识点)

One Markdown note per AI problem, shown in the workspace's 知识点 tab (Hot 100 problems don't
have one). A note teaches what a beginner needs to know **before** attempting the problem: the
background, the math, the tools. It must not hand over the answer; that is what the Solution tab
is for.

```
knowledge/zh/<task_id>.md      required
knowledge/en/<task_id>.md      optional; the English UI falls back to the zh note
```

`<task_id>` is the problem id (the task file's name in `torch_judge/tasks/`). After editing, run:

```bash
npm run build:knowledge          # writes web/src/lib/knowledge.json
python -m pytest -q scripts/test_build_knowledge.py
```

## Writing

- Formulas in LaTeX: inline `$...$`, display blocks `$$...$$` on their own lines. Prefer a
  formula over a sentence whenever the idea is mathematical, and define every symbol right after
  it. Shapes are written `(B, S, D)`.
- No complete solution code. Short API snippets (`x.max(dim=-1, keepdim=True)`) are fine.
- Keep it to a few minutes of reading. Use `##` section headings, bullets, and tables where they
  help.

Sections, in this order (skip one only when it has nothing to say):

| Section | Content |
| --- | --- |
| 背景 | What it is, why models use it, where it sits in the network |
| 核心公式 | The math, each symbol defined |
| 形状 | How tensor shapes flow through the computation |
| 用到的 PyTorch 操作 | The ops needed, one line each on what they do |
| 常见坑 | Mistakes that fail the tests, and why |
