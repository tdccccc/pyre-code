# Concept notes (知识点)

One Markdown note per AI problem, shown in the workspace's 知识点 tab (Hot 100 problems don't
have one). A note teaches what a beginner needs to know **before** attempting the problem (the
background, the math, the tools), then what an interviewer asks **beyond** it. It must not hand
over the answer; that is what the Solution tab is for.

Problems on one concept family are grouped into a **topic** with a shared overview. A problem
belongs to at most one topic; its 知识点 tab then gets a switch between the note and the overview.

```
knowledge/zh/<task_id>.md             required
knowledge/en/<task_id>.md             optional; the English UI falls back to the zh note
knowledge/topics.json                 topics: id, titleZh, titleEn, problems in reading order
knowledge/zh/topics/<topic_id>.md     required for every topic in topics.json
knowledge/en/topics/<topic_id>.md     optional
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

Note sections, in this order (skip one only when it has nothing to say):

| Section | Content |
| --- | --- |
| 背景 | What it is, why models use it, where it sits in the network |
| 核心公式 | The math, each symbol defined |
| 形状 | How tensor shapes flow through the computation |
| 用到的 PyTorch 操作 | The ops needed, one line each on what they do |
| 常见坑 | Mistakes that fail the tests, and why |
| 延伸 | Why it is designed this way, its cost (compute, memory, communication), engineering details |
| 面试追问 | 3–5 numbered questions, each a bold question with a short answer on the next line |

延伸 and 面试追问 go past the problem but must not contain its solution code. Comparisons across
the family (BN vs LN, MHA vs GQA) belong in the topic overview, not repeated in each note.

Topic overview sections:

| Section | Content |
| --- | --- |
| 解决什么问题 | The shared problem, and the common formula the methods vary on |
| 对比 | A table of the methods: what differs, cost, where each is used |
| 演进 | How the family evolved and what each step changed |
| 怎么选 | Which method for which setting |
| 题目顺序 | The reading order of topics.json, one line each on why |
| 面试追问 | Cross-cutting questions, same format as in notes |

A topic may add a section for a comparison that needs one (normalization has Pre-LN vs Post-LN).
The member problem list is rendered above the overview, so don't repeat it as links.
