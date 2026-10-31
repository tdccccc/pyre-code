// Server only: like solutions.json, the full notes file stays out of client bundles.
// KnowledgeTab fetches one problem's note (and its topic) through /api/knowledge/[id]
import knowledgeData from '@/lib/knowledge.json';
import problemsData from '@/lib/problems.json';
import type { Locale } from '@/lib/i18n';

/** Chinese is always present; English only where it has been written */
export type KnowledgeNote = { zh: string } & Partial<Record<Locale, string>>;

interface TopicData {
  titleZh: string;
  titleEn: string;
  problems: string[];
  overview: KnowledgeNote;
}

/** A topic groups the problems on one concept family under a shared overview */
export interface KnowledgeTopic {
  id: string;
  titleZh: string;
  titleEn: string;
  problems: { id: string; title: string; titleZh: string }[];
  overview: KnowledgeNote;
}

export interface KnowledgeEntry {
  note: KnowledgeNote;
  topic: KnowledgeTopic | null;
}

const { notes, topics } = knowledgeData as {
  notes: Record<string, KnowledgeNote>;
  topics: Record<string, TopicData>;
};

const problemTitles = new Map(
  (problemsData as { problems: { id: string; title: string; titleZh: string }[] }).problems.map((p) => [p.id, p])
);

// build_knowledge.py guarantees each problem is in at most one topic
const topicOf = new Map<string, string>();
for (const [topicId, topic] of Object.entries(topics)) {
  for (const problemId of topic.problems) topicOf.set(problemId, topicId);
}

function getTopic(topicId: string): KnowledgeTopic {
  const { problems, ...rest } = topics[topicId];
  return {
    id: topicId,
    ...rest,
    problems: problems.map((id) => {
      const p = problemTitles.get(id);
      return { id, title: p?.title ?? id, titleZh: p?.titleZh ?? id };
    }),
  };
}

export function getKnowledge(problemId: string): KnowledgeEntry | undefined {
  const note = notes[problemId];
  if (!note) return undefined;
  const topicId = topicOf.get(problemId);
  return { note, topic: topicId ? getTopic(topicId) : null };
}
