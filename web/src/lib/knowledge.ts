// Server only: like solutions.json, the full notes file stays out of client bundles.
// KnowledgeTab fetches one problem's note through /api/knowledge/[id]
import knowledgeData from '@/lib/knowledge.json';
import type { Locale } from '@/lib/i18n';

/** Chinese is always present; English only where it has been written */
export type KnowledgeNote = { zh: string } & Partial<Record<Locale, string>>;

const knowledge = knowledgeData as Record<string, KnowledgeNote>;

export function getKnowledge(problemId: string): KnowledgeNote | undefined {
  return knowledge[problemId];
}
