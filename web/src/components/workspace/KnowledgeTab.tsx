'use client';

import { useEffect, useState } from 'react';
import { MarkdownContent } from './MarkdownContent';
import { useLocale } from '@/context/LocaleContext';
import type { KnowledgeNote as Note } from '@/lib/knowledge';

// The tab unmounts when hidden; keep each note so switching back doesn't refetch
const cache = new Map<string, Promise<Note | null>>();

function fetchNote(problemId: string): Promise<Note | null> {
  let pending = cache.get(problemId);
  if (!pending) {
    pending = fetch(`/api/knowledge/${encodeURIComponent(problemId)}`)
      .then((res) => (res.ok ? res.json() : null));
    pending.catch(() => cache.delete(problemId));
    cache.set(problemId, pending);
  }
  return pending;
}

interface KnowledgeTabProps {
  problemId: string;
}

export function KnowledgeTab({ problemId }: KnowledgeTabProps) {
  const { t, locale } = useLocale();
  const [loaded, setLoaded] = useState<{ id: string; note: Note | null } | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchNote(problemId)
      .then((note) => !cancelled && setLoaded({ id: problemId, note }))
      .catch(() => !cancelled && setLoaded({ id: problemId, note: null }));
    return () => {
      cancelled = true;
    };
  }, [problemId]);

  if (loaded?.id !== problemId) {
    return <div className="p-6 text-sm text-text-3">{t('loadingKnowledge')}</div>;
  }
  if (!loaded.note) {
    return <div className="p-6 text-sm text-text-3">{t('noKnowledge')}</div>;
  }
  return (
    <div className="px-7 py-6">
      <MarkdownContent content={loaded.note[locale] ?? loaded.note.zh} />
    </div>
  );
}
