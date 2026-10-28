'use client';

import { useEffect, useState } from 'react';
import type { SolutionCell } from '@/lib/problemContext';

// One request per problem per page load, shared by every component that asks
const cache = new Map<string, Promise<SolutionCell[]>>();

function fetchCells(problemId: string): Promise<SolutionCell[]> {
  let pending = cache.get(problemId);
  if (!pending) {
    pending = fetch(`/api/solutions/${encodeURIComponent(problemId)}`)
      .then((res) => (res.ok ? res.json() : { cells: [] }))
      .then((data: { cells?: SolutionCell[] }) => data.cells ?? []);
    // A failed request may succeed later; don't cache it
    pending.catch(() => cache.delete(problemId));
    cache.set(problemId, pending);
  }
  return pending;
}

/**
 * Solution cells, fetched on demand (the full solutions file stays on the server).
 * `null` while loading; an empty list when the problem has no solution or the request failed.
 */
export function useSolutionCells(problemId: string): SolutionCell[] | null {
  const [loaded, setLoaded] = useState<{ id: string; cells: SolutionCell[] } | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchCells(problemId)
      .then((cells) => !cancelled && setLoaded({ id: problemId, cells }))
      .catch(() => !cancelled && setLoaded({ id: problemId, cells: [] }));
    return () => {
      cancelled = true;
    };
  }, [problemId]);

  return loaded?.id === problemId ? loaded.cells : null;
}
