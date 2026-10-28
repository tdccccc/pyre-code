import type { CodeMode } from '@/lib/types';

const DRAFT_KEY_PREFIX = 'pyre-code-draft:';
const MODE_KEY = 'pyre-code-mode';

// Core mode keeps the original key, so drafts saved before ACM mode existed still load
function getDraftKey(problemId: string, mode: CodeMode = 'core') {
  return `${DRAFT_KEY_PREFIX}${problemId}${mode === 'acm' ? ':acm' : ''}`;
}

export function loadCodeDraft(problemId: string, mode?: CodeMode): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage.getItem(getDraftKey(problemId, mode));
  } catch {
    return null;
  }
}

export function saveCodeDraft(problemId: string, code: string, mode?: CodeMode) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(getDraftKey(problemId, mode), code);
  } catch {
    // Ignore storage write failures (private mode, quota, etc).
  }
}

/** The learner's preferred Hot 100 answer mode, shared across problems */
export function loadCodeMode(): CodeMode {
  try {
    return window.localStorage.getItem(MODE_KEY) === 'acm' ? 'acm' : 'core';
  } catch {
    return 'core';
  }
}

export function saveCodeMode(mode: CodeMode) {
  try {
    window.localStorage.setItem(MODE_KEY, mode);
  } catch {
    // Ignore storage write failures.
  }
}
