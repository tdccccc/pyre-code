import type { CodeMode } from '@/lib/types';

const DRAFT_KEY_PREFIX = 'pyre-code-draft:';
const DRAFT_TIME_KEY_PREFIX = 'pyre-code-draft-at:';
const MODE_KEY = 'pyre-code-mode';
const PUSH_DELAY_MS = 1000;

/** Editor code plus its edit time (epoch ms); 0 marks a draft saved before times were recorded */
interface CodeDraft {
  code: string;
  updatedAt: number;
}

// Core mode keeps the original key, so drafts saved before ACM mode existed still load
function getDraftSuffix(problemId: string, mode: CodeMode = 'core') {
  return `${problemId}${mode === 'acm' ? ':acm' : ''}`;
}

function readLocalDraft(problemId: string, mode: CodeMode): CodeDraft | null {
  try {
    const suffix = getDraftSuffix(problemId, mode);
    const code = window.localStorage.getItem(DRAFT_KEY_PREFIX + suffix);
    if (code === null) return null;
    return { code, updatedAt: Number(window.localStorage.getItem(DRAFT_TIME_KEY_PREFIX + suffix)) || 0 };
  } catch {
    return null;
  }
}

function writeLocalDraft(problemId: string, mode: CodeMode, draft: CodeDraft) {
  try {
    const suffix = getDraftSuffix(problemId, mode);
    window.localStorage.setItem(DRAFT_KEY_PREFIX + suffix, draft.code);
    window.localStorage.setItem(DRAFT_TIME_KEY_PREFIX + suffix, String(draft.updatedAt));
  } catch {
    // Ignore storage write failures (private mode, quota, etc).
  }
}

async function fetchServerDraft(problemId: string, mode: CodeMode): Promise<CodeDraft | null> {
  try {
    const res = await fetch(`/api/drafts/${encodeURIComponent(problemId)}?mode=${mode}`);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

function pushServerDraft(problemId: string, mode: CodeMode, draft: CodeDraft) {
  // keepalive lets the last push survive the tab closing
  fetch(`/api/drafts/${encodeURIComponent(problemId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...draft, mode }),
    keepalive: true,
  }).catch(() => {
    // The local copy is newer, so the next resolveCodeDraft pushes it again
  });
}

/**
 * The learner's draft for a problem, or null for none. Takes the newer of the
 * browser and server copies and brings the older one up to date, so a draft
 * written on one device opens on another.
 */
export async function resolveCodeDraft(problemId: string, mode: CodeMode): Promise<string | null> {
  const local = readLocalDraft(problemId, mode);
  const server = await fetchServerDraft(problemId, mode);
  if (server && (!local || server.updatedAt > local.updatedAt)) {
    writeLocalDraft(problemId, mode, server);
    return server.code;
  }
  if (local && local.code !== server?.code) pushServerDraft(problemId, mode, local);
  return local?.code ?? null;
}

let pending: { problemId: string; mode: CodeMode; draft: CodeDraft } | null = null;
let pushTimer: ReturnType<typeof setTimeout> | undefined;

/** Saves an edit to the browser now and to the server after typing pauses */
export function saveCodeDraft(problemId: string, code: string, mode: CodeMode = 'core') {
  const draft = { code, updatedAt: Date.now() };
  writeLocalDraft(problemId, mode, draft);
  if (pending && (pending.problemId !== problemId || pending.mode !== mode)) flushCodeDraft();
  pending = { problemId, mode, draft };
  clearTimeout(pushTimer);
  pushTimer = setTimeout(flushCodeDraft, PUSH_DELAY_MS);
}

/** Pushes a waiting edit to the server right away (on leaving the problem or hiding the tab) */
export function flushCodeDraft() {
  clearTimeout(pushTimer);
  if (!pending) return;
  const { problemId, mode, draft } = pending;
  pending = null;
  pushServerDraft(problemId, mode, draft);
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
