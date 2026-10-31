import type { AiChatMessage } from '@/lib/types';

/** The saved AI help conversation for a problem; [] when there is none or loading fails */
export async function loadAiChat(problemId: string): Promise<AiChatMessage[]> {
  try {
    const res = await fetch(`/api/ai-chats/${encodeURIComponent(problemId)}`);
    return res.ok ? await res.json() : [];
  } catch {
    return [];
  }
}

/** Saves the whole conversation; an empty one deletes it */
export function saveAiChat(problemId: string, messages: AiChatMessage[]) {
  fetch(`/api/ai-chats/${encodeURIComponent(problemId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  }).catch(() => {
    // Best effort: the conversation still shows for this visit
  });
}
