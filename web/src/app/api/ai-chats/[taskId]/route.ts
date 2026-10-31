import { NextResponse } from 'next/server';
import { GRADING_SERVICE_URL } from '@/lib/constants';
import { getSessionToken } from '@/lib/session';

/** The learner's saved AI help conversation for one problem ([] for none). */
export async function GET(_req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const sessionToken = await getSessionToken();
  if (!sessionToken) return NextResponse.json([]);

  const userRes = await fetch(`${GRADING_SERVICE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken }),
  });
  if (!userRes.ok) return NextResponse.json([], { status: 502 });

  const { userId } = await userRes.json();
  const res = await fetch(`${GRADING_SERVICE_URL}/ai-chats/${userId}/${encodeURIComponent(taskId)}`);
  if (!res.ok) return NextResponse.json([], { status: 502 });
  return NextResponse.json(await res.json());
}

export async function PUT(req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const { messages } = await req.json().catch(() => ({}));
  if (!Array.isArray(messages)) return NextResponse.json({ error: 'Missing messages' }, { status: 400 });

  // Chatting can be a learner's first action, so it creates the anonymous session like submit does
  const existingToken = await getSessionToken();
  const sessionToken = existingToken ?? crypto.randomUUID();
  const userRes = await fetch(`${GRADING_SERVICE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken }),
  });
  if (!userRes.ok) return NextResponse.json({ error: 'Failed to initialize user session' }, { status: 502 });

  const res = await fetch(`${GRADING_SERVICE_URL}/ai-chats`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken, taskId, messages }),
  });
  if (!res.ok) return NextResponse.json({ error: await res.text() }, { status: res.status });

  const response = NextResponse.json({ ok: true });
  if (!existingToken) {
    response.cookies.set('session_token', sessionToken, { httpOnly: true, maxAge: 60 * 60 * 24 * 30 });
  }
  return response;
}
