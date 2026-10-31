import { NextResponse } from 'next/server';
import { GRADING_SERVICE_URL } from '@/lib/constants';
import { getSessionToken } from '@/lib/session';

/** The learner's server-side draft for one problem and answer mode, or null. */
export async function GET(req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const mode = new URL(req.url).searchParams.get('mode') ?? 'core';
  const sessionToken = await getSessionToken();
  if (!sessionToken) return NextResponse.json(null);

  const userRes = await fetch(`${GRADING_SERVICE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken }),
  });
  if (!userRes.ok) return NextResponse.json(null, { status: 502 });

  const { userId } = await userRes.json();
  const res = await fetch(
    `${GRADING_SERVICE_URL}/drafts/${userId}/${encodeURIComponent(taskId)}?mode=${encodeURIComponent(mode)}`,
  );
  if (!res.ok) return NextResponse.json(null, { status: 502 });
  return NextResponse.json(await res.json());
}

export async function PUT(req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const { code, mode, updatedAt } = await req.json().catch(() => ({}));
  if (typeof code !== 'string' || typeof updatedAt !== 'number') {
    return NextResponse.json({ error: 'Missing code or updatedAt' }, { status: 400 });
  }

  // A draft can be a learner's first action, so it creates the anonymous session like submit does
  const existingToken = await getSessionToken();
  const sessionToken = existingToken ?? crypto.randomUUID();
  const userRes = await fetch(`${GRADING_SERVICE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken }),
  });
  if (!userRes.ok) return NextResponse.json({ error: 'Failed to initialize user session' }, { status: 502 });

  const res = await fetch(`${GRADING_SERVICE_URL}/drafts`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken, taskId, mode: mode ?? 'core', code, updatedAt: Math.round(updatedAt) }),
  });
  if (!res.ok) return NextResponse.json({ error: await res.text() }, { status: res.status });

  const response = NextResponse.json({ ok: true });
  if (!existingToken) {
    response.cookies.set('session_token', sessionToken, { httpOnly: true, maxAge: 60 * 60 * 24 * 30 });
  }
  return response;
}
