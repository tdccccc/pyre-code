import { NextResponse } from 'next/server';
import { GRADING_SERVICE_URL } from '@/lib/constants';
import { getSessionToken } from '@/lib/session';

/** Save the learner's note on one of their own submissions. */
export async function PUT(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const sessionToken = await getSessionToken();
  if (!sessionToken) return NextResponse.json({ error: 'No session' }, { status: 401 });

  const { note } = await req.json().catch(() => ({ note: undefined }));
  if (typeof note !== 'string') return NextResponse.json({ error: 'Missing note' }, { status: 400 });

  const res = await fetch(`${GRADING_SERVICE_URL}/submissions/${encodeURIComponent(id)}/note`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken, note }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) return NextResponse.json({ error: data.detail ?? 'Failed to save note' }, { status: res.status });
  return NextResponse.json(data);
}
