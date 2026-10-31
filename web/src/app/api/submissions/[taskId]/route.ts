import { NextResponse } from 'next/server';
import { GRADING_SERVICE_URL } from '@/lib/constants';
import { getSessionToken } from '@/lib/session';

export async function GET(_req: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  const sessionToken = await getSessionToken();
  if (!sessionToken) return NextResponse.json([]);

  const userRes = await fetch(`${GRADING_SERVICE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken }),
  });
  if (!userRes.ok) return NextResponse.json([]);

  const { userId } = await userRes.json();
  const res = await fetch(`${GRADING_SERVICE_URL}/submissions/${userId}/${taskId}`);
  if (!res.ok) return NextResponse.json([]);
  return NextResponse.json(await res.json());
}
