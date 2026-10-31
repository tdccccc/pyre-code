import { NextResponse } from 'next/server';
import { GRADING_SERVICE_URL } from '@/lib/constants';
import { getSessionToken } from '@/lib/session';

export async function GET() {
  const sessionToken = await getSessionToken();

  if (!sessionToken) {
    return NextResponse.json({ progress: {} });
  }

  const userRes = await fetch(`${GRADING_SERVICE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken }),
  });
  if (!userRes.ok) return NextResponse.json({ progress: {} });

  const { userId } = await userRes.json();
  const progressRes = await fetch(`${GRADING_SERVICE_URL}/progress/${userId}`);
  if (!progressRes.ok) return NextResponse.json({ progress: {} });

  const progress = await progressRes.json();
  return NextResponse.json({ progress });
}
