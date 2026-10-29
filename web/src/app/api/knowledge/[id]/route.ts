import { NextResponse } from 'next/server';
import { getKnowledge } from '@/lib/knowledge';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const note = getKnowledge(id);
  if (!note) {
    return NextResponse.json({ error: 'Knowledge note not found' }, { status: 404 });
  }
  return NextResponse.json(note);
}
