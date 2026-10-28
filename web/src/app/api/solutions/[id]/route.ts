import { NextResponse } from 'next/server';
import { getSolutionCells } from '@/lib/solutions';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const cells = getSolutionCells(id);
  if (!cells) {
    return NextResponse.json({ error: 'Solution not found' }, { status: 404 });
  }
  return NextResponse.json({ cells });
}
