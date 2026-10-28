import { NextResponse } from 'next/server';
import problems from '@/lib/problems.json';
import type { Problem } from '@/lib/types';

export async function GET() {
  // Return problem list without test code (for the list page)
  const list = (problems.problems as Problem[]).map(({
    id, title, titleZh, difficulty, functionName, hint, hintZh, descriptionEn, descriptionZh,
    track, category, categoryZh, order, leetcodeId, leetcodeSlug,
  }) => ({
    id, title, titleZh, difficulty, functionName, hint, hintZh, descriptionEn, descriptionZh,
    track, category, categoryZh, order, leetcodeId, leetcodeSlug,
  }));
  return NextResponse.json({ problems: list, total: list.length });
}
