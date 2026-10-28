import { NextResponse } from 'next/server';
import problems from '@/lib/problems.json';
import starters from '@/lib/starters.json';
import type { Problem } from '@/lib/types';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const problem = (problems.problems as Problem[]).find((p) => p.id === id);
  if (!problem) {
    return NextResponse.json({ error: 'Problem not found' }, { status: 404 });
  }
  if (problem.starters) {
    // Hot 100: the page picks the starter for the active answer mode
    return NextResponse.json({ ...problem, starterCode: problem.starters.core });
  }
  const startersMap = starters as Record<string, string>;
  const body = startersMap[id] || `def ${problem.functionName}(...):\n    pass`;
  const header = `import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport numpy as np\n\n`;
  return NextResponse.json({ ...problem, starterCode: header + body });
}
