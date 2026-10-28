import solutionsData from '@/lib/solutions.json';
import type { CodeMode, Problem } from '@/lib/types';

interface SolutionCell {
  type: string;
  source: string;
  role: string;
}

/** Reference solution; Hot 100 problems also have a full ACM program for `mode: 'acm'`. */
export function getSolutionCode(problemId: string, mode: CodeMode = 'core'): string {
  const data = (solutionsData as Record<string, { cells: SolutionCell[] }>)[problemId];
  const cells = data?.cells ?? [];
  const role = mode === 'acm' && cells.some((cell) => cell.role === 'solution_acm') ? 'solution_acm' : 'solution';

  return cells
    .filter((cell) => cell.role === role)
    .map((cell) => cell.source)
    .join('\n\n')
    .trim();
}

export function formatTestCode(code: string, functionName: string): string {
  return code
    .replace(/\{fn\}/g, functionName)
    .split('\n')
    .filter((line) => !line.startsWith('import ') && !line.startsWith('from '))
    .join('\n')
    .trim();
}

export function getSampleTests(problem: Problem, limit = 2): Array<{ name: string; code: string }> {
  return problem.tests.slice(0, limit).map((test) => ({
    name: test.name,
    code: formatTestCode(test.code, problem.functionName),
  }));
}
