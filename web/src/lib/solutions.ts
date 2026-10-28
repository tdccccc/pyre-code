// Server only: the full solutions file must not end up in a client bundle. Client components
// fetch one problem's cells with useSolutionCells (lib/solutionCells.ts)
import solutionsData from '@/lib/solutions.json';
import { solutionCodeFromCells, type SolutionCell } from '@/lib/problemContext';
import type { CodeMode } from '@/lib/types';

const solutions = solutionsData as Record<string, { cells: SolutionCell[] }>;

export function getSolutionCells(problemId: string): SolutionCell[] | undefined {
  return solutions[problemId]?.cells;
}

export function getSolutionCode(problemId: string, mode: CodeMode = 'core'): string {
  return solutionCodeFromCells(getSolutionCells(problemId) ?? [], mode);
}
