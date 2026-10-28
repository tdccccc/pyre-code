'use client';

import { CodeEditor } from './CodeEditor';
import { MarkdownContent } from './MarkdownContent';
import { useLocale } from '@/context/LocaleContext';
import { solutionCodeFromCells } from '@/lib/problemContext';
import { useSolutionCells } from '@/lib/solutionCells';
import { useProblemStore } from '@/store/problemStore';

interface SolutionTabProps {
  problemId: string;
}

export function SolutionTab({ problemId }: SolutionTabProps) {
  const { t, locale } = useLocale();
  const codeMode = useProblemStore((s) => s.codeMode);
  const cells = useSolutionCells(problemId);

  if (cells === null) {
    return <div className="p-6 text-sm text-text-3">{t('loadingSolution')}</div>;
  }
  if (cells.length === 0) {
    return <div className="p-6 text-sm text-text-3">{t('noSolution')}</div>;
  }

  const solutionCode = solutionCodeFromCells(cells, codeMode);
  const demoCode = cells.filter((c) => c.role === 'demo').map((c) => c.source).join('\n\n');
  // Hot 100 ships the explanation in both languages; AI problems only have 'explanation'
  const hasZh = cells.some((c) => c.role === 'explanation_zh');
  const explanationRole = locale === 'zh' && hasZh ? 'explanation_zh' : 'explanation';
  const explanations = cells.filter((c) => c.role === explanationRole);

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {solutionCode && (
        <div className="flex flex-col gap-1">
          <p className="text-xs font-medium text-text-2 px-1">Solution</p>
          <div className="rounded-lg overflow-hidden" style={{ border: '1px solid var(--line)' }}>
            <CodeEditor
              value={solutionCode}
              onChange={() => {}}
              readOnly
              height={`${Math.max(120, solutionCode.split('\n').length * 22 + 32)}px`}
              allowParentScrollOnWheel
            />
          </div>
        </div>
      )}
      {explanations.map((c, i) => (
        <div key={i} className="text-sm text-text-2 px-1"><MarkdownContent content={c.source} /></div>
      ))}
      {demoCode && (
        <div className="flex flex-col gap-1">
          <p className="text-xs font-medium text-text-2 px-1">Demo</p>
          <div className="rounded-lg overflow-hidden" style={{ border: '1px solid var(--line)' }}>
            <CodeEditor
              value={demoCode}
              onChange={() => {}}
              readOnly
              height={`${Math.max(80, demoCode.split('\n').length * 22 + 32)}px`}
              allowParentScrollOnWheel
            />
          </div>
        </div>
      )}
    </div>
  );
}
