'use client';

import { useState } from 'react';
import { Lightbulb, ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { CaseBlock } from '@/components/workspace/CaseBlock';
import { MarkdownContent } from '@/components/workspace/MarkdownContent';
import { useLocale } from '@/context/LocaleContext';
import { useProblemStore } from '@/store/problemStore';
import type { Problem } from '@/lib/types';

function parseInline(text: string): (string | JSX.Element)[] {
  const parts: (string | JSX.Element)[] = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    const codeMatch = remaining.match(/`(.+?)`/);
    const boldIdx = boldMatch?.index ?? Infinity;
    const codeIdx = codeMatch?.index ?? Infinity;

    if (boldIdx === Infinity && codeIdx === Infinity) {
      parts.push(remaining);
      break;
    }

    if (boldIdx <= codeIdx && boldMatch) {
      if (boldIdx > 0) parts.push(remaining.slice(0, boldIdx));
      parts.push(<strong key={key++} className="font-medium text-text">{boldMatch[1]}</strong>);
      remaining = remaining.slice(boldIdx + boldMatch[0].length);
    } else if (codeMatch) {
      if (codeIdx! > 0) parts.push(remaining.slice(0, codeIdx!));
      parts.push(
        <code
          key={key++}
          className="mono text-[12.5px] px-[5px] py-px rounded text-text"
          style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
        >
          {codeMatch[1]}
        </code>
      );
      remaining = remaining.slice(codeIdx! + codeMatch[0].length);
    }
  }
  return parts;
}

interface DescriptionTabProps {
  problem: Problem;
}

export function DescriptionTab({ problem }: DescriptionTabProps) {
  const [hintOpen, setHintOpen] = useState(false);
  const { locale, t } = useLocale();
  const codeMode = useProblemStore((s) => s.codeMode);

  const description = locale === 'zh' ? problem.descriptionZh : problem.descriptionEn;
  const hint = locale === 'zh' && problem.hintZh ? problem.hintZh : problem.hint;
  const isHot100 = problem.track === 'hot100';
  const isAcm = isHot100 && codeMode === 'acm';
  // Design problems (LRUCache, ...) name a class; everything else is a Solution method
  const isClass = /^[A-Z]/.test(problem.functionName);

  const task = !isHot100
    ? t('implementFn', { fn: problem.functionName })
    : isAcm
      ? t('acmTask')
      : t(isClass ? 'implementClass' : 'implementMethod', { fn: problem.functionName });
  const sample = problem.cases?.acm[0];

  return (
    <div className="px-7 py-6 space-y-6">
      <div>
        <div className="flex items-center gap-3 mb-2">
          <h1 className="text-[22px] tracking-[-0.02em] font-semibold">
            {isHot100 && <span className="text-text-3 font-normal mr-2 tabular-nums">{problem.leetcodeId}.</span>}
            {locale === 'zh' ? problem.titleZh : problem.title}
          </h1>
          <Badge variant={problem.difficulty.toLowerCase() as 'easy' | 'medium' | 'hard'}>
            {problem.difficulty.toUpperCase()}
          </Badge>
        </div>
        {isHot100 && (
          <div className="flex items-center gap-3 mb-2 text-xs text-text-3">
            <span>{locale === 'zh' ? problem.categoryZh : problem.category}</span>
            <a
              href={`https://${locale === 'zh' ? 'leetcode.cn' : 'leetcode.com'}/problems/${problem.leetcodeSlug}/`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 hover:text-accent transition-colors"
            >
              {t('viewOnLeetcode')}
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        )}
        <p className="text-sm text-text-2">{task}</p>
      </div>

      {description && <MarkdownContent content={description} />}

      {isAcm && problem.acm && (
        <div className="space-y-4">
          <div className="space-y-1">
            <h3 className="text-sm font-medium text-text">{t('inputFormat')}</h3>
            <MarkdownContent content={locale === 'zh' ? problem.acm.inputZh : problem.acm.inputEn} />
          </div>
          <div className="space-y-1">
            <h3 className="text-sm font-medium text-text">{t('outputFormat')}</h3>
            <MarkdownContent content={locale === 'zh' ? problem.acm.outputZh : problem.acm.outputEn} />
          </div>
          {sample?.input != null && (
            <div className="grid grid-cols-2 gap-3 max-[700px]:grid-cols-1">
              <CaseBlock label={t('sampleInput')} text={sample.input} />
              <CaseBlock label={t('sampleOutput')} text={sample.expected ?? ''} />
            </div>
          )}
        </div>
      )}

      {hint && (
        <div>
          <button
            onClick={() => setHintOpen(!hintOpen)}
            className="flex items-center gap-2 text-sm text-text-2 hover:text-accent transition-colors"
          >
            <Lightbulb className="w-4 h-4" />
            <span>{t('hint')}</span>
            {hintOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>
          {hintOpen && (
            <div
              className="mt-2 p-3 px-3.5 rounded-[9px] text-sm text-text-2 leading-relaxed space-y-1"
              style={{
                background: 'color-mix(in oklab, var(--accent) 4%, var(--bg))',
                border: '1px solid var(--accent-line)',
                borderLeft: '3px solid var(--accent)',
              }}
            >
              <span className="mono text-[10.5px] tracking-[0.12em] uppercase text-accent font-semibold block mb-1">⚑ HINT</span>
              {/* Hints are pseudo-code sketches: keep their indentation */}
              {hint.split('\n').map((line, i) => (
                <p key={i} className="whitespace-pre-wrap">{parseInline(line)}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
