'use client';

import { useEffect, useState } from 'react';
import { ArrowLeft, Check, CheckCircle2, ChevronRight, Copy, RotateCcw, XCircle } from 'lucide-react';
import { useProblemStore } from '@/store/problemStore';
import { useLocale } from '@/context/LocaleContext';
import { CodeEditor, CodeDiffViewer } from './CodeEditor';
import { cn } from '@/lib/utils';
import type { SubmissionHistory as Submission } from '@/lib/types';

const actionClass =
  'inline-flex items-center gap-1.5 h-7 px-2 rounded-md text-xs text-text-2 hover:text-text hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)] transition-colors cursor-pointer disabled:opacity-40 disabled:pointer-events-none';

const segmentClass =
  'px-2.5 h-[24px] rounded-[6px] text-xs text-text-2 cursor-pointer transition-[background,color] duration-150';

/* The grading service stores SQLite datetime('now'): UTC without a zone marker, which
   browsers would otherwise read as local time. */
function parseSubmittedAt(value: string): Date {
  return new Date(/Z$|[+-]\d{2}:\d{2}$/.test(value) ? value : `${value.replace(' ', 'T')}Z`);
}

function StatusLabel({ passed }: { passed: boolean }) {
  const { t } = useLocale();
  const Icon = passed ? CheckCircle2 : XCircle;
  return (
    <span className={cn('inline-flex items-center gap-1.5 text-sm font-medium', passed ? 'text-easy' : 'text-hard')}>
      <Icon className="w-4 h-4" />
      {passed ? t('submissionAccepted') : t('submissionWrong')}
    </span>
  );
}

export function SubmissionHistory() {
  const { t, locale } = useLocale();
  const { submissionHistory } = useProblemStore();
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const formatTime = (value: string) =>
    parseSubmittedAt(value).toLocaleString(locale === 'zh' ? 'zh-CN' : 'en-US', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });

  if (!submissionHistory.length) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-text-3">
        {t('submissionsEmpty')}
      </div>
    );
  }

  // History is newest first, so attempt numbers count down
  const selectedIndex = submissionHistory.findIndex((s) => s.id === selectedId);
  if (selectedIndex >= 0) {
    return (
      <SubmissionDetail
        submission={submissionHistory[selectedIndex]}
        attempt={submissionHistory.length - selectedIndex}
        time={formatTime(submissionHistory[selectedIndex].submittedAt)}
        onBack={() => setSelectedId(null)}
      />
    );
  }

  return (
    <div className="overflow-y-auto h-full">
      {submissionHistory.map((s, index) => (
        <button
          key={s.id}
          type="button"
          onClick={() => setSelectedId(s.id)}
          className="w-full flex items-center gap-3 px-4 py-3 text-left transition-colors cursor-pointer hover:bg-[color-mix(in_oklab,var(--text)_3%,transparent)]"
          style={{ borderBottom: '1px solid var(--line)' }}
        >
          <div className="flex-1 min-w-0 space-y-1">
            <div className="flex items-center gap-3">
              <StatusLabel passed={s.passed} />
              <span className="text-xs text-text-3">{t('submissionAttempt', { n: submissionHistory.length - index })}</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-text-3">
              <span>{formatTime(s.submittedAt)}</span>
              {s.execTimeMs != null && <span>· {s.execTimeMs.toFixed(1)}ms</span>}
              <span>· {t('submissionLines', { n: s.code.split('\n').length })}</span>
            </div>
          </div>
          <ChevronRight className="w-4 h-4 text-text-3 flex-shrink-0" />
        </button>
      ))}
    </div>
  );
}

interface SubmissionDetailProps {
  submission: Submission;
  attempt: number;
  time: string;
  onBack: () => void;
}

function SubmissionDetail({ submission, attempt, time, onBack }: SubmissionDetailProps) {
  const { t } = useLocale();
  const { currentCode, setCurrentCode } = useProblemStore();
  const [mode, setMode] = useState<'code' | 'diff'>('code');
  const [copied, setCopied] = useState(false);
  const [confirmRestore, setConfirmRestore] = useState(false);
  const isSameAsCurrent = currentCode === submission.code;

  // Restoring overwrites the editor, so it takes a second click within a few seconds
  useEffect(() => {
    if (!confirmRestore) return;
    const timer = setTimeout(() => setConfirmRestore(false), 3000);
    return () => clearTimeout(timer);
  }, [confirmRestore]);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timer);
  }, [copied]);

  const handleCopy = () => {
    navigator.clipboard.writeText(submission.code).then(() => setCopied(true)).catch(() => {});
  };

  const handleRestore = () => {
    if (!confirmRestore) {
      setConfirmRestore(true);
      return;
    }
    setCurrentCode(submission.code);
    setConfirmRestore(false);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-shrink-0 px-3 py-2 space-y-2" style={{ borderBottom: '1px solid var(--line)' }}>
        <div className="flex items-center gap-1">
          <button type="button" onClick={onBack} className={cn(actionClass, 'mr-auto')}>
            <ArrowLeft className="w-3.5 h-3.5" />
            {t('submissionBack')}
          </button>
          <button type="button" onClick={handleCopy} className={actionClass}>
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? t('submissionCopied') : t('submissionCopy')}
          </button>
          <button
            type="button"
            onClick={handleRestore}
            disabled={isSameAsCurrent}
            className={cn(actionClass, confirmRestore && 'text-hard hover:text-hard')}
          >
            {isSameAsCurrent ? <Check className="w-3.5 h-3.5" /> : <RotateCcw className="w-3.5 h-3.5" />}
            {isSameAsCurrent
              ? t('submissionRestored')
              : confirmRestore ? t('submissionRestoreConfirm') : t('submissionRestore')}
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-1">
          <StatusLabel passed={submission.passed} />
          <span className="text-xs text-text-3">
            {t('submissionAttempt', { n: attempt })} · {time}
            {submission.execTimeMs != null && ` · ${submission.execTimeMs.toFixed(1)}ms`}
          </span>
          <div className="flex gap-0.5 ml-auto">
            {(['code', 'diff'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn(
                  segmentClass,
                  mode === m ? 'bg-[color-mix(in_oklab,var(--text)_7%,transparent)] text-text' : 'hover:text-text',
                )}
              >
                {m === 'code' ? t('submissionViewCode') : t('submissionDiff')}
              </button>
            ))}
          </div>
        </div>
      </div>

      {mode === 'code' ? (
        <div className="flex-1 min-h-0">
          <CodeEditor value={submission.code} onChange={() => {}} readOnly />
        </div>
      ) : isSameAsCurrent ? (
        <div className="flex-1 flex items-center justify-center text-sm text-text-3">{t('submissionSame')}</div>
      ) : (
        <div className="flex-1 min-h-0 flex flex-col">
          <div className="px-4 py-1.5 text-xs text-text-3 flex-shrink-0">{t('submissionDiffHint')}</div>
          <div className="flex-1 min-h-0">
            <CodeDiffViewer original={submission.code} modified={currentCode} />
          </div>
        </div>
      )}
    </div>
  );
}
