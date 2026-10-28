'use client';

import { useEffect, useRef, useState } from 'react';
import * as Collapsible from '@radix-ui/react-collapsible';
import { Check, ChevronDown, Loader2, RefreshCw, Settings2, Sparkles, ServerCog } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useLocale } from '@/context/LocaleContext';
import { MarkdownContent } from '@/components/workspace/MarkdownContent';
import { AI_HELP_PROMPT_PRESETS, DEFAULT_AI_HELP_PROMPT_PRESET } from '@/lib/aiHelpPrompts';
import { getSampleTests, getSolutionCode } from '@/lib/problemContext';
import { cn } from '@/lib/utils';
import { useProblemStore } from '@/store/problemStore';
import type { Problem, AiHelpRequest, AiHelpResponse } from '@/lib/types';

const STORAGE_KEY = 'ai_help_config';
const CUSTOM_PROMPT_STORAGE_KEY = 'ai_help_custom_prompt';

interface AIHelpTabProps {
  problem: Problem;
}

export function AIHelpTab({ problem }: AIHelpTabProps) {
  const { locale, t } = useLocale();
  const hasLoadedConfig = useRef(false);
  const [serverConfigured, setServerConfigured] = useState(false);
  const modelBoxRef = useRef<HTMLDivElement>(null);
  const [models, setModels] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [modelListOpen, setModelListOpen] = useState(false);
  const [modelQuery, setModelQuery] = useState('');
  const {
    currentCode,
    aiHelpConfig, setAiHelpConfig,
    aiHelpConfigOpen, setAiHelpConfigOpen,
    aiHelpCustomPrompt, setAiHelpCustomPrompt,
    aiHelpResponse, setAiHelpResponse,
    aiHelpError, setAiHelpError,
    aiHelpLoading, setAiHelpLoading,
  } = useProblemStore();

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    hasLoadedConfig.current = true;
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as Partial<typeof aiHelpConfig>;
        setAiHelpConfig({ baseUrl: parsed.baseUrl ?? '', apiKey: parsed.apiKey ?? '', model: parsed.model ?? '' });
      } catch {}
    }
    // First visit (nothing saved yet) starts from the default template
    const savedPrompt = localStorage.getItem(CUSTOM_PROMPT_STORAGE_KEY);
    setAiHelpCustomPrompt(savedPrompt ?? DEFAULT_AI_HELP_PROMPT_PRESET.prompt[locale]);
  }, [locale, setAiHelpConfig, setAiHelpCustomPrompt]);

  // Read from the store, not the render closure: on mount the closure still holds the
  // pre-load values, and persisting those would clobber what was just loaded.
  useEffect(() => {
    if (!hasLoadedConfig.current) return;
    const { baseUrl, apiKey, model } = useProblemStore.getState().aiHelpConfig;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ baseUrl, apiKey, model }));
  }, [aiHelpConfig.baseUrl, aiHelpConfig.apiKey, aiHelpConfig.model]);

  useEffect(() => {
    if (!hasLoadedConfig.current) return;
    localStorage.setItem(CUSTOM_PROMPT_STORAGE_KEY, useProblemStore.getState().aiHelpCustomPrompt);
  }, [aiHelpCustomPrompt]);

  useEffect(() => {
    fetch('/api/ai-help/status')
      .then((r) => r.json())
      .then((d: { configured: boolean }) => setServerConfigured(d.configured))
      .catch(() => setServerConfigured(false));
  }, []);

  useEffect(() => {
    if (!modelListOpen) return;
    const handlePointerDown = (e: MouseEvent) => {
      if (!modelBoxRef.current?.contains(e.target as Node)) setModelListOpen(false);
    };
    document.addEventListener('mousedown', handlePointerDown);
    return () => document.removeEventListener('mousedown', handlePointerDown);
  }, [modelListOpen]);

  const handleFetchModels = async () => {
    if (!aiHelpConfig.baseUrl.trim() || !aiHelpConfig.apiKey.trim()) {
      setModelsError(t('aiHelpFetchModelsNeedConfig'));
      return;
    }
    setModelsLoading(true);
    setModelsError(null);
    try {
      const res = await fetch('/api/ai-help/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ baseUrl: aiHelpConfig.baseUrl, apiKey: aiHelpConfig.apiKey }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const data: { models: string[] } = await res.json();
      setModels(data.models);
      setModelQuery('');
      setModelListOpen(true);
    } catch {
      setModelsError(t('aiHelpFetchModelsFailed'));
    } finally {
      setModelsLoading(false);
    }
  };

  const activePresetId = AI_HELP_PROMPT_PRESETS.find(
    (p) => p.prompt.zh === aiHelpCustomPrompt || p.prompt.en === aiHelpCustomPrompt,
  )?.id;
  const filteredModels = models.filter((m) => m.toLowerCase().includes(modelQuery.trim().toLowerCase()));

  const handleGenerate = async () => {
    if (!serverConfigured && (!aiHelpConfig.baseUrl.trim() || !aiHelpConfig.apiKey.trim() || !aiHelpConfig.model.trim())) {
      setAiHelpError(t('aiHelpMissingConfig'));
      return;
    }
    setAiHelpLoading(true);
    setAiHelpError(null);
    setAiHelpResponse(null);
    try {
      const description = locale === 'zh' ? problem.descriptionZh : problem.descriptionEn;
      const solutionCode = getSolutionCode(problem.id);
      const sampleTests = getSampleTests(problem);
      const body: AiHelpRequest = {
        problemId: problem.id,
        problemTitle: problem.title,
        functionName: problem.functionName,
        description,
        solutionCode: solutionCode || '',
        sampleTests,
        customPrompt: aiHelpCustomPrompt || undefined,
        userCode: aiHelpConfig.includeUserCode !== false ? currentCode : undefined,
        locale,
        config: { baseUrl: aiHelpConfig.baseUrl, apiKey: aiHelpConfig.apiKey, model: aiHelpConfig.model },
      };
      const res = await fetch('/api/ai-help', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!res.ok) throw new Error(`${res.status}`);
      const data: AiHelpResponse = await res.json();
      setAiHelpResponse(data.guidance);
    } catch {
      setAiHelpError(t('aiHelpRequestFailed'));
    } finally {
      setAiHelpLoading(false);
    }
  };

  const inputStyle = {
    background: 'var(--bg-elev)',
    border: '1px solid var(--line)',
  };

  return (
    <div className="p-4 space-y-4 overflow-y-auto h-full">
      <div className="text-xs text-text-3 flex items-center gap-1.5">
        <Sparkles className="w-3.5 h-3.5" />
        {t('aiHelpSafetyNote')}
      </div>

      <div className="space-y-3">
        <div className="space-y-1.5">
          <span className="text-xs font-medium text-text-2">{t('aiHelpCustomPrompt')}</span>
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-text-3">{t('aiHelpPresets')}</span>
            {AI_HELP_PROMPT_PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => setAiHelpCustomPrompt(preset.prompt[locale])}
                className={cn(
                  'rounded-full border px-2.5 py-0.5 text-xs transition-colors cursor-pointer',
                  activePresetId === preset.id
                    ? 'border-accent-line bg-accent-wash text-accent-ink'
                    : 'border-line text-text-2 hover:border-line-strong hover:text-text',
                )}
              >
                {preset.label[locale]}
              </button>
            ))}
          </div>
          <textarea
            value={aiHelpCustomPrompt}
            onChange={(e) => setAiHelpCustomPrompt(e.target.value)}
            className="w-full rounded-lg px-3 py-2 text-sm text-text outline-none resize-y"
            style={inputStyle}
            rows={6}
            placeholder={t('aiHelpCustomPromptPlaceholder')}
            aria-label={t('aiHelpCustomPrompt')}
          />
        </div>

        <label className="flex items-center gap-2 text-sm text-text-2">
          <input
            type="checkbox"
            checked={aiHelpConfig.includeUserCode !== false}
            onChange={(e) => setAiHelpConfig({ includeUserCode: e.target.checked })}
            className="accent-[var(--accent)]"
          />
          {t('aiHelpIncludeCode')}
        </label>

        <Button onClick={handleGenerate} disabled={aiHelpLoading} className="w-full">
          {aiHelpLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
          {aiHelpLoading ? t('aiHelpGenerating') : t('aiHelpGenerate')}
        </Button>

        {serverConfigured && (
          <div className="flex items-center gap-1.5 text-xs text-text-3">
            <ServerCog className="w-3.5 h-3.5" />
            {t('aiHelpServerConfigured')}
          </div>
        )}

        <Collapsible.Root open={aiHelpConfigOpen} onOpenChange={setAiHelpConfigOpen}>
          <Collapsible.Trigger className="flex items-center gap-1.5 text-xs text-text-3 hover:text-text-2 transition-colors cursor-pointer">
            <Settings2 className="w-3.5 h-3.5" />
            {aiHelpConfigOpen ? t('aiHelpHideConfig') : t('aiHelpShowConfig')}
            <ChevronDown className={`w-3 h-3 transition-transform ${aiHelpConfigOpen ? 'rotate-180' : ''}`} />
          </Collapsible.Trigger>
          <Collapsible.Content className="mt-3 space-y-3">
            {!serverConfigured && (
              <>
                <label className="block space-y-1">
                  <span className="text-xs font-medium text-text-2">{t('aiHelpBaseUrl')}</span>
                  <input
                    type="text"
                    value={aiHelpConfig.baseUrl}
                    onChange={(e) => setAiHelpConfig({ baseUrl: e.target.value })}
                    className="w-full rounded-lg px-3 py-2 text-sm text-text outline-none"
                    style={inputStyle}
                    placeholder="https://api.openai.com/v1"
                  />
                </label>
                <label className="block space-y-1">
                  <span className="text-xs font-medium text-text-2">{t('aiHelpApiKey')}</span>
                  <input
                    type="password"
                    value={aiHelpConfig.apiKey}
                    onChange={(e) => setAiHelpConfig({ apiKey: e.target.value })}
                    className="w-full rounded-lg px-3 py-2 text-sm text-text outline-none"
                    style={inputStyle}
                    placeholder="sk-..."
                  />
                </label>
                <div className="space-y-1">
                  <span className="text-xs font-medium text-text-2">{t('aiHelpModel')}</span>
                  <div ref={modelBoxRef} className="relative">
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={aiHelpConfig.model}
                        onChange={(e) => {
                          setAiHelpConfig({ model: e.target.value });
                          setModelQuery(e.target.value);
                          if (models.length) setModelListOpen(true);
                        }}
                        onFocus={() => models.length && setModelListOpen(true)}
                        onKeyDown={(e) => e.key === 'Escape' && setModelListOpen(false)}
                        className="w-full min-w-0 rounded-lg px-3 py-2 text-sm text-text outline-none"
                        style={inputStyle}
                        placeholder="gpt-4o-mini"
                        aria-label={t('aiHelpModel')}
                        autoComplete="off"
                      />
                      <Button variant="secondary" onClick={handleFetchModels} disabled={modelsLoading} className="shrink-0">
                        {modelsLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                        {t('aiHelpFetchModels')}
                      </Button>
                    </div>
                    {modelListOpen && models.length > 0 && (
                      <ul
                        className="absolute z-20 mt-1 w-full max-h-60 overflow-y-auto rounded-lg py-1 text-sm shadow-lg"
                        style={{ background: 'var(--bg-elev)', border: '1px solid var(--line)' }}
                      >
                        {filteredModels.length ? filteredModels.map((m) => (
                          <li key={m}>
                            <button
                              type="button"
                              onClick={() => {
                                setAiHelpConfig({ model: m });
                                setModelListOpen(false);
                              }}
                              className="flex w-full items-center justify-between gap-2 px-3 py-1.5 text-left text-text-2 hover:text-text hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)] cursor-pointer"
                            >
                              <span className="truncate">{m}</span>
                              {m === aiHelpConfig.model && <Check className="w-3.5 h-3.5 shrink-0 text-accent" />}
                            </button>
                          </li>
                        )) : (
                          <li className="px-3 py-1.5 text-text-3">{t('aiHelpNoModels')}</li>
                        )}
                      </ul>
                    )}
                  </div>
                  {modelsError && <div className="text-xs text-hard">{modelsError}</div>}
                </div>
              </>
            )}
          </Collapsible.Content>
        </Collapsible.Root>
      </div>

      {aiHelpError && (
        <div className="rounded-xl px-4 py-3 text-sm text-hard whitespace-pre-wrap" style={{ border: '1px solid color-mix(in oklab, var(--hard) 20%, var(--line))', background: 'color-mix(in oklab, var(--hard) 5%, var(--bg-elev))' }}>
          {aiHelpError}
        </div>
      )}

      {aiHelpResponse ? (
        <div className="rounded-xl px-4 py-4" style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-text-3">{t('aiHelpResponseTitle')}</div>
          <div className="text-sm text-text-2"><MarkdownContent content={aiHelpResponse} /></div>
        </div>
      ) : !aiHelpError && (
        <div className="rounded-xl px-4 py-6 text-sm text-text-3" style={{ border: '1px dashed var(--line)' }}>
          {t('aiHelpEmpty')}
        </div>
      )}
    </div>
  );
}
