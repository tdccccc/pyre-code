import { create } from 'zustand';
import type { SubmissionResult, CustomTest, AiHelpConfig, AiChatMessage, SubmissionHistory } from '@/lib/types';

interface ProblemStore {
  currentCode: string;
  setCurrentCode: (code: string) => void;
  submissionResult: SubmissionResult | null;
  setSubmissionResult: (result: SubmissionResult | null) => void;
  isSubmitting: boolean;
  setIsSubmitting: (v: boolean) => void;
  drawerOpen: boolean;
  setDrawerOpen: (v: boolean) => void;
  // test panel
  bottomTab: 'testcases' | 'testresults';
  setBottomTab: (tab: 'testcases' | 'testresults') => void;
  selectedCaseIndex: number;
  setSelectedCaseIndex: (i: number) => void;
  customTests: CustomTest[];
  addCustomTest: (test: CustomTest) => void;
  removeCustomTest: (index: number) => void;
  updateCustomTest: (index: number, test: CustomTest) => void;
  isRunning: boolean;
  setIsRunning: (v: boolean) => void;
  runResult: SubmissionResult | null;
  setRunResult: (r: SubmissionResult | null) => void;
  submissionHistory: SubmissionHistory[];
  setSubmissionHistory: (h: SubmissionHistory[]) => void;
  aiHelpConfig: AiHelpConfig;
  setAiHelpConfig: (patch: Partial<AiHelpConfig>) => void;
  aiHelpConfigOpen: boolean;
  setAiHelpConfigOpen: (value: boolean) => void;
  aiHelpMessages: AiChatMessage[];
  setAiHelpMessages: (update: AiChatMessage[] | ((prev: AiChatMessage[]) => AiChatMessage[])) => void;
  aiHelpError: string | null;
  setAiHelpError: (error: string | null) => void;
  aiHelpLoading: boolean;
  setAiHelpLoading: (value: boolean) => void;
  aiHelpAbortController: AbortController | null;
  setAiHelpAbortController: (controller: AbortController | null) => void;
  resetAiHelp: () => void;
  resetTestPanel: () => void;
}

export const useProblemStore = create<ProblemStore>((set, get) => ({
  currentCode: '',
  setCurrentCode: (code) => set({ currentCode: code }),
  submissionResult: null,
  setSubmissionResult: (result) => set({ submissionResult: result }),
  isSubmitting: false,
  setIsSubmitting: (v) => set({ isSubmitting: v }),
  drawerOpen: false,
  setDrawerOpen: (v) => set({ drawerOpen: v }),
  // test panel
  bottomTab: 'testcases',
  setBottomTab: (tab) => set({ bottomTab: tab }),
  selectedCaseIndex: 0,
  setSelectedCaseIndex: (i) => set({ selectedCaseIndex: i }),
  customTests: [],
  addCustomTest: (test) => set((s) => ({ customTests: [...s.customTests, test] })),
  removeCustomTest: (index) => set((s) => ({ customTests: s.customTests.filter((_, i) => i !== index) })),
  updateCustomTest: (index, test) => set((s) => {
    const next = [...s.customTests];
    next[index] = test;
    return { customTests: next };
  }),
  isRunning: false,
  setIsRunning: (v) => set({ isRunning: v }),
  runResult: null,
  setRunResult: (r) => set({ runResult: r }),
  submissionHistory: [],
  setSubmissionHistory: (h) => set({ submissionHistory: h }),
  aiHelpConfig: {
    baseUrl: '',
    apiKey: '',
    model: '',
    includeUserCode: true,
  },
  setAiHelpConfig: (patch) => set((s) => ({ aiHelpConfig: { ...s.aiHelpConfig, ...patch } })),
  aiHelpConfigOpen: false,
  setAiHelpConfigOpen: (value) => set({ aiHelpConfigOpen: value }),
  aiHelpMessages: [],
  setAiHelpMessages: (update) => set((s) => ({
    aiHelpMessages: typeof update === 'function' ? update(s.aiHelpMessages) : update,
  })),
  aiHelpError: null,
  setAiHelpError: (error) => set({ aiHelpError: error }),
  aiHelpLoading: false,
  setAiHelpLoading: (value) => set({ aiHelpLoading: value }),
  aiHelpAbortController: null,
  setAiHelpAbortController: (controller) => set({ aiHelpAbortController: controller }),
  resetAiHelp: () => {
    get().aiHelpAbortController?.abort();
    set({ aiHelpMessages: [], aiHelpError: null, aiHelpLoading: false, aiHelpAbortController: null });
  },
  resetTestPanel: () => set({ bottomTab: 'testcases', selectedCaseIndex: 0, customTests: [], runResult: null }),
}));
