import type { Locale } from '@/lib/i18n';

/** Quick actions in the AI help chat: clicking one sends `prompt`, the bubble shows `label`. */
export interface AiHelpQuickAction {
  id: string;
  label: Record<Locale, string>;
  prompt: Record<Locale, string>;
}

export const AI_HELP_QUICK_ACTIONS: AiHelpQuickAction[] = [
  {
    id: 'diagnose',
    label: { en: 'Full diagnosis', zh: '全面诊断' },
    prompt: {
      zh: `请对我目前的进度做一次全面诊断，按以下结构回答：

## 诊断
用一句话判断我现在的状态：没有思路 / 思路对但有 bug / 基本正确只差细节。

## 原理定位
说明这道题对应大模型里的哪个组件或训练环节，解决什么问题，和相关方法有什么区别。最多 3 句话。

## 代码检查
（如果有我的代码）指出具体哪一行有问题、为什么错、可能导致哪个测试失败。

## 分层提示
给 3 级提示，从模糊到具体：
1. 只点方向
2. 说明关键步骤和中间张量的形状
3. 可以给不超过 3 行的关键代码片段

## 自测方法
给 1–2 个我能在本地快速跑的检查。

如果我的代码已经正确，直接说明，然后讲可改进的地方或实际大模型中的工程做法。总长度控制在 600 字以内。`,
      en: `Please give a full diagnosis of where I am, using this structure:

## Diagnosis
One sentence: no idea yet / right idea but buggy / basically correct, only details left.

## Where this fits
Which LLM component or training stage this corresponds to, what it solves, and how it differs from related methods. At most 3 sentences.

## Code review
(If my code is provided) the exact lines that are wrong, why, and which test they would likely fail.

## Layered hints
3 levels, from vague to specific:
1. Just the direction
2. Key steps and intermediate tensor shapes
3. At most a 3-line key code snippet

## Self-checks
1–2 quick checks I can run locally.

If my code is already correct, say so, then cover improvements or how real LLM systems do it. Keep it under 400 words.`,
    },
  },
  {
    id: 'hint',
    label: { en: 'Give me a hint', zh: '给个提示' },
    prompt: {
      zh: '我卡住了，给我一个最小的提示让我能继续往下写，不要直接给答案。',
      en: "I'm stuck. Give me the smallest hint that lets me keep going, without giving away the answer.",
    },
  },
  {
    id: 'idea',
    label: { en: 'Idea only', zh: '只要思路' },
    prompt: {
      zh: '只讲这个模块在大模型中的作用和计算流程，标出每一步张量形状，不要给代码。',
      en: 'Only explain the role of this module in LLMs and its computation flow, annotating tensor shapes at each step. Do not give code.',
    },
  },
  {
    id: 'debug',
    label: { en: 'Find bugs', zh: '专门找 bug' },
    prompt: {
      zh: '只检查我代码里的错误，按严重程度排序，重点看 mask、形状和数值稳定性，每条说明出错行、原因和修改方向。',
      en: 'Only check my code for bugs, ordered by severity, focusing on masks, shapes and numerical stability. For each, give the line, the cause and how to fix it.',
    },
  },
  {
    id: 'compare',
    label: { en: 'Compare', zh: '对比辨析' },
    prompt: {
      zh: '把这道题的方法和它的前身或变体做对比（解决了什么问题、计算量和显存的变化、哪些模型在用），再指出实现时最容易写错的一处。',
      en: 'Compare this method with its predecessors or variants (what problem it solves, changes in compute and memory, which models use it), then point out the single easiest place to get wrong when implementing it.',
    },
  },
  {
    id: 'engineering',
    label: { en: 'Engineering', zh: '工程视角' },
    prompt: {
      zh: '说明这个组件在真实的大模型训练或推理框架里是怎么实现和优化的，和题目里的简化版有什么差别。',
      en: 'Explain how this component is implemented and optimized in real LLM training or inference frameworks, and how that differs from the simplified version in this problem.',
    },
  },
];
