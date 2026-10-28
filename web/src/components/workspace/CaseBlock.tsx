/** A labelled monospace block for Hot 100 case data (input, expected, the learner's answer). */
export function CaseBlock({ label, text, tone }: { label: string; text: string; tone?: 'good' | 'bad' }) {
  const color = tone === 'good' ? 'var(--easy)' : tone === 'bad' ? 'var(--hard)' : undefined;
  return (
    <div>
      <h4 className="eyebrow mb-1.5">{label}</h4>
      <pre
        className="p-3 rounded-lg text-xs font-mono overflow-x-auto whitespace-pre-wrap break-words leading-relaxed min-h-[2.25rem]"
        style={{
          background: color ? `color-mix(in oklab, ${color} 6%, var(--bg-elev))` : 'var(--bg-sunken)',
          color: color ?? 'var(--text)',
        }}
      >
        {text}
      </pre>
    </div>
  );
}
