'use client';

export default function Toggle({
  label,
  checked,
  onChange,
  disabled,
  hint,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  hint?: string;
}) {
  return (
    <label
      className={`flex items-center justify-between gap-3 py-2 ${
        disabled ? 'cursor-not-allowed opacity-40' : 'cursor-pointer'
      }`}
    >
      <span>
        <span className="block text-sm font-medium text-textMain">{label}</span>
        {hint && <span className="block text-xs text-textDim">{hint}</span>}
      </span>
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative h-6 w-11 shrink-0 rounded-full transition-all duration-200 ${
          checked ? 'bg-accent-gradient shadow-glow' : 'bg-card2 border border-border'
        } ${!disabled ? 'hover:scale-105 active:scale-95' : ''}`}
      >
        <span
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform duration-200 ease-[cubic-bezier(0.34,1.56,0.64,1)] ${
            checked ? 'translate-x-5' : 'translate-x-0.5'
          }`}
        />
      </button>
    </label>
  );
}
