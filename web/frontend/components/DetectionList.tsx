'use client';

import { motion } from 'framer-motion';
import type { Detection } from '@/lib/api';

const RISK_STYLE: Record<
  string,
  { card: string; badge: string; bar: string; dot: string }
> = {
  HIGH: {
    card: 'border-danger/35 bg-danger/10 hover:border-danger/60',
    badge: 'bg-danger/20 text-danger border-danger/40',
    bar: 'bg-gradient-to-r from-danger/70 to-danger',
    dot: 'bg-danger animate-ping',
  },
  MEDIUM: {
    card: 'border-warning/35 bg-warning/10 hover:border-warning/60',
    badge: 'bg-warning/20 text-warning border-warning/40',
    bar: 'bg-gradient-to-r from-warning/70 to-warning',
    dot: 'bg-warning',
  },
  LOW: {
    card: 'border-ok/30 bg-ok/10 hover:border-ok/50',
    badge: 'bg-ok/20 text-ok border-ok/40',
    bar: 'bg-gradient-to-r from-ok/70 to-ok',
    dot: 'bg-ok',
  },
};

const CLASS_ICON: Record<string, string> = {
  person: '🚶',
  car: '🚗',
  truck: '🚚',
  bus: '🚌',
  motorcycle: '🏍️',
  bicycle: '🚲',
  'traffic light': '🚦',
  'stop sign': '🛑',
  pothole: '🕳️',
};

export default function DetectionList({
  title,
  items,
}: {
  title: string;
  items: Detection[];
}) {
  if (items.length === 0) return null;
  return (
    <div className="mt-4">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-textDim">
          <span className="h-1.5 w-1.5 rounded-full bg-accent" />
          {title}
        </div>
        <span className="rounded-full border border-accent/30 bg-accent/10 px-2 py-0.5 font-mono text-[11px] font-bold text-accent">
          {items.length} detected
        </span>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 md:grid-cols-3">
        {items.map((d, i) => {
          const riskStyle = d.risk ? RISK_STYLE[d.risk] : null;
          const pct = Math.round(d.conf * 100);
          const icon = CLASS_ICON[d.name.toLowerCase()] ?? '🎯';

          return (
            <motion.div
              key={`${d.name}-${i}`}
              initial={{ opacity: 0, scale: 0.92, y: 6 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              whileHover={{ y: -2 }}
              transition={{ duration: 0.25, delay: Math.min(i * 0.04, 0.35) }}
              className={`relative overflow-hidden rounded-xl border p-2.5 transition-colors ${
                riskStyle
                  ? riskStyle.card
                  : 'border-border bg-card2/80 hover:border-accent/40'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="text-sm">{icon}</span>
                  <span className="truncate text-xs font-bold capitalize text-textMain">
                    {d.name} #{i + 1}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <span className="font-mono text-xs font-extrabold text-textMain">
                    {pct}%
                  </span>
                  {d.risk && riskStyle && (
                    <span
                      className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[10px] font-bold ${riskStyle.badge}`}
                    >
                      <span className={`h-1.5 w-1.5 rounded-full ${riskStyle.dot}`} />
                      {d.risk}
                    </span>
                  )}
                </div>
              </div>

              {/* Animated confidence bar */}
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-base/60">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{
                    duration: 0.55,
                    delay: Math.min(i * 0.04 + 0.1, 0.4),
                    ease: [0.16, 1, 0.3, 1],
                  }}
                  className={`h-full rounded-full ${
                    riskStyle ? riskStyle.bar : 'bg-accent-gradient'
                  }`}
                />
              </div>

              {d.box && d.box.length === 4 && (
                <div className="mt-1.5 flex items-center justify-between font-mono text-[10px] text-textDim/80">
                  <span>
                    [{d.box[0]}, {d.box[1]}] → [{d.box[2]}, {d.box[3]}]
                  </span>
                  <span>
                    {d.box[2] - d.box[0]}×{d.box[3] - d.box[1]}px
                  </span>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
