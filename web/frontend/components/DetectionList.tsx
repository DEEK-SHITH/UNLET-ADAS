'use client';

import { motion } from 'framer-motion';
import type { Detection } from '@/lib/api';

const RISK_COLOR: Record<string, string> = {
  HIGH: 'bg-danger/15 text-danger border-danger/30',
  MEDIUM: 'bg-warning/15 text-warning border-warning/30',
  LOW: 'bg-ok/15 text-ok border-ok/30',
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
    <div className="mt-3">
      <div className="mb-1.5 text-xs font-bold uppercase tracking-wide text-textDim">
        {title} ({items.length})
      </div>
      <div className="flex flex-wrap gap-1.5">
        {items.map((d, i) => (
          <motion.span
            key={`${d.name}-${i}`}
            initial={{ opacity: 0, scale: 0.8, y: 4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.25, delay: Math.min(i * 0.04, 0.3) }}
            className={`rounded-full border px-2.5 py-1 text-xs font-medium ${
              d.risk ? RISK_COLOR[d.risk] : 'border-border bg-card2 text-textMain'
            }`}
          >
            {d.name} · {(d.conf * 100).toFixed(0)}%
            {d.risk ? ` · ${d.risk}` : ''}
          </motion.span>
        ))}
      </div>
    </div>
  );
}
