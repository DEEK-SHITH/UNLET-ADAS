'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, animate } from 'framer-motion';

/** Animated numeric stat card: crossfades in and counts up/down to a
 * new numeric value whenever it changes, instead of snapping. Falls
 * back to a plain text render for non-numeric values (e.g. "Yes"). */
export default function StatCard({
  label,
  value,
  sublabel,
  danger,
  icon,
}: {
  label: string;
  value: string;
  sublabel?: string;
  danger?: boolean;
  icon?: string;
}) {
  const numericMatch = value.match(/^-?\d+(\.\d+)?/);
  const suffix = numericMatch ? value.slice(numericMatch[0].length) : '';
  const numericValue = numericMatch ? parseFloat(numericMatch[0]) : null;
  const decimals = numericMatch && numericMatch[0].includes('.')
    ? numericMatch[0].split('.')[1].length
    : 0;

  const [display, setDisplay] = useState(numericValue ?? 0);
  const prevRef = useRef(numericValue ?? 0);

  useEffect(() => {
    if (numericValue === null) return;
    const controls = animate(prevRef.current, numericValue, {
      duration: 0.55,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => setDisplay(v),
    });
    prevRef.current = numericValue;
    return () => controls.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numericValue]);

  return (
    <motion.div
      layout
      whileHover={{ y: -2, scale: 1.015 }}
      initial={{ opacity: 0, scale: 0.92 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className={`glass group relative overflow-hidden rounded-xl px-3.5 py-3 text-center transition-colors ${
        danger ? 'border-danger/40 bg-danger/5' : 'hover:border-accent/40'
      }`}
    >
      <div
        className={`absolute inset-x-0 top-0 h-[2px] ${
          danger
            ? 'bg-gradient-to-r from-transparent via-danger to-transparent'
            : 'bg-gradient-to-r from-transparent via-accent/60 to-transparent opacity-70 group-hover:opacity-100'
        }`}
      />
      <div
        className={`text-xl font-extrabold tracking-tight tabular-nums ${
          danger ? 'text-danger' : 'text-accent'
        }`}
      >
        {icon && <span className="mr-1.5 text-base">{icon}</span>}
        {numericValue !== null ? display.toFixed(decimals) : value}
        {suffix}
      </div>
      <div className="mt-0.5 text-[11px] font-semibold uppercase tracking-wider text-textDim">
        {label}
      </div>
      {sublabel && (
        <div className="mt-0.5 text-[10px] font-mono text-textDim/80">
          {sublabel}
        </div>
      )}
    </motion.div>
  );
}
