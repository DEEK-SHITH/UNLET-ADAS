'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, animate } from 'framer-motion';

/** Animated numeric stat card: crossfades in and counts up/down to a
 * new numeric value whenever it changes, instead of snapping. Falls
 * back to a plain text render for non-numeric values (e.g. "Yes"). */
export default function StatCard({
  label,
  value,
  danger,
  icon,
}: {
  label: string;
  value: string;
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
      duration: 0.5,
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
      initial={{ opacity: 0, scale: 0.92 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="glass relative overflow-hidden rounded-xl px-3 py-2.5 text-center"
    >
      <div
        className={`text-lg font-extrabold tabular-nums ${danger ? 'text-danger' : 'text-accent'}`}
      >
        {icon && <span className="mr-1">{icon}</span>}
        {numericValue !== null ? display.toFixed(decimals) : value}
        {suffix}
      </div>
      <div className="mt-0.5 text-[11px] text-textDim">{label}</div>
    </motion.div>
  );
}
