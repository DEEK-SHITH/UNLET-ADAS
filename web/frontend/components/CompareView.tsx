'use client';

import { useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type ViewMode = 'side-by-side' | 'split';

export default function CompareView({
  before,
  after,
  origBrightness,
  enhBrightness,
  objectCount,
}: {
  before: string;
  after: string;
  origBrightness?: number;
  enhBrightness?: number;
  objectCount?: number;
}) {
  const [mode, setMode] = useState<ViewMode>('side-by-side');
  const [zoomEnabled, setZoomEnabled] = useState(false);
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null);
  const [pos, setPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updateFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.min(100, Math.max(0, pct)));
  }, []);

  const handleCardMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!zoomEnabled) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100;
      const y = ((e.clientY - rect.top) / rect.height) * 100;
      setHoverPos({
        x: Math.min(100, Math.max(0, x)),
        y: Math.min(100, Math.max(0, y)),
      });
    },
    [zoomEnabled]
  );

  const gainPct =
    origBrightness !== undefined &&
    enhBrightness !== undefined &&
    origBrightness > 0.001
      ? Math.round(((enhBrightness - origBrightness) / origBrightness) * 100)
      : null;

  return (
    <div className="space-y-3">
      {/* Comparison Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-border/80 bg-card/70 px-3 py-2">
        <div className="flex items-center gap-1 rounded-lg border border-border bg-base/70 p-1">
          {(['side-by-side', 'split'] as const).map((m) => {
            const active = mode === m;
            return (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className="relative rounded-md px-3 py-1 text-xs font-bold transition-colors"
              >
                {active && (
                  <motion.span
                    layoutId="compare-mode-pill"
                    className="absolute inset-0 rounded-md bg-accent-gradient shadow-glow"
                    transition={{ type: 'spring', stiffness: 420, damping: 30 }}
                  />
                )}
                <span
                  className={`relative z-10 flex items-center gap-1.5 ${
                    active ? 'text-base' : 'text-textDim hover:text-textMain'
                  }`}
                >
                  <span>{m === 'side-by-side' ? '◫' : '⇔'}</span>
                  {m === 'side-by-side' ? 'Side-by-Side' : 'Split Curtain'}
                </span>
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-2">
          {mode === 'side-by-side' && (
            <button
              type="button"
              onClick={() => {
                setZoomEnabled((z) => !z);
                setHoverPos(null);
              }}
              className={`rounded-lg border px-2.5 py-1 text-xs font-semibold transition ${
                zoomEnabled
                  ? 'border-accent/60 bg-accent/15 text-accent shadow-glow'
                  : 'border-border bg-card2 text-textDim hover:border-borderLt hover:text-textMain'
              }`}
            >
              🔍 {zoomEnabled ? 'Sync Loupe: ON' : 'Sync Loupe'}
            </button>
          )}
          <a
            href={after}
            download="unlet_adas_enhanced.jpg"
            className="flex items-center gap-1.5 rounded-lg border border-accent/40 bg-accent/10 px-3 py-1 text-xs font-bold text-accent transition hover:bg-accent/20"
          >
            ⬇ Download Enhanced
          </a>
        </div>
      </div>

      {/* Viewport */}
      <AnimatePresence mode="wait">
        {mode === 'side-by-side' ? (
          <motion.div
            key="side-by-side"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="grid grid-cols-1 gap-4 md:grid-cols-2"
          >
            {/* Left: Original Dark Frame */}
            <motion.div
              whileHover={{ y: -2 }}
              transition={{ type: 'spring', stiffness: 300, damping: 22 }}
              onMouseMove={handleCardMouseMove}
              onMouseLeave={() => setHoverPos(null)}
              className="group relative flex flex-col overflow-hidden rounded-2xl border border-border bg-black/90 shadow-lg"
            >
              <div className="flex items-center justify-between gap-2 border-b border-border/60 bg-card/80 px-3.5 py-2.5 backdrop-blur-md">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="h-2 w-2 shrink-0 rounded-full bg-danger" />
                  <span className="truncate text-xs font-bold uppercase tracking-wider text-textMain">
                    Original Input
                  </span>
                </div>
                {origBrightness !== undefined && (
                  <span className="shrink-0 whitespace-nowrap rounded-full border border-border bg-base/80 px-2.5 py-0.5 font-mono text-[11px] font-semibold tabular-nums text-textDim">
                    Lum: {origBrightness.toFixed(3)}
                  </span>
                )}
              </div>

              <div className="relative flex min-h-[280px] max-h-[540px] flex-1 items-center justify-center overflow-hidden bg-black p-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={before}
                  alt="Original Low-Light"
                  className="max-h-[520px] w-auto max-w-full rounded-lg object-contain transition-transform duration-300"
                  style={
                    zoomEnabled && hoverPos
                      ? {
                          transform: 'scale(2.2)',
                          transformOrigin: `${hoverPos.x}% ${hoverPos.y}%`,
                        }
                      : undefined
                  }
                  draggable={false}
                />
              </div>
            </motion.div>

            {/* Right: UNLET-ADAS Enhanced + Detections */}
            <motion.div
              whileHover={{ y: -2 }}
              transition={{ type: 'spring', stiffness: 300, damping: 22 }}
              onMouseMove={handleCardMouseMove}
              onMouseLeave={() => setHoverPos(null)}
              className="glow-border group relative flex flex-col overflow-hidden rounded-2xl border bg-black/90 shadow-glow"
            >
              <div className="flex items-center justify-between gap-2 border-b border-accent/25 bg-card/80 px-3.5 py-2.5 backdrop-blur-md">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-accent" />
                  <span className="truncate bg-accent-gradient bg-clip-text text-xs font-extrabold uppercase tracking-wider text-transparent">
                    UNLET-ADAS Enhanced
                  </span>
                </div>
                <div className="flex shrink-0 items-center gap-1.5">
                  {gainPct !== null && gainPct > 0 && (
                    <span className="whitespace-nowrap rounded-full border border-ok/30 bg-ok/15 px-2.5 py-0.5 font-mono text-[11px] font-bold tabular-nums text-ok">
                      +{gainPct}% Lum
                    </span>
                  )}
                  {objectCount !== undefined && objectCount > 0 && (
                    <span className="whitespace-nowrap rounded-full border border-accent/30 bg-accent/15 px-2.5 py-0.5 font-mono text-[11px] font-bold tabular-nums text-accent">
                      {objectCount} {objectCount === 1 ? 'Object' : 'Objects'}
                    </span>
                  )}
                </div>
              </div>

              <div className="relative flex min-h-[280px] max-h-[540px] flex-1 items-center justify-center overflow-hidden bg-black p-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={after}
                  alt="UNLET-ADAS Enhanced"
                  className="max-h-[520px] w-auto max-w-full rounded-lg object-contain transition-transform duration-300"
                  style={
                    zoomEnabled && hoverPos
                      ? {
                          transform: 'scale(2.2)',
                          transformOrigin: `${hoverPos.x}% ${hoverPos.y}%`,
                        }
                      : undefined
                  }
                  draggable={false}
                />
              </div>
            </motion.div>
          </motion.div>
        ) : (
          <motion.div
            key="split"
            ref={containerRef}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="glow-border relative mx-auto flex max-h-[560px] w-full cursor-ew-resize select-none items-center justify-center overflow-hidden rounded-2xl border bg-black"
            onMouseDown={(e) => {
              dragging.current = true;
              updateFromClientX(e.clientX);
            }}
            onMouseMove={(e) => {
              if (dragging.current) updateFromClientX(e.clientX);
            }}
            onMouseUp={() => (dragging.current = false)}
            onMouseLeave={() => (dragging.current = false)}
            onTouchStart={(e) => updateFromClientX(e.touches[0].clientX)}
            onTouchMove={(e) => updateFromClientX(e.touches[0].clientX)}
          >
            {/* Base layer: Enhanced */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={after}
              alt="Enhanced"
              className="block max-h-[560px] w-full object-contain"
              draggable={false}
            />
            {/* Clipped layer: Original */}
            <div
              className="pointer-events-none absolute inset-0 flex items-center justify-center"
              style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={before}
                alt="Original"
                className="block max-h-[560px] w-full object-contain"
                draggable={false}
              />
            </div>

            {/* Glowing Split Divider */}
            <div
              className="pointer-events-none absolute inset-y-0 w-0.5 bg-accent shadow-[0_0_12px_#5eead4]"
              style={{ left: `${pos}%` }}
            >
              <div className="
                absolute left-1/2 top-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2
                items-center justify-center rounded-full border-2 border-white/40
                bg-accent-gradient text-base shadow-glow
              ">
                <span className="text-xs font-extrabold text-base">⇔</span>
              </div>
            </div>

            <span className="pointer-events-none absolute left-3 top-3 rounded-full border border-white/10 bg-black/70 px-2.5 py-1 text-[11px] font-bold uppercase tracking-wider text-textMain backdrop-blur-md">
              Original
            </span>
            <span className="pointer-events-none absolute right-3 top-3 rounded-full border border-accent/30 bg-black/70 px-2.5 py-1 text-[11px] font-bold uppercase tracking-wider text-accent backdrop-blur-md">
              UNLET Enhanced
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
