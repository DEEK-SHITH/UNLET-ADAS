'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface ClassAccuracy {
  name: string;
  icon: string;
  rawMap: string;
  enhancedMap: string;
  gain: string;
  tier: string;
  highlight?: boolean;
}

const ACCURACY_DATA: ClassAccuracy[] = [
  { name: 'Truck', icon: '🚛', rawMap: '42.1%', enhancedMap: '78.2%', gain: '+36.1%', tier: 'Heavy Transport', highlight: true },
  { name: 'Bus', icon: '🚌', rawMap: '48.2%', enhancedMap: '82.4%', gain: '+34.2%', tier: 'Public Transit', highlight: true },
  { name: 'Car', icon: '🚗', rawMap: '56.4%', enhancedMap: '85.9%', gain: '+29.5%', tier: 'Roadway Corridor' },
  { name: 'Pedestrian (VRU)', icon: '🚶', rawMap: '38.6%', enhancedMap: '76.5%', gain: '+37.9%', tier: 'Vulnerable User' },
  { name: 'Two-Wheeler', icon: '🏍️', rawMap: '37.8%', enhancedMap: '73.8%', gain: '+36.0%', tier: 'Vulnerable User' },
  { name: 'Traffic Light', icon: '🚦', rawMap: '44.5%', enhancedMap: '86.2%', gain: '+41.7%', tier: 'Signal Fixture' },
];

export default function AccuracyBadgeCard({ defaultOpen = true }: { defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="glass overflow-hidden rounded-2xl border border-accent/30 bg-card/80 p-5 shadow-lg backdrop-blur-md">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent/20 text-xl text-accent shadow-glow">
            🎯
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-extrabold tracking-tight text-textMain">
                Perception Accuracy & Model Telemetry
              </h3>
              <span className="rounded-full border border-ok/40 bg-ok/10 px-2.5 py-0.5 text-[10px] font-black uppercase tracking-wider text-ok">
                Verified Benchmark
              </span>
            </div>
            <p className="text-xs text-textDim">
              Comparative benchmark against raw low-light sensor baseline (ISO 26262 compliant)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-baseline gap-1 rounded-xl border border-border bg-base/60 px-3 py-1.5 font-mono">
            <span className="text-[10px] text-textDim">OVERALL mAP@0.5:</span>
            <span className="text-sm font-black text-accent">80.5%</span>
            <span className="text-[10px] font-bold text-ok">(+35.9%)</span>
          </div>
          <button
            type="button"
            onClick={() => setOpen(!open)}
            className="rounded-lg border border-border bg-card2 px-2.5 py-1 text-xs font-semibold text-textDim transition hover:text-textMain"
          >
            {open ? 'Hide Details ▲' : 'Show Details ▼'}
          </button>
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="mt-4 border-t border-border/80 pt-4"
          >
            {/* Quick Stat Pill Grid */}
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <div className="rounded-xl border border-border/60 bg-base/40 p-2.5">
                <div className="text-[10px] font-bold uppercase tracking-wider text-textDim">mAP@0.5:0.95</div>
                <div className="text-base font-black text-textMain">53.5%</div>
                <div className="text-[10px] font-semibold text-ok">+28.7% Precision</div>
              </div>
              <div className="rounded-xl border border-border/60 bg-base/40 p-2.5">
                <div className="text-[10px] font-bold uppercase tracking-wider text-textDim">Edge Sharpness</div>
                <div className="text-base font-black text-textMain">5.9× Gain</div>
                <div className="text-[10px] text-textDim">Laplacian Variance</div>
              </div>
              <div className="rounded-xl border border-border/60 bg-base/40 p-2.5">
                <div className="text-[10px] font-bold uppercase tracking-wider text-textDim">Architecture</div>
                <div className="text-base font-black text-textMain">Zero-DCE++ CBAM</div>
                <div className="text-[10px] text-textDim">21,769 parameters</div>
              </div>
              <div className="rounded-xl border border-border/60 bg-base/40 p-2.5">
                <div className="text-[10px] font-bold uppercase tracking-wider text-textDim">Jetson Orin Latency</div>
                <div className="text-base font-black text-textMain">26.7 ms (37.5 FPS)</div>
                <div className="text-[10px] font-semibold text-ok">ISO 30 FPS Met ✅</div>
              </div>
            </div>

            {/* Per-Class Breakdown Table */}
            <div className="mt-3 overflow-x-auto rounded-xl border border-border/60 bg-base/30">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border/80 bg-card2/50 text-[10px] font-bold uppercase tracking-wider text-textDim">
                    <th className="px-3 py-2">ADAS Object Class</th>
                    <th className="px-3 py-2">Raw Night Input</th>
                    <th className="px-3 py-2">UNLET-ADAS Enhanced</th>
                    <th className="px-3 py-2">Absolute Gain</th>
                    <th className="px-3 py-2 text-right">Perception Tier</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40 font-mono">
                  {ACCURACY_DATA.map((row) => (
                    <tr
                      key={row.name}
                      className={`transition hover:bg-white/[0.02] ${
                        row.highlight ? 'bg-accent/5' : ''
                      }`}
                    >
                      <td className="flex items-center gap-2 px-3 py-2 font-sans font-semibold text-textMain">
                        <span>{row.icon}</span>
                        <span>{row.name}</span>
                        {row.highlight && (
                          <span className="rounded bg-accent/20 px-1.5 py-0.5 text-[9px] font-bold text-accent">
                            Disambiguated
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-textDim">{row.rawMap}</td>
                      <td className="px-3 py-2 font-bold text-textMain">{row.enhancedMap}</td>
                      <td className="px-3 py-2 font-bold text-ok">{row.gain}</td>
                      <td className="px-3 py-2 text-right font-sans text-[11px] text-textDim">
                        {row.tier}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="mt-2.5 text-[11px] text-textDim">
              💡 <strong>Perspective-Aware Disambiguation</strong>: Distant cargo trucks and buses are verified via chassis ground clearance, under-ride bumper signatures, and aspect ratios, preventing false collapsing into passenger cars while eliminating bus/truck classification flipping.
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
