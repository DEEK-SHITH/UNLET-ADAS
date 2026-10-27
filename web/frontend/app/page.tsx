'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DropZone from '@/components/DropZone';
import Toggle from '@/components/Toggle';
import SliderField from '@/components/SliderField';
import CompareView from '@/components/CompareView';
import DetectionList from '@/components/DetectionList';
import StatCard from '@/components/StatCard';
import PageTransition from '@/components/PageTransition';
import {
  AppConfig,
  ImageOptions,
  ImageResult,
  enhanceImage,
  fetchConfig,
} from '@/lib/api';

const DEFAULT_OPTS: ImageOptions = {
  adaptive: true,
  det_conf: 0.25,
  det_imgsz: 640,
  enable_detect: true,
  enable_pothole: false,
  enable_signs: false,
  enable_lanes: false,
  use_depth_risk: false,
};

export default function ImagePage() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [opts, setOpts] = useState<ImageOptions>(DEFAULT_OPTS);
  const [result, setResult] = useState<ImageResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchConfig()
      .then(setConfig)
      .catch(() => setConfig(null));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function onFile(f: File) {
    setFile(f);
    setResult(null);
    setError(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(f);
    });
  }

  async function onEnhance() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const r = await enhanceImage(file, opts);
      setResult(r);
    } catch (e: any) {
      setError(e.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  }

  return (
    <PageTransition>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[340px_1fr]">
        {/* Controls */}
        <motion.aside
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.05 }}
          className="glass h-fit rounded-2xl p-5"
        >
          <h2 className="mb-1 bg-accent-gradient bg-clip-text text-lg font-bold text-transparent">
            Image Enhancement
          </h2>
          <p className="mb-4 text-sm text-textDim">
            Upload a low-light frame and run it through the enhancer +
            detection pipeline.
          </p>

          <DropZone
            accept="image/*"
            onFile={onFile}
            fileName={file?.name}
            hint="JPG, PNG, or BMP"
          />

          <div className="mt-5 border-t border-border pt-4">
            <div className="mb-2 text-xs font-bold uppercase tracking-wide text-textDim">
              Enhancement
            </div>
            <Toggle
              label="Adaptive day/night blending"
              hint="Skip already well-lit frames"
              checked={opts.adaptive}
              onChange={(v) => setOpts({ ...opts, adaptive: v })}
            />
          </div>

          <div className="mt-4 border-t border-border pt-4">
            <div className="mb-2 text-xs font-bold uppercase tracking-wide text-textDim">
              Detection
            </div>
            <Toggle
              label="Object detection"
              hint={config?.has_detect ? 'Vehicles, people, signs' : 'Model unavailable'}
              checked={opts.enable_detect}
              disabled={config ? !config.has_detect : false}
              onChange={(v) => setOpts({ ...opts, enable_detect: v })}
            />
            <Toggle
              label="Pothole detection"
              hint={config?.has_pothole ? undefined : 'Model unavailable'}
              checked={opts.enable_pothole}
              disabled={config ? !config.has_pothole : true}
              onChange={(v) => setOpts({ ...opts, enable_pothole: v })}
            />
            <Toggle
              label="Road sign detection"
              hint={config?.has_signs ? undefined : 'Model unavailable'}
              checked={opts.enable_signs}
              disabled={config ? !config.has_signs : true}
              onChange={(v) => setOpts({ ...opts, enable_signs: v })}
            />
            <Toggle
              label="Lane detection"
              checked={opts.enable_lanes}
              onChange={(v) => setOpts({ ...opts, enable_lanes: v })}
            />
            <Toggle
              label="Depth-based risk"
              hint={config?.has_depth ? 'MiDaS proximity' : 'Falls back to box geometry'}
              checked={opts.use_depth_risk}
              onChange={(v) => setOpts({ ...opts, use_depth_risk: v })}
            />
          </div>

          <div className="mt-4 border-t border-border pt-4">
            <SliderField
              label="Confidence threshold"
              value={opts.det_conf}
              min={0.1}
              max={0.9}
              step={0.05}
              format={(v) => v.toFixed(2)}
              onChange={(v) => setOpts({ ...opts, det_conf: v })}
            />
            <SliderField
              label="Detection resolution"
              value={opts.det_imgsz}
              min={320}
              max={960}
              step={32}
              format={(v) => `${v}px`}
              onChange={(v) => setOpts({ ...opts, det_imgsz: v })}
            />
          </div>

          <motion.button
            whileHover={{ scale: file && !loading ? 1.02 : 1 }}
            whileTap={{ scale: file && !loading ? 0.97 : 1 }}
            onClick={onEnhance}
            disabled={!file || loading}
            className="mt-5 w-full rounded-xl bg-accent-gradient px-4 py-2.5 text-sm font-bold text-base shadow-glow transition disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-base/30 border-t-base" />
                Enhancing…
              </span>
            ) : (
              'Enhance Image'
            )}
          </motion.button>

          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.aside>

        {/* Results */}
        <motion.section
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
        >
          {!result && !loading && (
            <div className="glass relative flex min-h-[380px] flex-col items-center justify-center overflow-hidden rounded-2xl border-dashed p-6 text-textDim">
              {previewUrl ? (
                <div className="flex w-full flex-col items-center gap-4">
                  <div className="relative max-h-[520px] overflow-hidden rounded-xl border border-border bg-base/80 shadow-2xl">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="max-h-[480px] w-auto object-contain"
                    />
                    <div className="absolute left-3 top-3 rounded-full border border-white/15 bg-base/80 px-3 py-1 text-xs font-semibold text-textMain backdrop-blur-md">
                      Ready for Enhancement · {file?.name}
                    </div>
                  </div>
                  <p className="text-xs text-textDim">
                    Click <span className="font-bold text-accent">Enhance Image</span> in the left panel to run Zero-DCE + YOLOv8s side-by-side analysis.
                  </p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3 text-center">
                  <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-accent/30 bg-accent/10 text-2xl shadow-glow">
                    🌙
                  </div>
                  <div className="text-sm font-semibold text-textMain">
                    Upload a low-light road or indoor frame
                  </div>
                  <p className="max-w-sm text-xs text-textDim">
                    Compare the raw dark frame side-by-side with Zero-DCE curve enhancement, synchronized 2.2× magnifier loupe, and YOLOv8s object detection.
                  </p>
                </div>
              )}
            </div>
          )}

          {loading && (
            <div className="glass relative flex min-h-[420px] flex-col items-center justify-center overflow-hidden rounded-2xl border border-accent/30 bg-panel p-6">
              {previewUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={previewUrl}
                  alt="Scanning"
                  className="max-h-[440px] w-auto rounded-xl object-contain opacity-35 blur-[1px]"
                />
              )}
              {/* Animated laser scanline */}
              <div className="pointer-events-none absolute inset-x-0 top-0 h-28 w-full animate-scanline bg-gradient-to-b from-transparent via-accent/30 to-accent/5 border-b-2 border-accent shadow-glow" />

              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-base/40 backdrop-blur-[2px]">
                <div className="flex items-center gap-2.5 rounded-full border border-accent/40 bg-card/90 px-4 py-2 text-xs font-bold text-accent shadow-glow">
                  <span className="h-2.5 w-2.5 animate-ping rounded-full bg-accent" />
                  NEURAL PERCEPTION PIPELINE ACTIVE
                </div>
                <div className="flex flex-wrap justify-center gap-2 text-[11px] font-mono text-textDim">
                  <span className="rounded-md border border-border bg-base/80 px-2.5 py-1 text-textMain">
                    1. Zero-DCE Illumination
                  </span>
                  <span className="rounded-md border border-border bg-base/80 px-2.5 py-1 text-textMain">
                    2. YOLOv8s Detection
                  </span>
                  <span className="rounded-md border border-border bg-base/80 px-2.5 py-1 text-textMain">
                    3. Proximity Risk HUD
                  </span>
                </div>
              </div>
            </div>
          )}

          {result && (
            <div className="glass rounded-2xl p-5">
              <CompareView
                before={result.original_image}
                after={result.enhanced_image}
                origBrightness={result.brightness}
                enhBrightness={result.enhanced_brightness}
                objectCount={result.detections.length}
              />

              <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-5">
                <StatCard
                  label="Raw Lum."
                  value={result.brightness.toFixed(3)}
                  sublabel="Original frame"
                />
                <StatCard
                  label="Enh. Lum."
                  value={(result.enhanced_brightness ?? result.brightness).toFixed(3)}
                  sublabel={
                    result.enhanced_brightness && result.brightness > 0
                      ? `+${(result.enhanced_brightness / result.brightness).toFixed(1)}× gain`
                      : 'Zero-DCE output'
                  }
                />
                <StatCard
                  label="Objects"
                  value={`${result.detections.length}`}
                  sublabel={`Conf ≥ ${result.effective_confidence.toFixed(2)}`}
                />
                <StatCard
                  label="High Risk"
                  value={result.high_risk ? 'ALERT' : 'Clear'}
                  danger={result.high_risk}
                  sublabel={result.high_risk ? 'Proximity warning' : 'Safe distance'}
                />
                <StatCard
                  label="Latency"
                  value={`${(result.processing_ms / 1000).toFixed(2)}s`}
                  sublabel={`${result.processing_ms.toFixed(0)} ms total`}
                />
              </div>

              <DetectionList title="Detected Objects" items={result.detections} />
              <DetectionList title="Potholes" items={result.potholes} />
              <DetectionList title="Road Signs" items={result.signs} />
              {result.lanes_found && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-ok/30 bg-ok/10 px-3 py-1 text-xs font-semibold text-ok"
                >
                  ✓ Lane boundaries detected and overlaid
                </motion.div>
              )}
            </div>
          )}
        </motion.section>
      </div>
    </PageTransition>
  );
}
