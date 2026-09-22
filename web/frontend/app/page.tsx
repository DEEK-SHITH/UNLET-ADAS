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

  function onFile(f: File) {
    setFile(f);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(f));
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
            <div className="glass flex aspect-video items-center justify-center rounded-2xl border-dashed text-textDim">
              {previewUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="h-full w-full rounded-2xl object-contain"
                />
              ) : (
                'Upload an image to get started'
              )}
            </div>
          )}

          {loading && (
            <div className="shimmer flex aspect-video items-center justify-center rounded-2xl border border-border bg-panel text-textDim">
              Running enhancement + detection…
            </div>
          )}

          {result && (
            <div className="glass rounded-2xl p-5">
              <CompareView
                before={result.original_image}
                after={result.enhanced_image}
              />

              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <StatCard label="Brightness" value={result.brightness.toFixed(3)} />
                <StatCard
                  label="Eff. confidence"
                  value={result.effective_confidence.toFixed(2)}
                />
                <StatCard
                  label="High risk"
                  value={result.high_risk ? 'Yes' : 'No'}
                  danger={result.high_risk}
                />
                <StatCard
                  label="Processing"
                  value={`${(result.processing_ms / 1000).toFixed(1)}s`}
                />
              </div>

              <DetectionList title="Objects" items={result.detections} />
              <DetectionList title="Potholes" items={result.potholes} />
              <DetectionList title="Signs" items={result.signs} />
              {result.lanes_found && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-3 text-xs font-medium text-accent"
                >
                  ✓ Lane boundaries detected and drawn
                </motion.div>
              )}
            </div>
          )}
        </motion.section>
      </div>
    </PageTransition>
  );
}
