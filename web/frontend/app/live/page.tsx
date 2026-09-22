'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Toggle from '@/components/Toggle';
import SliderField from '@/components/SliderField';
import DetectionList from '@/components/DetectionList';
import StatCard from '@/components/StatCard';
import CrossfadeImage from '@/components/CrossfadeImage';
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

// Frames captured for the continuous stream are downscaled before
// upload -- backend enhance+detect time scales with pixel count, so
// this is the single biggest lever on perceived smoothness/FPS. A
// one-off snapshot keeps full camera resolution instead.
const LIVE_CAPTURE_MAX_WIDTH = 768;

export default function LivePage() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [opts, setOpts] = useState<ImageOptions>(DEFAULT_OPTS);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const [streaming, setStreaming] = useState(false);
  const [liveResult, setLiveResult] = useState<ImageResult | null>(null);
  const [fps, setFps] = useState<number | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [streamError, setStreamError] = useState<string | null>(null);

  const [snapLoading, setSnapLoading] = useState(false);
  const [snapResult, setSnapResult] = useState<ImageResult | null>(null);
  const [snapError, setSnapError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const streamingRef = useRef(false);
  const optsRef = useRef(opts);
  const fpsHistoryRef = useRef<number[]>([]);

  useEffect(() => {
    optsRef.current = opts;
  }, [opts]);

  useEffect(() => {
    fetchConfig()
      .then(setConfig)
      .catch(() => setConfig(null));
    return () => {
      streamingRef.current = false;
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function startCamera() {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
    } catch (e: any) {
      setCameraError(
        e?.name === 'NotAllowedError'
          ? 'Camera access was denied. Allow camera permission for this site and try again.'
          : e?.message || 'Could not access the camera.'
      );
    }
  }

  function stopCamera() {
    stopLiveStream();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraOn(false);
  }

  async function captureFrameFile(maxWidth?: number): Promise<File | null> {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0) return null;

    let w = video.videoWidth;
    let h = video.videoHeight;
    if (maxWidth && w > maxWidth) {
      h = Math.round((h * maxWidth) / w);
      w = maxWidth;
    }
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, w, h);

    const blob: Blob | null = await new Promise((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', 0.85)
    );
    if (!blob) return null;
    return new File([blob], 'live-frame.jpg', { type: 'image/jpeg' });
  }

  function startLiveStream() {
    if (streamingRef.current) return;
    streamingRef.current = true;
    setStreaming(true);
    setStreamError(null);
    setFrameCount(0);
    fpsHistoryRef.current = [];
    void liveLoop();
  }

  function stopLiveStream() {
    streamingRef.current = false;
    setStreaming(false);
  }

  // Self-pacing loop: captures a (downscaled) frame, waits for the
  // backend's enhance+detect response, displays it, then immediately
  // captures the next one. Real frame rate is whatever the backend can
  // sustain rather than a fixed interval -- this way it never piles up
  // requests faster than they can be processed.
  async function liveLoop() {
    while (streamingRef.current) {
      const file = await captureFrameFile(LIVE_CAPTURE_MAX_WIDTH);
      if (!file) {
        await new Promise((r) => setTimeout(r, 100));
        continue;
      }
      const t0 = performance.now();
      try {
        const r = await enhanceImage(file, optsRef.current);
        if (!streamingRef.current) break;
        setLiveResult(r);
        setFrameCount((n) => n + 1);

        const elapsedS = (performance.now() - t0) / 1000;
        const hist = fpsHistoryRef.current;
        hist.push(1 / Math.max(elapsedS, 0.001));
        if (hist.length > 8) hist.shift();
        setFps(hist.reduce((a, b) => a + b, 0) / hist.length);
      } catch (e: any) {
        setStreamError(e.message || 'Live stream request failed');
        streamingRef.current = false;
        setStreaming(false);
        break;
      }
    }
  }

  async function takeSnapshot() {
    const file = await captureFrameFile();
    if (!file || !cameraOn) return;
    setSnapLoading(true);
    setSnapError(null);
    try {
      const r = await enhanceImage(file, opts);
      setSnapResult(r);
    } catch (e: any) {
      setSnapError(e.message || 'Something went wrong');
    } finally {
      setSnapLoading(false);
    }
  }

  const displayImage = streaming
    ? liveResult?.enhanced_image
    : snapResult?.enhanced_image;
  const displayResult = streaming ? liveResult : snapResult;

  return (
    <PageTransition>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[340px_1fr]">
        <motion.aside
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.05 }}
          className="glass h-fit rounded-2xl p-5"
        >
          <h2 className="mb-1 bg-accent-gradient bg-clip-text text-lg font-bold text-transparent">
            Live Camera
          </h2>
          <p className="mb-4 text-sm text-textDim">
            Continuously enhances your browser&apos;s camera feed and runs
            it through detection — frame rate is whatever your backend
            (CPU or GPU) can sustain, shown live below.
          </p>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={cameraOn ? stopCamera : startCamera}
            className="w-full rounded-xl border border-borderLt bg-card px-4 py-2.5 text-sm font-bold text-textMain transition hover:border-accent/60"
          >
            {cameraOn ? '📷 Turn Camera Off' : '📷 Turn Camera On'}
          </motion.button>
          <AnimatePresence>
            {cameraError && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger"
              >
                {cameraError}
              </motion.div>
            )}
          </AnimatePresence>

          <div className="mt-5 border-t border-border pt-4">
            <div className="mb-2 text-xs font-bold uppercase tracking-wide text-textDim">
              Enhancement
            </div>
            <Toggle
              label="Adaptive day/night blending"
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
          </div>

          <motion.button
            whileHover={{ scale: cameraOn ? 1.02 : 1 }}
            whileTap={{ scale: cameraOn ? 0.97 : 1 }}
            onClick={streaming ? stopLiveStream : startLiveStream}
            disabled={!cameraOn}
            className={`mt-5 w-full rounded-xl px-4 py-2.5 text-sm font-bold transition disabled:cursor-not-allowed disabled:opacity-40 ${
              streaming
                ? 'animate-pulse-ring bg-gradient-to-r from-danger to-red-700 text-white'
                : 'bg-accent-gradient text-base shadow-glow'
            }`}
          >
            {streaming ? '⏹ Stop Live Streaming' : '▶ Start Live Streaming'}
          </motion.button>

          <button
            onClick={takeSnapshot}
            disabled={!cameraOn || streaming || snapLoading}
            className="mt-2 w-full rounded-xl border border-borderLt bg-card px-4 py-2 text-xs font-bold text-textDim transition hover:border-accent/60 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {snapLoading ? 'Enhancing…' : '📸 Or take one single snapshot'}
          </button>

          <AnimatePresence>
            {(streamError || snapError) && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger"
              >
                {streamError || snapError}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.aside>

        <motion.section
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
        >
          <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="glass overflow-hidden rounded-2xl">
              <div className="flex items-center gap-2 border-b border-border/50 px-3 py-1.5 text-[11px] font-bold uppercase tracking-wide text-textDim">
                {streaming && (
                  <span className="h-1.5 w-1.5 animate-rec rounded-full bg-danger" />
                )}
                Raw camera
              </div>
              {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
              <video
                ref={videoRef}
                className={`aspect-video w-full bg-black object-contain ${cameraOn ? '' : 'hidden'}`}
                playsInline
                muted
              />
              {!cameraOn && (
                <div className="flex aspect-video items-center justify-center bg-black text-sm text-textDim">
                  Camera is off
                </div>
              )}
            </div>

            <div
              className={`glass overflow-hidden rounded-2xl ${streaming ? 'glow-border' : ''}`}
            >
              <div className="flex items-center justify-between border-b border-border/50 px-3 py-1.5">
                <span className="text-[11px] font-bold uppercase tracking-wide text-accent">
                  Enhanced {streaming ? '(live)' : ''}
                </span>
                <AnimatePresence>
                  {streaming && fps !== null && (
                    <motion.span
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      className="rounded-full bg-accent/15 px-2 py-0.5 text-[11px] font-bold tabular-nums text-accent"
                    >
                      {fps.toFixed(1)} FPS · frame {frameCount}
                    </motion.span>
                  )}
                </AnimatePresence>
              </div>
              {displayImage ? (
                <CrossfadeImage
                  src={displayImage}
                  alt="Enhanced"
                  className="aspect-video w-full bg-black"
                />
              ) : (
                <div className="flex aspect-video items-center justify-center bg-black text-sm text-textDim">
                  {cameraOn
                    ? 'Start live streaming or take a snapshot'
                    : 'Turn the camera on to begin'}
                </div>
              )}
            </div>
          </div>
          <canvas ref={canvasRef} className="hidden" />

          <AnimatePresence mode="wait">
            {displayResult && (
              <motion.div
                key={streaming ? 'live-stats' : 'snap-stats'}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25 }}
                className="glass rounded-2xl p-5"
              >
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <StatCard label="Brightness" value={displayResult.brightness.toFixed(3)} />
                  <StatCard
                    label="Eff. confidence"
                    value={displayResult.effective_confidence.toFixed(2)}
                  />
                  <StatCard
                    label="High risk"
                    value={displayResult.high_risk ? 'Yes' : 'No'}
                    danger={displayResult.high_risk}
                  />
                  <StatCard
                    label={streaming ? 'Frame time' : 'Processing'}
                    value={`${(displayResult.processing_ms / 1000).toFixed(1)}s`}
                  />
                </div>

                <DetectionList title="Objects" items={displayResult.detections} />
                <DetectionList title="Potholes" items={displayResult.potholes} />
                <DetectionList title="Signs" items={displayResult.signs} />
                {displayResult.lanes_found && (
                  <div className="mt-3 text-xs font-medium text-accent">
                    ✓ Lane boundaries detected and drawn
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.section>
      </div>
    </PageTransition>
  );
}
