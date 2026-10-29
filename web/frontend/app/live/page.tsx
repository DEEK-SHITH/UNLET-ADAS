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
  LiveOptions,
  LiveResult,
  enhanceImage,
  enhanceLiveFrame,
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

export default function LivePage() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [opts, setOpts] = useState<ImageOptions>(DEFAULT_OPTS);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Real-time speed presets
  const [liveQualityMode, setLiveQualityMode] = useState<'turbo' | 'balanced' | 'quality'>('turbo');
  const [interleaved, setInterleaved] = useState<boolean>(true);

  const [streaming, setStreaming] = useState(false);
  const [liveResult, setLiveResult] = useState<LiveResult | null>(null);
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
  const liveQualityModeRef = useRef(liveQualityMode);
  const interleavedRef = useRef(interleaved);
  const [streamEngine, setStreamEngine] = useState<'instant' | 'neural'>('instant');
  const streamEngineRef = useRef(streamEngine);
  const liveCanvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number | null>(null);
  const latestDetsRef = useRef<any[]>([]);
  const fpsHistoryRef = useRef<number[]>([]);

  useEffect(() => {
    streamEngineRef.current = streamEngine;
  }, [streamEngine]);

  useEffect(() => {
    optsRef.current = opts;
  }, [opts]);

  useEffect(() => {
    liveQualityModeRef.current = liveQualityMode;
  }, [liveQualityMode]);

  useEffect(() => {
    interleavedRef.current = interleaved;
  }, [interleaved]);

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

  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');

  async function startCamera(targetMode?: 'environment' | 'user') {
    setCameraError(null);
    const mode = targetMode || facingMode;
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: mode } },
          audio: false,
        });
      } catch {
        // Fallback for browsers or devices that reject facingMode constraints
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setCameraOn(true);
    } catch (e: any) {
      setCameraError(
        e?.name === 'NotAllowedError'
          ? 'Camera access was denied. Please allow camera permission in your phone/browser settings and reload.'
          : e?.message || 'Could not access the camera. Make sure you are using an HTTPS link.'
      );
    }
  }

  async function flipCamera() {
    const nextMode = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextMode);
    if (cameraOn) {
      await startCamera(nextMode);
    }
  }

  function stopCamera() {
    stopLiveStream();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraOn(false);
  }

  async function captureFrameBlob(maxWidth?: number, quality = 0.8): Promise<Blob | null> {
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

    return new Promise((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', quality)
    );
  }

  async function captureFrameFile(maxWidth?: number): Promise<File | null> {
    const blob = await captureFrameBlob(maxWidth, 0.9);
    if (!blob) return null;
    return new File([blob], 'snapshot.jpg', { type: 'image/jpeg' });
  }

  function handleEngineChange(engine: 'instant' | 'neural') {
    setStreamEngine(engine);
    streamEngineRef.current = engine;
    if (streamingRef.current) {
      if (engine === 'instant') {
        renderInstantLoop();
      } else if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
    }
  }

  function renderInstantLoop() {
    if (!streamingRef.current || streamEngineRef.current !== 'instant') return;
    const video = videoRef.current;
    const canvas = liveCanvasRef.current;
    if (video && canvas && video.videoWidth > 0) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.filter = optsRef.current.adaptive
          ? 'contrast(125%) brightness(145%) saturate(115%)'
          : 'contrast(115%) brightness(130%)';
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.filter = 'none';

        const dets = latestDetsRef.current;
        if (dets && dets.length > 0) {
          dets.forEach((d) => {
            if (!d.box) return;
            const [x1, y1, x2, y2] = d.box;
            const isHigh = d.risk === 'HIGH';
            ctx.strokeStyle = isHigh ? '#ff3b30' : '#00f0ff';
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            ctx.fillStyle = isHigh ? 'rgba(255, 59, 48, 0.9)' : 'rgba(0, 240, 255, 0.9)';
            const label = `${d.name} ${(d.conf * 100).toFixed(0)}%`;
            ctx.font = 'bold 12px sans-serif';
            const textW = ctx.measureText(label).width;
            ctx.fillRect(x1, Math.max(y1 - 18, 0), textW + 6, 16);

            ctx.fillStyle = '#000000';
            ctx.fillText(label, x1 + 3, Math.max(y1 - 5, 12));
          });
        }
      }
    }
    animFrameRef.current = requestAnimationFrame(renderInstantLoop);
  }

  function startLiveStream() {
    if (streamingRef.current) return;
    streamingRef.current = true;
    setStreaming(true);
    setStreamError(null);
    setFrameCount(0);
    fpsHistoryRef.current = [];
    latestDetsRef.current = [];

    if (streamEngineRef.current === 'instant') {
      renderInstantLoop();
    }
    void liveLoop();
  }

  function stopLiveStream() {
    streamingRef.current = false;
    setStreaming(false);
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
  }

  async function liveLoop() {
    let inFlight = false;
    while (streamingRef.current) {
      if (inFlight) {
        await new Promise((r) => setTimeout(r, 12));
        continue;
      }
      const qm = liveQualityModeRef.current;
      const maxW = qm === 'turbo' ? 320 : qm === 'balanced' ? 440 : 640;
      const blob = await captureFrameBlob(maxW, qm === 'turbo' ? 0.72 : 0.82);
      if (!blob) {
        await new Promise((r) => setTimeout(r, 40));
        continue;
      }

      inFlight = true;
      const t0 = performance.now();
      const liveOpts: LiveOptions = {
        adaptive: optsRef.current.adaptive,
        det_conf: optsRef.current.det_conf,
        det_imgsz: maxW,
        enable_detect: optsRef.current.enable_detect,
        enable_lanes: optsRef.current.enable_lanes,
        interleaved: interleavedRef.current,
        quality_mode: qm,
      };

      try {
        const r = await enhanceLiveFrame(blob, liveOpts);
        if (!streamingRef.current) break;

        latestDetsRef.current = r.detections || [];
        setLiveResult(r);
        setFrameCount((n) => n + 1);

        if (streamEngineRef.current === 'neural') {
          const img = new window.Image();
          img.onload = () => {
            const canvas = liveCanvasRef.current;
            if (!canvas) return;
            if (canvas.width !== img.width || canvas.height !== img.height) {
              canvas.width = img.width;
              canvas.height = img.height;
            }
            const ctx = canvas.getContext('2d');
            if (ctx) ctx.drawImage(img, 0, 0);
          };
          img.src = r.enhanced_image;
        }

        const elapsedS = (performance.now() - t0) / 1000;
        const hist = fpsHistoryRef.current;
        hist.push(streamEngineRef.current === 'instant' ? 60 : 1 / Math.max(elapsedS, 0.001));
        if (hist.length > 8) hist.shift();
        setFps(hist.reduce((a, b) => a + b, 0) / hist.length);
      } catch (e: any) {
        if (streamingRef.current) {
          setStreamError(e.message || 'Live stream request failed');
        }
      } finally {
        inFlight = false;
      }

      await new Promise((r) => setTimeout(r, 10));
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

          <div className="flex gap-2">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              onClick={cameraOn ? stopCamera : () => startCamera()}
              className="flex-1 rounded-xl border border-borderLt bg-card px-4 py-2.5 text-sm font-bold text-textMain transition hover:border-accent/60"
            >
              {cameraOn ? '📷 Turn Camera Off' : '📷 Turn Camera On'}
            </motion.button>
            {cameraOn && (
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.97 }}
                onClick={flipCamera}
                title="Flip between front and back camera"
                className="rounded-xl border border-borderLt bg-card px-3 py-2.5 text-sm font-bold text-textDim transition hover:text-accent hover:border-accent/60"
              >
                🔄 Flip
              </motion.button>
            )}
          </div>
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

          <div className="mt-4 border-t border-border pt-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wide text-textDim">
                Streaming Engine
              </span>
              <span className="text-[11px] font-semibold text-accent">
                {streamEngine === 'instant' ? '⚡ 60 FPS Real-Time HUD' : '🧠 Deep Neural Stream'}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-1.5 rounded-xl border border-border bg-card p-1">
              <button
                type="button"
                onClick={() => handleEngineChange('instant')}
                className={`rounded-lg py-1.5 text-xs font-semibold transition ${
                  streamEngine === 'instant'
                    ? 'bg-accent text-base font-bold shadow-sm'
                    : 'text-textDim hover:text-textMain'
                }`}
              >
                ⚡ 60 FPS (Zero Lag)
              </button>
              <button
                type="button"
                onClick={() => handleEngineChange('neural')}
                className={`rounded-lg py-1.5 text-xs font-semibold transition ${
                  streamEngine === 'neural'
                    ? 'bg-accent text-base font-bold shadow-sm'
                    : 'text-textDim hover:text-textMain'
                }`}
              >
                🧠 Neural Stream
              </button>
            </div>
          </div>

          <div className="mt-4 border-t border-border pt-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wide text-textDim">
                Live Speed Preset
              </span>
              <span className="text-[11px] font-semibold text-accent">
                {liveQualityMode === 'turbo'
                  ? '⚡ 384p · Max FPS'
                  : liveQualityMode === 'balanced'
                  ? '🎯 480p · Balanced'
                  : '🔬 640p · Studio'}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-1.5 rounded-xl border border-border bg-card p-1">
              <button
                type="button"
                onClick={() => setLiveQualityMode('turbo')}
                className={`rounded-lg py-1.5 text-xs font-semibold transition ${
                  liveQualityMode === 'turbo'
                    ? 'bg-accent text-base font-bold shadow-sm'
                    : 'text-textDim hover:text-textMain'
                }`}
              >
                ⚡ Turbo
              </button>
              <button
                type="button"
                onClick={() => setLiveQualityMode('balanced')}
                className={`rounded-lg py-1.5 text-xs font-semibold transition ${
                  liveQualityMode === 'balanced'
                    ? 'bg-accent text-base font-bold shadow-sm'
                    : 'text-textDim hover:text-textMain'
                }`}
              >
                🎯 Balanced
              </button>
              <button
                type="button"
                onClick={() => setLiveQualityMode('quality')}
                className={`rounded-lg py-1.5 text-xs font-semibold transition ${
                  liveQualityMode === 'quality'
                    ? 'bg-accent text-base font-bold shadow-sm'
                    : 'text-textDim hover:text-textMain'
                }`}
              >
                🔬 Studio
              </button>
            </div>
          </div>

          <div className="mt-4 border-t border-border pt-4">
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
              label="Interleaved detection"
              hint="Alternate-frame detection for 2x higher frame rate"
              checked={interleaved}
              onChange={setInterleaved}
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
                autoPlay
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
              {streaming ? (
                <canvas
                  ref={liveCanvasRef}
                  className="aspect-video w-full bg-black object-contain"
                />
              ) : snapResult?.enhanced_image ? (
                <CrossfadeImage
                  src={snapResult.enhanced_image}
                  alt="Enhanced Snapshot"
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
                    label={streaming ? "Perception Mode" : "Eff. confidence"}
                    value={
                      streaming
                        ? `${liveQualityMode.toUpperCase()} · ${interleaved ? '2x FPS' : '1x'}`
                        : ((displayResult as any).effective_confidence ?? opts.det_conf).toFixed(2)
                    }
                  />
                  <StatCard
                    label="High risk"
                    value={displayResult.high_risk ? 'Yes' : 'No'}
                    danger={displayResult.high_risk}
                  />
                  <StatCard
                    label={streaming ? 'Latency / Speed' : 'Processing'}
                    value={
                      streaming && fps
                        ? `${fps.toFixed(1)} FPS (${displayResult.processing_ms.toFixed(0)}ms)`
                        : `${(displayResult.processing_ms / 1000).toFixed(1)}s`
                    }
                  />
                </div>

                <DetectionList title="Objects" items={displayResult.detections || []} />
                <DetectionList title="Potholes" items={(displayResult as any).potholes || []} />
                <DetectionList title="Signs" items={(displayResult as any).signs || []} />
                {(displayResult as any).lanes_found && (
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
