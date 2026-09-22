'use client';

import { useEffect, useRef, useState } from 'react';
import Toggle from '@/components/Toggle';
import SliderField from '@/components/SliderField';
import DetectionList from '@/components/DetectionList';
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

  async function captureFrameFile(): Promise<File | null> {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

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

  // Self-pacing loop: captures a frame, waits for the backend's
  // enhance+detect response, displays it, then immediately captures
  // the next one. Real frame rate is whatever the backend can sustain
  // (CPU: roughly 1-5 fps depending which detectors are on; much
  // higher with a CUDA GPU) rather than a fixed interval -- this way
  // it never piles up requests faster than they can be processed.
  async function liveLoop() {
    while (streamingRef.current) {
      const file = await captureFrameFile();
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
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[340px_1fr]">
      <aside className="h-fit rounded-2xl border border-border bg-panel p-5">
        <h2 className="mb-1 text-lg font-bold">Live Camera</h2>
        <p className="mb-4 text-sm text-textDim">
          Continuously enhances your browser&apos;s camera feed and runs
          it through detection — frame rate is whatever your backend
          (CPU or GPU) can sustain, shown live below.
        </p>

        <button
          onClick={cameraOn ? stopCamera : startCamera}
          className="w-full rounded-xl border border-borderLt bg-card px-4 py-2.5 text-sm font-bold text-textMain transition hover:border-accent/60"
        >
          {cameraOn ? '📷 Turn Camera Off' : '📷 Turn Camera On'}
        </button>
        {cameraError && (
          <div className="mt-3 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
            {cameraError}
          </div>
        )}

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

        <button
          onClick={streaming ? stopLiveStream : startLiveStream}
          disabled={!cameraOn}
          className={`mt-5 w-full rounded-xl px-4 py-2.5 text-sm font-bold transition disabled:cursor-not-allowed disabled:opacity-40 ${
            streaming
              ? 'animate-pulse-ring bg-gradient-to-r from-danger to-red-700 text-white'
              : 'bg-accent-gradient text-base'
          }`}
        >
          {streaming ? '⏹ Stop Live Streaming' : '▶ Start Live Streaming'}
        </button>

        <button
          onClick={takeSnapshot}
          disabled={!cameraOn || streaming || snapLoading}
          className="mt-2 w-full rounded-xl border border-borderLt bg-card px-4 py-2 text-xs font-bold text-textDim transition hover:border-accent/60 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {snapLoading ? 'Enhancing…' : '📸 Or take one single snapshot'}
        </button>

        {(streamError || snapError) && (
          <div className="mt-3 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
            {streamError || snapError}
          </div>
        )}
      </aside>

      <section>
        <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="overflow-hidden rounded-2xl border border-border bg-black">
            <div className="border-b border-border/50 bg-panel px-3 py-1.5 text-[11px] font-bold uppercase tracking-wide text-textDim">
              Raw camera
            </div>
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video
              ref={videoRef}
              className={`aspect-video w-full object-contain ${cameraOn ? '' : 'hidden'}`}
              playsInline
              muted
            />
            {!cameraOn && (
              <div className="flex aspect-video items-center justify-center text-sm text-textDim">
                Camera is off
              </div>
            )}
          </div>

          <div className="overflow-hidden rounded-2xl border border-border bg-black">
            <div className="flex items-center justify-between border-b border-border/50 bg-panel px-3 py-1.5">
              <span className="text-[11px] font-bold uppercase tracking-wide text-accent">
                Enhanced {streaming ? '(live)' : ''}
              </span>
              {streaming && fps !== null && (
                <span className="rounded-full bg-accent/15 px-2 py-0.5 text-[11px] font-bold text-accent">
                  {fps.toFixed(1)} FPS · frame {frameCount}
                </span>
              )}
            </div>
            {displayImage ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={displayImage}
                alt="Enhanced"
                className="aspect-video w-full object-contain"
              />
            ) : (
              <div className="flex aspect-video items-center justify-center text-sm text-textDim">
                {cameraOn
                  ? 'Start live streaming or take a snapshot'
                  : 'Turn the camera on to begin'}
              </div>
            )}
          </div>
        </div>
        <canvas ref={canvasRef} className="hidden" />

        {displayResult && (
          <div className="rounded-2xl border border-border bg-panel p-5">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Brightness" value={displayResult.brightness.toFixed(3)} />
              <Stat
                label="Eff. confidence"
                value={displayResult.effective_confidence.toFixed(2)}
              />
              <Stat
                label="High risk"
                value={displayResult.high_risk ? 'Yes' : 'No'}
                danger={displayResult.high_risk}
              />
              <Stat
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
          </div>
        )}
      </section>
    </div>
  );
}

function Stat({
  label,
  value,
  danger,
}: {
  label: string;
  value: string;
  danger?: boolean;
}) {
  return (
    <div className="rounded-xl border border-border bg-card px-3 py-2.5 text-center">
      <div
        className={`text-lg font-extrabold ${danger ? 'text-danger' : 'text-accent'}`}
      >
        {value}
      </div>
      <div className="mt-0.5 text-[11px] text-textDim">{label}</div>
    </div>
  );
}
