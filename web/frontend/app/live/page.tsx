'use client';

import { useEffect, useRef, useState } from 'react';
import Toggle from '@/components/Toggle';
import SliderField from '@/components/SliderField';
import CompareView from '@/components/CompareView';
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
  const [result, setResult] = useState<ImageResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    fetchConfig()
      .then(setConfig)
      .catch(() => setConfig(null));
    return () => stopCamera();
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
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraOn(false);
  }

  async function captureAndEnhance() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !cameraOn) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob: Blob | null = await new Promise((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', 0.92)
    );
    if (!blob) return;
    const file = new File([blob], 'live-snapshot.jpg', { type: 'image/jpeg' });

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
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[340px_1fr]">
      <aside className="h-fit rounded-2xl border border-border bg-panel p-5">
        <h2 className="mb-1 text-lg font-bold">Live Camera</h2>
        <p className="mb-4 text-sm text-textDim">
          Uses your browser&apos;s camera. Take a snapshot and run it
          through the same enhance + detect pipeline — not a continuous
          live stream, since full-quality enhancement + detection is too
          slow per-frame on CPU for that.
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
          onClick={captureAndEnhance}
          disabled={!cameraOn || loading}
          className="mt-5 w-full rounded-xl bg-accent-gradient px-4 py-2.5 text-sm font-bold text-base transition disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? 'Enhancing…' : '📸 Take Photo & Enhance'}
        </button>

        {error && (
          <div className="mt-3 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
            {error}
          </div>
        )}
      </aside>

      <section>
        <div className="mb-4 overflow-hidden rounded-2xl border border-border bg-black">
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video
            ref={videoRef}
            className={`aspect-video w-full object-contain ${cameraOn ? '' : 'hidden'}`}
            playsInline
            muted
          />
          {!cameraOn && (
            <div className="flex aspect-video items-center justify-center text-textDim">
              Camera is off. Turn it on to see a preview here.
            </div>
          )}
        </div>
        <canvas ref={canvasRef} className="hidden" />

        {loading && (
          <div className="flex aspect-video animate-pulse items-center justify-center rounded-2xl border border-border bg-panel text-textDim">
            Running enhancement + detection…
          </div>
        )}

        {result && !loading && (
          <div className="rounded-2xl border border-border bg-panel p-5">
            <CompareView
              before={result.original_image}
              after={result.enhanced_image}
            />

            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Brightness" value={result.brightness.toFixed(3)} />
              <Stat
                label="Eff. confidence"
                value={result.effective_confidence.toFixed(2)}
              />
              <Stat
                label="High risk"
                value={result.high_risk ? 'Yes' : 'No'}
                danger={result.high_risk}
              />
              <Stat
                label="Processing"
                value={`${(result.processing_ms / 1000).toFixed(1)}s`}
              />
            </div>

            <DetectionList title="Objects" items={result.detections} />
            <DetectionList title="Potholes" items={result.potholes} />
            <DetectionList title="Signs" items={result.signs} />
            {result.lanes_found && (
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
