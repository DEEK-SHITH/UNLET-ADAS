export function getApiUrl(): string {
  const envUrl = (process.env.NEXT_PUBLIC_API_URL || '').replace(/\/$/, '');
  if (typeof window !== 'undefined') {
    const host = window.location.hostname;
    const isRemoteVisitor = host !== 'localhost' && host !== '127.0.0.1';
    if (isRemoteVisitor) {
      // If NEXT_PUBLIC_API_URL points to a real public cloud backend, use it;
      // if it's empty or still pointing to localhost from .env.local, use same-origin ('')
      if (
        envUrl &&
        !envUrl.includes('localhost') &&
        !envUrl.includes('127.0.0.1')
      ) {
        return envUrl;
      }
      return '';
    }
  }
  return envUrl || 'http://localhost:8000';
}

export const API_URL = getApiUrl();

export interface Detection {
  name: string;
  conf: number;
  box: [number, number, number, number];
  risk?: 'LOW' | 'MEDIUM' | 'HIGH';
}

export interface ImageResult {
  brightness: number;
  enhanced_brightness?: number;
  effective_confidence: number;
  detections: Detection[];
  potholes: Detection[];
  signs: Detection[];
  counts: Record<string, number>;
  high_risk: boolean;
  lanes_found: boolean;
  depth_risk_used: boolean;
  processing_ms: number;
  original_image: string;
  enhanced_image: string;
  comparison_image?: string;
}

export interface AppConfig {
  has_detect: boolean;
  has_pothole: boolean;
  has_signs: boolean;
  has_depth: boolean;
  classes: Record<string, string>;
}

export interface JobStatus {
  id: string;
  status: 'queued' | 'processing' | 'done' | 'error' | 'cancelled';
  progress: number;
  frames_done: number;
  frames_total: number;
  error: string | null;
  counts: Record<string, number>;
  high_risk: boolean;
  ready: boolean;
  has_comparison?: boolean;
}

export interface ImageOptions {
  adaptive: boolean;
  det_conf: number;
  det_imgsz: number;
  enable_detect: boolean;
  enable_pothole: boolean;
  enable_signs: boolean;
  enable_lanes: boolean;
  use_depth_risk: boolean;
}

export interface VideoOptions {
  adaptive: boolean;
  det_conf: number;
  det_imgsz: number;
  enable_detect: boolean;
  enable_pothole: boolean;
  enable_signs: boolean;
  enable_lanes: boolean;
  use_depth_risk: boolean;
  fast_mode: boolean;
  quality_preset?: 'turbo' | 'high' | 'ultra';
}

export interface LiveOptions {
  adaptive: boolean;
  det_conf: number;
  det_imgsz: number;
  enable_detect: boolean;
  enable_lanes: boolean;
  interleaved: boolean;
  quality_mode: 'turbo' | 'balanced' | 'quality';
}

export interface LiveResult {
  brightness: number;
  enhanced_brightness: number;
  detections: Detection[];
  counts: Record<string, number>;
  high_risk: boolean;
  processing_ms: number;
  fps: number;
  enhanced_image: string;
}

export async function fetchConfig(): Promise<AppConfig> {
  const res = await fetch(`${getApiUrl()}/api/config`);
  if (!res.ok) throw new Error('Failed to load backend config');
  return res.json();
}

export async function enhanceImage(
  file: File,
  opts: ImageOptions
): Promise<ImageResult> {
  const form = new FormData();
  form.append('file', file);
  Object.entries(opts).forEach(([k, v]) => form.append(k, String(v)));
  const res = await fetch(`${getApiUrl()}/api/enhance/image`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Enhance failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function enhanceLiveFrame(
  blob: Blob,
  opts: LiveOptions,
  signal?: AbortSignal
): Promise<LiveResult> {
  const form = new FormData();
  form.append('file', blob, 'live-frame.jpg');
  Object.entries(opts).forEach(([k, v]) => form.append(k, String(v)));
  const res = await fetch(`${getApiUrl()}/api/enhance/live-frame`, {
    method: 'POST',
    body: form,
    signal,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Live frame failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function submitVideo(
  file: File,
  opts: VideoOptions
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append('file', file);
  Object.entries(opts).forEach(([k, v]) => form.append(k, String(v)));
  const res = await fetch(`${getApiUrl()}/api/enhance/video`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Video submit failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${getApiUrl()}/api/jobs/${jobId}`, {
    cache: 'no-store',
  });
  if (!res.ok) throw new Error('Failed to fetch job status');
  return res.json();
}

export async function cancelJob(jobId: string): Promise<void> {
  await fetch(`${getApiUrl()}/api/jobs/${jobId}/cancel`, { method: 'POST' });
}

export function jobVideoUrl(jobId: string, mode: 'enhanced' | 'comparison' = 'enhanced'): string {
  return `${getApiUrl()}/api/jobs/${jobId}/video?mode=${mode}`;
}
