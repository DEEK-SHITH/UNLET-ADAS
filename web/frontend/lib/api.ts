export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Detection {
  name: string;
  conf: number;
  box: [number, number, number, number];
  risk?: 'LOW' | 'MEDIUM' | 'HIGH';
}

export interface ImageResult {
  brightness: number;
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
  enable_lanes: boolean;
  fast_mode: boolean;
}

export async function fetchConfig(): Promise<AppConfig> {
  const res = await fetch(`${API_URL}/api/config`);
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
  const res = await fetch(`${API_URL}/api/enhance/image`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Enhance failed (${res.status}): ${detail}`);
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
  const res = await fetch(`${API_URL}/api/enhance/video`, {
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
  const res = await fetch(`${API_URL}/api/jobs/${jobId}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch job status');
  return res.json();
}

export async function cancelJob(jobId: string): Promise<void> {
  await fetch(`${API_URL}/api/jobs/${jobId}/cancel`, { method: 'POST' });
}

export function jobVideoUrl(jobId: string): string {
  return `${API_URL}/api/jobs/${jobId}/video`;
}
