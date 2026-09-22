'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DropZone from '@/components/DropZone';
import Toggle from '@/components/Toggle';
import SliderField from '@/components/SliderField';
import PageTransition from '@/components/PageTransition';
import {
  AppConfig,
  JobStatus,
  VideoOptions,
  cancelJob,
  fetchConfig,
  getJobStatus,
  jobVideoUrl,
  submitVideo,
} from '@/lib/api';

const DEFAULT_OPTS: VideoOptions = {
  adaptive: true,
  det_conf: 0.25,
  det_imgsz: 640,
  enable_detect: true,
  enable_pothole: false,
  enable_lanes: false,
  fast_mode: true,
};

export default function VideoPage() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [opts, setOpts] = useState<VideoOptions>(DEFAULT_OPTS);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetchConfig()
      .then(setConfig)
      .catch(() => setConfig(null));
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  function startPolling(jobId: string) {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await getJobStatus(jobId);
        setJob(s);
        if (s.status === 'done' || s.status === 'error' || s.status === 'cancelled') {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    }, 1500);
  }

  async function onSubmit() {
    if (!file) return;
    setSubmitting(true);
    setError(null);
    setJob(null);
    try {
      const { job_id } = await submitVideo(file, opts);
      setJob({
        id: job_id,
        status: 'queued',
        progress: 0,
        frames_done: 0,
        frames_total: 0,
        error: null,
        counts: {},
        high_risk: false,
        ready: false,
      });
      startPolling(job_id);
    } catch (e: any) {
      setError(e.message || 'Something went wrong');
    } finally {
      setSubmitting(false);
    }
  }

  async function onCancel() {
    if (!job) return;
    await cancelJob(job.id);
  }

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
            Video Enhancement
          </h2>
          <p className="mb-4 text-sm text-textDim">
            Upload a night-driving clip. Processed in the background —
            detection runs every 2nd frame by default for speed.
          </p>

          <DropZone
            accept="video/mp4,video/quicktime,video/x-msvideo"
            onFile={(f) => {
              setFile(f);
              setJob(null);
              setError(null);
            }}
            fileName={file?.name}
            hint="MP4, MOV, or AVI · capped at 900 frames for this demo API"
          />

          <div className="mt-5 border-t border-border pt-4">
            <div className="mb-2 text-xs font-bold uppercase tracking-wide text-textDim">
              Enhancement
            </div>
            <Toggle
              label="Adaptive day/night blending"
              checked={opts.adaptive}
              onChange={(v) => setOpts({ ...opts, adaptive: v })}
            />
            <Toggle
              label="Fast mode"
              hint="Detect objects every 2nd frame"
              checked={opts.fast_mode}
              onChange={(v) => setOpts({ ...opts, fast_mode: v })}
            />
          </div>

          <div className="mt-4 border-t border-border pt-4">
            <div className="mb-2 text-xs font-bold uppercase tracking-wide text-textDim">
              Detection
            </div>
            <Toggle
              label="Object detection + tracking"
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
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={onSubmit}
            disabled={!file || submitting || (job !== null && job.status === 'processing')}
            className="mt-5 w-full rounded-xl bg-accent-gradient px-4 py-2.5 text-sm font-bold text-base shadow-glow transition disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? 'Submitting…' : 'Enhance Video'}
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

        <motion.section
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
        >
          {!job && (
            <div className="glass flex aspect-video items-center justify-center rounded-2xl border-dashed text-textDim">
              Upload a video to get started
            </div>
          )}

          <AnimatePresence mode="wait">
            {job && (
              <motion.div
                key={job.id}
                initial={{ opacity: 0, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                className="glass rounded-2xl p-5"
              >
                <div className="mb-3 flex items-center justify-between">
                  <div className="text-sm font-bold text-textMain">
                    Job <span className="text-textDim">{job.id}</span>
                  </div>
                  <StatusBadge status={job.status} />
                </div>

                {(job.status === 'queued' || job.status === 'processing') && (
                  <>
                    <div className="h-2.5 w-full overflow-hidden rounded-full bg-card2">
                      <motion.div
                        className="h-full rounded-full bg-accent-gradient"
                        animate={{ width: `${job.progress}%` }}
                        transition={{ duration: 0.4, ease: 'easeOut' }}
                      />
                    </div>
                    <div className="mt-2 flex items-center justify-between text-xs text-textDim">
                      <span>
                        {job.frames_done} / {job.frames_total || '?'} frames ·{' '}
                        {job.progress}%
                      </span>
                      <button
                        onClick={onCancel}
                        className="rounded-md border border-danger/40 px-2 py-1 text-danger transition hover:bg-danger/10"
                      >
                        Cancel
                      </button>
                    </div>
                  </>
                )}

                {job.status === 'done' && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-2"
                  >
                    <video
                      src={jobVideoUrl(job.id)}
                      controls
                      className="w-full rounded-xl border border-border bg-black"
                    />
                    <div className="mt-3 flex flex-wrap items-center gap-3">
                      <motion.a
                        whileHover={{ scale: 1.03 }}
                        whileTap={{ scale: 0.97 }}
                        href={jobVideoUrl(job.id)}
                        download
                        className="rounded-xl bg-accent-gradient px-4 py-2 text-sm font-bold text-base shadow-glow"
                      >
                        Download enhanced video
                      </motion.a>
                      {job.high_risk && (
                        <span className="rounded-full border border-danger/30 bg-danger/10 px-3 py-1 text-xs font-semibold text-danger">
                          High-risk object detected in this clip
                        </span>
                      )}
                    </div>
                    {Object.keys(job.counts).length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {Object.entries(job.counts).map(([name, n]) => (
                          <span
                            key={name}
                            className="rounded-full border border-border bg-card2 px-2.5 py-1 text-xs font-medium text-textMain"
                          >
                            {name} × {n}
                          </span>
                        ))}
                      </div>
                    )}
                  </motion.div>
                )}

                {job.status === 'error' && (
                  <div className="mt-2 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
                    {job.error || 'Processing failed'}
                  </div>
                )}

                {job.status === 'cancelled' && (
                  <div className="mt-2 rounded-lg border border-border bg-card2 px-3 py-2 text-xs text-textDim">
                    Job cancelled.
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

function StatusBadge({ status }: { status: JobStatus['status'] }) {
  const styles: Record<JobStatus['status'], string> = {
    queued: 'border-border bg-card2 text-textDim',
    processing: 'border-accent2/40 bg-accent2/10 text-accent2',
    done: 'border-ok/30 bg-ok/10 text-ok',
    error: 'border-danger/30 bg-danger/10 text-danger',
    cancelled: 'border-border bg-card2 text-textDim',
  };
  return (
    <span
      className={`rounded-full border px-2.5 py-1 text-xs font-bold uppercase tracking-wide ${
        status === 'processing' ? 'animate-pulse' : ''
      } ${styles[status]}`}
    >
      {status}
    </span>
  );
}
