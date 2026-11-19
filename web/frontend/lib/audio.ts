/**
 * Web Audio API HUD Chime Generator for UNLET-ADAS
 * Zero external audio assets required. Synthesizes authentic automotive
 * instrument-cluster chimes directly via the Web Audio API.
 */
class AdasAudioHud {
  private ctx: AudioContext | null = null;
  private muted: boolean = false;
  private lastAlertTime: number = 0;

  constructor() {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('unlet_adas_audio_muted');
      if (stored !== null) {
        this.muted = stored === 'true';
      }
    }
  }

  private initContext() {
    if (!this.ctx && typeof window !== 'undefined') {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  public isMuted(): boolean {
    return this.muted;
  }

  public setMuted(muted: boolean) {
    this.muted = muted;
    if (typeof window !== 'undefined') {
      localStorage.setItem('unlet_adas_audio_muted', String(muted));
    }
  }

  public toggleMute(): boolean {
    this.setMuted(!this.muted);
    return this.muted;
  }

  /**
   * High Collision Risk Alert: Dual-frequency urgent sweep (880Hz -> 587Hz)
   */
  public playHighRiskAlarm() {
    if (this.muted) return;
    const now = Date.now();
    if (now - this.lastAlertTime < 800) return; // Debounce 800ms
    this.lastAlertTime = now;

    this.initContext();
    if (!this.ctx) return;

    try {
      const t = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(880, t);
      osc.frequency.exponentialRampToValueAtTime(440, t + 0.22);

      gain.gain.setValueAtTime(0.25, t);
      gain.gain.exponentialRampToValueAtTime(0.01, t + 0.22);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start(t);
      osc.stop(t + 0.22);
    } catch {
      // Audio permitted only on user interaction
    }
  }

  /**
   * Caution / Lane Departure Chime: Single chime tone (587.3Hz)
   */
  public playCautionChime() {
    if (this.muted) return;
    const now = Date.now();
    if (now - this.lastAlertTime < 1200) return;
    this.lastAlertTime = now;

    this.initContext();
    if (!this.ctx) return;

    try {
      const t = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(587.33, t);

      gain.gain.setValueAtTime(0.18, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.3);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start(t);
      osc.stop(t + 0.3);
    } catch {
      // Ignored
    }
  }
}

export const audioHud = new AdasAudioHud();
