// Sovereign Ambient Web Audio Engine (zero dependencies)

class AmbientSoundEngine {
  private ctx: AudioContext | null = null
  private masterGain: GainNode | null = null
  private droneGain: GainNode | null = null
  private isMuted: boolean = true
  private isNight: boolean = false
  private timers: ReturnType<typeof setTimeout>[] = []

  public init() {
    if (this.ctx) return
    const AudioCtx =
      window.AudioContext ||
      (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext

    if (!AudioCtx) return
    const ctx = new AudioCtx()
    this.ctx = ctx

    const master = ctx.createGain()
    master.gain.value = 0
    master.connect(ctx.destination)
    this.masterGain = master

    // Shared low atmospheric drone
    const droneGain = ctx.createGain()
    droneGain.gain.value = 0.14
    droneGain.connect(master)
    this.droneGain = droneGain

    const filter = ctx.createBiquadFilter()
    filter.type = "lowpass"
    filter.frequency.value = 680
    filter.connect(droneGain)

    const droneA = ctx.createOscillator()
    droneA.type = "sine"
    droneA.frequency.value = 108

    const droneB = ctx.createOscillator()
    droneB.type = "sine"
    droneB.frequency.value = 108 * 1.008

    droneA.connect(filter)
    droneB.connect(filter)

    const lfo = ctx.createOscillator()
    lfo.frequency.value = 0.04
    const lfoGain = ctx.createGain()
    lfoGain.gain.value = 240
    lfo.connect(lfoGain)
    lfoGain.connect(filter.frequency)

    droneA.start()
    droneB.start()
    lfo.start()

    this.scheduleAtmosphere()
  }

  private scheduleAtmosphere() {
    if (!this.ctx || !this.masterGain) return
    const ctx = this.ctx

    const keyClick = (time: number) => {
      if (!this.ctx || this.isMuted) return
      const osc = ctx.createOscillator()
      osc.type = "triangle"
      osc.frequency.value = 260 + Math.random() * 150

      const g = ctx.createGain()
      g.gain.setValueAtTime(0.0001, time)
      g.gain.linearRampToValueAtTime(0.035, time + 0.01)
      g.gain.exponentialRampToValueAtTime(0.0001, time + 0.12)

      osc.connect(g)
      g.connect(this.masterGain!)
      osc.start(time)
      osc.stop(time + 0.15)
    }

    let t = 0
    const count = 2 + Math.floor(Math.random() * 4)
    for (let i = 0; i < count; i++) {
      this.timers.push(setTimeout(() => keyClick(ctx.currentTime), t))
      t += 140 + Math.random() * 180
    }
    this.timers.push(
      setTimeout(() => this.scheduleAtmosphere(), t + 1200 + Math.random() * 2200)
    )
  }

  public setMuted(muted: boolean) {
    this.isMuted = muted
    if (!this.ctx) {
      if (!muted) this.init()
      else return
    }
    if (this.ctx && this.ctx.state === "suspended" && !muted) {
      this.ctx.resume()
    }
    if (this.masterGain && this.ctx) {
      const now = this.ctx.currentTime
      this.masterGain.gain.cancelScheduledValues(now)
      this.masterGain.gain.linearRampToValueAtTime(muted ? 0 : 0.45, now + 0.5)
    }
  }

  public setNight(night: boolean) {
    this.isNight = night
    if (this.droneGain && this.ctx) {
      const now = this.ctx.currentTime
      this.droneGain.gain.cancelScheduledValues(now)
      this.droneGain.gain.linearRampToValueAtTime(night ? 0.22 : 0.12, now + 0.6)
    }
  }

  public dispose() {
    this.timers.forEach(clearTimeout)
    this.timers = []
    if (this.ctx) {
      this.ctx.close()
      this.ctx = null
    }
  }
}

export const ambientSound = new AmbientSoundEngine()
