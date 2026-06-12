// Keyboard + mouse (pointer-lock) input. Produces a normalized control
// vector that the physics step consumes. Mouse drives pitch & roll like a
// self-centring sidestick; keys handle throttle, rudder, and toggles.

export class Controls {
  constructor(canvas) {
    this.canvas = canvas;
    this.keys = new Set();
    this.captured = false;

    // Sidestick position, -1..1, eased toward the mouse-driven target.
    this.stickX = 0;
    this.stickY = 0;
    this._targetX = 0;
    this._targetY = 0;

    this.gear = true;
    this.toggles = { camera: 0, switchAircraft: false, reset: false };

    this._bind();
  }

  _bind() {
    addEventListener("keydown", (e) => {
      if (e.repeat) return;
      const k = e.key.toLowerCase();
      this.keys.add(k);
      if (k === "g") this.gear = !this.gear;
      if (k === "c") this.toggles.camera = (this.toggles.camera + 1) % 3;
      if (k === "tab") { e.preventDefault(); this.toggles.switchAircraft = true; }
      if (k === "r") this.toggles.reset = true;
    });
    addEventListener("keyup", (e) => this.keys.delete(e.key.toLowerCase()));

    this.canvas.addEventListener("click", () => {
      if (!this.captured) this.canvas.requestPointerLock();
    });
    document.addEventListener("pointerlockchange", () => {
      this.captured = document.pointerLockElement === this.canvas;
    });
    document.addEventListener("mousemove", (e) => {
      if (!this.captured) return;
      const sens = 0.0016;
      this._targetX = clamp(this._targetX + e.movementX * sens, -1, 1);
      this._targetY = clamp(this._targetY + e.movementY * sens, -1, 1);
    });
  }

  has(...k) { return k.some((x) => this.keys.has(x)); }

  // Sample the control state for this frame.
  sample(dt) {
    // Stick eases toward target and gently re-centres so it isn't twitchy.
    const ease = 1 - Math.exp(-12 * dt);
    this.stickX += (this._targetX - this.stickX) * ease;
    this.stickY += (this._targetY - this.stickY) * ease;
    this._targetX *= 1 - 0.9 * dt; // self-centring
    this._targetY *= 1 - 0.9 * dt;

    let throttleDelta = 0;
    if (this.has("w")) throttleDelta += 0.4;
    if (this.has("s")) throttleDelta -= 0.4;

    let rudder = 0;
    if (this.has("a")) rudder -= 1;
    if (this.has("d")) rudder += 1;

    return {
      elevator: -this.stickY, // pull back (mouse up) = nose up
      aileron: this.stickX,
      rudder,
      throttleDelta,
      afterburner: this.has("shift"),
      airbrake: this.has("b"),
    };
  }

  consumeToggles() {
    const t = { ...this.toggles };
    this.toggles.switchAircraft = false;
    this.toggles.reset = false;
    return t;
  }

  reset() { this.stickX = this.stickY = this._targetX = this._targetY = 0; }
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
