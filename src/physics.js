// 3-DOF point-mass flight dynamics with attitude kinematics.
//
// State is expressed in geographic terms (lon/lat/height) plus an airspeed
// scalar and an attitude (heading / pitch / bank). Forces are real:
// thrust, parasitic + induced drag, lift vs. weight, and a genuine stall
// when the wing can no longer generate the lift the load factor demands.

const G = 9.80665;
const RHO0 = 1.225; // sea-level air density (kg/m^3)
const SCALE_HEIGHT = 8500; // m

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const deg = (r) => (r * 180) / Math.PI;
const rad = (d) => (d * Math.PI) / 180;

export function createState(loc, spec) {
  return {
    lon: rad(loc.lon),
    lat: rad(loc.lat),
    height: loc.alt, // metres MSL
    v: spec.cornerSpeed * 1.4, // start at a comfortable cruise
    heading: rad(loc.heading), // clockwise from north
    pitch: 0,
    bank: 0,
    throttle: 0.65,
    // telemetry (filled by step)
    loadFactor: 1,
    stalled: false,
    verticalSpeed: 0,
    ias: 0,
    aoa: 0,
    crashed: false,
  };
}

function airDensity(height) {
  return RHO0 * Math.exp(-Math.max(height, 0) / SCALE_HEIGHT);
}

// input: { elevator, aileron, rudder, throttleDelta, afterburner, airbrake } each normalized
export function step(s, spec, input, dt, terrainHeight) {
  if (s.crashed) return s;

  // --- throttle ---
  s.throttle = clamp(s.throttle + input.throttleDelta * dt, 0, 1);
  const burnerMult = input.afterburner ? spec.afterburner : 1;
  const thrust = s.throttle * spec.maxThrust * burnerMult;

  const rho = airDensity(s.height);
  const v = Math.max(s.v, 0.1);
  const qS = 0.5 * rho * v * v * spec.wingArea; // dynamic pressure * area

  // --- control authority scales with airspeed (mushy controls when slow) ---
  const authority = clamp(v / spec.cornerSpeed, 0.15, 1.15);

  // Load factor commanded by the pilot pulling on the elevator, plus the
  // structural load required to hold the current bank angle in a turn.
  const turnLoad = 1 / Math.max(Math.cos(s.bank), 0.2);
  const commandedG = turnLoad + Math.max(input.elevator, 0) * 3.5;

  // Lift coefficient the wing must produce for that load factor.
  const weight = spec.mass * G;
  const clRequired = (commandedG * weight) / Math.max(qS, 1);

  // --- stall check ---
  s.stalled = clRequired > spec.clMax;
  const clActual = Math.min(clRequired, spec.clMax);
  s.loadFactor = (clActual * qS) / weight;
  s.aoa = (clActual / spec.clMax) * 15; // approx degrees, for display

  // --- attitude integration ---
  let pitchRate = input.elevator * spec.maxPitchRate * authority;
  if (s.stalled) {
    // Wing dropped: nose falls regardless of stick, controls go soft.
    pitchRate -= 0.8 * dt * 30;
    pitchRate = Math.min(pitchRate, -0.2);
  }
  s.pitch = clamp(s.pitch + pitchRate * dt, rad(-89), rad(89));

  const rollRate = input.aileron * spec.maxRollRate * authority;
  s.bank = clamp(s.bank + rollRate * dt, rad(-160), rad(160));

  // Coordinated turn: bank curves the flight path. Rudder adds a yaw nudge.
  const turnRate = (G * Math.tan(s.bank)) / v + input.rudder * spec.maxYawRate * 0.3;
  s.heading = (s.heading + turnRate * dt) % (2 * Math.PI);
  if (s.heading < 0) s.heading += 2 * Math.PI;

  // --- forces along the flight path ---
  const cd = spec.cd0 + spec.inducedK * clActual * clActual + (input.airbrake ? 0.08 : 0);
  const drag = cd * qS;
  const gammaForce = -G * Math.sin(s.pitch); // gravity component along path
  const dv = (thrust - drag) / spec.mass + gammaForce;
  s.v = Math.max(s.v + dv * dt, 0);

  // Overspeed bleed (poor-man's compressibility / structural limit).
  if (s.v > spec.vne) s.v -= (s.v - spec.vne) * 0.5 * dt;

  // --- translate position ---
  const groundSpeed = s.v * Math.cos(s.pitch);
  s.verticalSpeed = s.v * Math.sin(s.pitch);
  s.height += s.verticalSpeed * dt;

  // Move across the ellipsoid (flat-earth approximation over one step).
  const R = 6371000 + s.height;
  const dNorth = (groundSpeed * Math.cos(s.heading) * dt) / R;
  const dEast = (groundSpeed * Math.sin(s.heading) * dt) / (R * Math.cos(s.lat));
  s.lat += dNorth;
  s.lon += dEast;

  s.ias = s.v * Math.sqrt(rho / RHO0);

  // --- ground / crash ---
  const floor = (terrainHeight ?? 0) + 1.5;
  if (s.height <= floor) {
    s.height = floor;
    const sinkFpm = -s.verticalSpeed * 196.85;
    const tooFast = s.v > spec.cornerSpeed * 1.6;
    const tooBanked = Math.abs(deg(s.bank)) > 10;
    const tooNoseDown = deg(s.pitch) < -8;
    if (sinkFpm > 800 || tooFast || tooBanked || tooNoseDown) {
      s.crashed = true;
    } else {
      // Survivable contact: rolling along the ground.
      s.pitch = Math.max(s.pitch, 0);
      s.verticalSpeed = Math.max(s.verticalSpeed, 0);
      s.v *= 1 - 0.4 * dt; // rolling friction
    }
  }
  return s;
}

export const telemetry = (s) => ({
  iasKt: s.ias * 1.94384,
  altFt: s.height * 3.28084,
  vsFpm: s.verticalSpeed * 196.85,
  hdg: ((deg(s.heading) % 360) + 360) % 360,
  g: s.loadFactor,
});

export { deg, rad, clamp };
