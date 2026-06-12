// Aircraft catalogue and start locations.
// Each aircraft is a point-mass flight model; numbers are deliberately
// plausible (SI units) rather than exact manufacturer data.

const SAMPLE_MODEL =
  "https://cdn.jsdelivr.net/gh/CesiumGS/cesium@1.118/Apps/SampleData/models/CesiumAir/Cesium_Air.glb";

export const AIRCRAFT = [
  {
    id: "trainer",
    name: "C-172 Trainer",
    model: SAMPLE_MODEL,
    scale: 1.0,
    mass: 1100, // kg
    wingArea: 16.2, // m^2
    cd0: 0.027, // parasitic drag
    inducedK: 0.045, // induced drag factor
    clMax: 1.6, // stall lift coefficient
    maxThrust: 5500, // N (prop equivalent)
    afterburner: 1.0,
    maxPitchRate: 0.6, // rad/s at full control authority
    maxRollRate: 1.6,
    maxYawRate: 0.5,
    cornerSpeed: 45, // m/s where controls reach full authority
    vne: 90, // never-exceed speed (m/s)
  },
  {
    id: "airliner",
    name: "B737 Airliner",
    model: SAMPLE_MODEL,
    scale: 3.4,
    mass: 62000,
    wingArea: 124,
    cd0: 0.022,
    inducedK: 0.043,
    clMax: 1.4,
    maxThrust: 240000,
    afterburner: 1.0,
    maxPitchRate: 0.22,
    maxRollRate: 0.55,
    maxYawRate: 0.2,
    cornerSpeed: 140,
    vne: 290,
  },
  {
    id: "fighter",
    name: "F/A-18 Fighter",
    model: SAMPLE_MODEL,
    scale: 1.5,
    mass: 16000,
    wingArea: 38,
    cd0: 0.021,
    inducedK: 0.12,
    clMax: 1.8,
    maxThrust: 160000,
    afterburner: 1.6, // Shift = afterburner
    maxPitchRate: 1.3,
    maxRollRate: 3.6,
    maxYawRate: 0.8,
    cornerSpeed: 160,
    vne: 600,
  },
  {
    id: "aerobat",
    name: "Extra 330 Aerobat",
    model: SAMPLE_MODEL,
    scale: 0.8,
    mass: 950,
    wingArea: 10.7,
    cd0: 0.028,
    inducedK: 0.05,
    clMax: 2.0,
    maxThrust: 7800,
    afterburner: 1.0,
    maxPitchRate: 2.6,
    maxRollRate: 6.0,
    maxYawRate: 1.4,
    cornerSpeed: 55,
    vne: 120,
  },
];

export const LOCATIONS = [
  { id: "ksfo", name: "San Francisco — KSFO", lon: -122.375, lat: 37.619, alt: 1200, heading: 280 },
  { id: "knyc", name: "New York — Manhattan", lon: -74.013, lat: 40.705, alt: 900, heading: 30 },
  { id: "lswn", name: "Swiss Alps — Matterhorn", lon: 7.658, lat: 45.976, alt: 4200, heading: 0 },
  { id: "ksan", name: "San Diego — Coronado", lon: -117.17, lat: 32.69, alt: 800, heading: 270 },
  { id: "rjtt", name: "Tokyo — Haneda", lon: 139.78, lat: 35.55, alt: 1000, heading: 320 },
  { id: "egll", name: "London — Thames", lon: -0.12, lat: 51.5, alt: 900, heading: 90 },
  { id: "grand", name: "Grand Canyon", lon: -112.11, lat: 36.1, alt: 2800, heading: 0 },
];

export function aircraftById(id) {
  return AIRCRAFT.find((a) => a.id === id) || AIRCRAFT[0];
}
