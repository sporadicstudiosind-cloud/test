import { AIRCRAFT, LOCATIONS, aircraftById } from "./aircraft.js";
import { createState, step, telemetry, deg, rad, clamp } from "./physics.js";
import { Controls } from "./controls.js";

const Cesium = window.Cesium;

// Cesium_Air.glb is authored facing +X (East); our heading is compass-from-north.
const MODEL_HEADING_OFFSET = -Cesium.Math.PI_OVER_TWO;
const PHYS_DT = 1 / 120; // fixed physics timestep (s)

let viewer, scene, controls;
let player; // { state, spec, entity }
let traffic = []; // AI aircraft
let running = false;
let physAccumulator = 0;
let lastTime = performance.now();

// ----------------------------------------------------------------------------
// Menu wiring
// ----------------------------------------------------------------------------
function initMenu() {
  const ac = document.getElementById("aircraftSelect");
  AIRCRAFT.forEach((a) => ac.add(new Option(a.name, a.id)));
  const loc = document.getElementById("locationSelect");
  LOCATIONS.forEach((l) => loc.add(new Option(l.name, l.id)));

  const savedKey = localStorage.getItem("gmpKey");
  if (savedKey) document.getElementById("apiKey").value = savedKey;

  document.getElementById("startBtn").addEventListener("click", () => {
    const key = document.getElementById("apiKey").value.trim();
    if (key) localStorage.setItem("gmpKey", key);
    const spec = aircraftById(ac.value);
    const location = LOCATIONS.find((l) => l.id === loc.value) || LOCATIONS[0];
    start(key, spec, location);
  });
}

// ----------------------------------------------------------------------------
// World setup
// ----------------------------------------------------------------------------
async function buildWorld(googleKey) {
  viewer = new Cesium.Viewer("cesiumContainer", {
    baseLayer: false,
    baseLayerPicker: false,
    geocoder: false,
    homeButton: false,
    sceneModePicker: false,
    navigationHelpButton: false,
    animation: false,
    timeline: false,
    fullscreenButton: false,
    infoBox: false,
    selectionIndicator: false,
    terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    contextOptions: { webgl: { powerPreference: "high-performance" } },
  });
  scene = viewer.scene;
  scene.globe.depthTestAgainstTerrain = true;
  scene.skyAtmosphere.show = true;
  scene.fog.enabled = true;
  viewer.clock.shouldAnimate = false;

  let usingGoogle = false;
  if (googleKey) {
    try {
      Cesium.GoogleMaps.defaultApiKey = googleKey;
      const tileset = await Cesium.createGooglePhotorealistic3DTileset();
      scene.primitives.add(tileset);
      scene.globe.show = false; // tiles supply their own terrain + imagery
      usingGoogle = true;
    } catch (err) {
      console.error("Google 3D Tiles failed:", err);
      alert(
        "Could not load Google Photorealistic 3D Tiles with that key.\n" +
          "Check that the Map Tiles API is enabled. Falling back to OpenStreetMap terrain."
      );
    }
  }

  if (!usingGoogle) {
    viewer.imageryLayers.addImageryProvider(
      new Cesium.OpenStreetMapImageryProvider({ url: "https://tile.openstreetmap.org/" })
    );
  }
  return usingGoogle;
}

function makeEntity(spec, color) {
  return viewer.entities.add({
    model: {
      uri: spec.model,
      scale: spec.scale,
      minimumPixelSize: 48,
      color: color ? Cesium.Color.fromCssColorString(color) : undefined,
      colorBlendMode: color ? Cesium.ColorBlendMode.MIX : undefined,
      colorBlendAmount: 0.4,
    },
    position: new Cesium.Cartesian3(),
  });
}

function orient(entity, s) {
  const pos = Cesium.Cartesian3.fromRadians(s.lon, s.lat, s.height);
  const hpr = new Cesium.HeadingPitchRoll(
    s.heading + MODEL_HEADING_OFFSET,
    s.pitch,
    s.bank
  );
  entity.position = pos;
  entity.orientation = Cesium.Transforms.headingPitchRollQuaternion(pos, hpr);
  return pos;
}

// ----------------------------------------------------------------------------
// Start a flight
// ----------------------------------------------------------------------------
async function start(googleKey, spec, location) {
  document.getElementById("menu").classList.add("hidden");
  document.getElementById("loading").classList.remove("hidden");

  await buildWorld(googleKey);

  // Player
  player = { spec, state: createState(location, spec), entity: makeEntity(spec) };
  document.getElementById("hud-aircraft").textContent = spec.name.toUpperCase();

  // AI traffic circling the area at staggered altitudes.
  const aiColors = ["#ff7043", "#42a5f5", "#ffd54f"];
  traffic = aiColors.map((c, i) => {
    const aspec = AIRCRAFT[(i + 1) % AIRCRAFT.length];
    const loc = { ...location, alt: location.alt + 400 + i * 350, heading: i * 120 };
    return {
      spec: aspec,
      state: createState(loc, aspec),
      entity: makeEntity(aspec, c),
      ap: { targetAlt: loc.alt, targetBank: rad(18), targetV: aspec.cornerSpeed * 1.5 },
    };
  });

  controls = new Controls(viewer.canvas);
  viewer.canvas.setAttribute("tabindex", "0");
  viewer.canvas.focus();

  document.getElementById("loading").classList.add("hidden");
  document.getElementById("hud").classList.remove("hidden");

  running = true;
  lastTime = performance.now();
  scene.preUpdate.addEventListener(frame);
}

function getTerrainHeight(cart) {
  if (scene.sampleHeightSupported) {
    const exclude = [player.entity, ...traffic.map((t) => t.entity)];
    const h = scene.sampleHeight(cart, exclude);
    if (h != null) return h;
  }
  const g = scene.globe.show ? scene.globe.getHeight(cart) : null;
  return g ?? 0;
}

// ----------------------------------------------------------------------------
// AI autopilot: hold altitude, bank into a steady turn, hold speed.
// ----------------------------------------------------------------------------
function aiInput(t) {
  const s = t.state,
    ap = t.ap;
  const elevator = clamp((ap.targetAlt - s.height) * 0.0006 - s.verticalSpeed * 0.02, -0.5, 0.5);
  const aileron = clamp((ap.targetBank - s.bank) * 2.5, -1, 1);
  const throttleDelta = clamp((ap.targetV - s.v) * 0.4, -1, 1);
  return { elevator, aileron, rudder: 0, throttleDelta, afterburner: false, airbrake: false };
}

// ----------------------------------------------------------------------------
// Main loop — fixed-step physics, then render & camera.
// ----------------------------------------------------------------------------
function frame() {
  if (!running) return;
  const now = performance.now();
  let dt = (now - lastTime) / 1000;
  lastTime = now;
  dt = Math.min(dt, 0.1); // clamp huge frame gaps
  physAccumulator += dt;

  const tgl = controls.consumeToggles();
  if (tgl.switchAircraft) cycleAircraft();
  if (tgl.reset) {
    player.state.pitch = 0;
    player.state.bank = 0;
    controls.reset();
  }

  // Fixed timestep integration.
  while (physAccumulator >= PHYS_DT) {
    const input = controls.sample(PHYS_DT);
    const cartP = Cesium.Cartographic.fromRadians(
      player.state.lon,
      player.state.lat,
      player.state.height
    );
    step(player.state, player.spec, input, PHYS_DT, getTerrainHeight(cartP));

    for (const t of traffic) {
      const cart = Cesium.Cartographic.fromRadians(t.state.lon, t.state.lat, t.state.height);
      step(t.state, t.spec, aiInput(t), PHYS_DT, getTerrainHeight(cart));
    }
    physAccumulator -= PHYS_DT;
  }

  const pos = orient(player.entity, player.state);
  for (const t of traffic) orient(t.entity, t.state);

  updateCamera(pos, player.state, controls.toggles.camera);
  updateHud();
}

// ----------------------------------------------------------------------------
// Camera modes: 0 chase, 1 cockpit, 2 wide external
// ----------------------------------------------------------------------------
const _enu = new Cesium.Matrix4();
const _col = new Cesium.Cartesian3();
function enuAxes(pos) {
  Cesium.Transforms.eastNorthUpToFixedFrame(pos, undefined, _enu);
  const east = Cesium.Matrix4.getColumn(_enu, 0, new Cesium.Cartesian3());
  const north = Cesium.Matrix4.getColumn(_enu, 1, new Cesium.Cartesian3());
  const up = Cesium.Matrix4.getColumn(_enu, 2, new Cesium.Cartesian3());
  return { east, north, up };
}

function updateCamera(pos, s, mode) {
  const { east, north, up } = enuAxes(pos);
  // Forward unit vector (ENU -> ECEF) from heading & pitch.
  const ce = Math.sin(s.heading) * Math.cos(s.pitch);
  const cn = Math.cos(s.heading) * Math.cos(s.pitch);
  const cu = Math.sin(s.pitch);
  const fwd = new Cesium.Cartesian3();
  Cesium.Cartesian3.add(
    Cesium.Cartesian3.multiplyByScalar(east, ce, new Cesium.Cartesian3()),
    Cesium.Cartesian3.multiplyByScalar(north, cn, _col),
    fwd
  );
  Cesium.Cartesian3.add(fwd, Cesium.Cartesian3.multiplyByScalar(up, cu, _col), fwd);
  Cesium.Cartesian3.normalize(fwd, fwd);

  if (mode === 1) {
    // Cockpit: sit at the nose, look forward, bank with the aircraft.
    viewer.camera.setView({
      destination: Cesium.Cartesian3.add(
        pos,
        Cesium.Cartesian3.multiplyByScalar(up, 1.5, new Cesium.Cartesian3()),
        new Cesium.Cartesian3()
      ),
      orientation: { heading: s.heading, pitch: s.pitch, roll: -s.bank },
    });
    return;
  }

  // Chase (0) and wide (2): pull back along -forward, lift up.
  const dist = mode === 2 ? player.spec.scale * 38 + 60 : player.spec.scale * 16 + 22;
  const lift = mode === 2 ? dist * 0.45 : dist * 0.3;
  const dest = new Cesium.Cartesian3();
  Cesium.Cartesian3.add(
    pos,
    Cesium.Cartesian3.multiplyByScalar(fwd, -dist, new Cesium.Cartesian3()),
    dest
  );
  Cesium.Cartesian3.add(dest, Cesium.Cartesian3.multiplyByScalar(up, lift, _col), dest);
  viewer.camera.setView({
    destination: dest,
    orientation: { heading: s.heading, pitch: s.pitch - rad(8), roll: 0 },
  });
}

// ----------------------------------------------------------------------------
function cycleAircraft() {
  const idx = AIRCRAFT.findIndex((a) => a.id === player.spec.id);
  const next = AIRCRAFT[(idx + 1) % AIRCRAFT.length];
  viewer.entities.remove(player.entity);
  player.spec = next;
  player.entity = makeEntity(next);
  // keep position & motion, just swap airframe
  document.getElementById("hud-aircraft").textContent = next.name.toUpperCase();
}

// ----------------------------------------------------------------------------
function updateHud() {
  const s = player.state;
  const t = telemetry(s);
  const set = (id, v) => (document.getElementById(id).textContent = v);
  set("hud-ias", Math.round(t.iasKt));
  set("hud-alt", Math.round(t.altFt).toLocaleString());
  set("hud-vs", Math.round(t.vsFpm / 10) * 10);
  set("hud-hdg", String(Math.round(t.hdg)).padStart(3, "0"));
  set("hud-thr", Math.round(s.throttle * 100));
  set("hud-g", t.g.toFixed(1));

  const stall = document.getElementById("stall-warn");
  stall.classList.toggle("hidden", !s.stalled);

  if (s.crashed) {
    running = false;
    showCrash();
  }
}

function showCrash() {
  const menu = document.getElementById("menu");
  menu.querySelector(".menu-card").innerHTML =
    '<h1>Crashed.</h1><p class="tag">Bad day at the office. Try a gentler approach next time.</p>' +
    '<button id="startBtn" onclick="location.reload()">Restart</button>';
  menu.classList.remove("hidden");
  document.getElementById("hud").classList.add("hidden");
  if (document.pointerLockElement) document.exitPointerLock();
}

initMenu();
