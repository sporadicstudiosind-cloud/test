# EarthFlight ✈

A browser-based flight simulator that flies over **real-world photorealistic
terrain** streamed from **Google's Photorealistic 3D Tiles** (rendered with
[CesiumJS](https://cesium.com/platform/cesiumjs/)), with a genuine point-mass
flight model — thrust, drag, lift, gravity, coordinated turns, and real stalls.

> **Reality check:** "Google Earth terrain in a flight sim" is done the
> supported way here — via Google Maps Platform's **Map Tiles API**
> (Photorealistic 3D Tiles), *not* by scraping the Google Earth app. You bring
> your own API key; the sim never ships one. Without a key it falls back to
> OpenStreetMap imagery so it still runs.

## Run it

It's a static site — no build step.

```bash
# from the repo root, any static server works:
python3 -m http.server 8080
#  or:  npx serve .
```

Then open <http://localhost:8080>. (A server is required — ES modules and the
tile requests won't work from a `file://` URL.)

## Getting the photoreal terrain (Google key)

1. In the [Google Cloud Console](https://console.cloud.google.com/), create a
   project and enable the **Map Tiles API**.
2. Create an **API key** (restrict it to your domain / `localhost` for safety).
3. Paste it into the **API key** field on the start screen. It's stored only in
   your browser's `localStorage`.

Billing note: the Map Tiles API is a paid Google service with a monthly free
tier. Usage is on your own Google account.

## Controls

| Input            | Action                                    |
| ---------------- | ----------------------------------------- |
| **Mouse**        | Pitch & roll (click the canvas to capture)|
| **A / D**        | Rudder (yaw)                              |
| **W / S**        | Throttle up / down                        |
| **Shift**        | Afterburner / full power                  |
| **B**            | Airbrake                                  |
| **G**            | Landing gear                              |
| **C**            | Cycle camera (chase / cockpit / external) |
| **Tab**          | Switch aircraft                           |
| **R**            | Reset attitude (wings level)              |
| **Esc**          | Release mouse / open menu                 |

## Aircraft

Four hand-tuned flight models — a Cessna-style trainer, a 737-class airliner,
an F/A-18-style fighter (with afterburner), and an aerobatic monoplane. Each
has its own mass, wing area, drag, thrust, stall behaviour, and control rates.
Three AI aircraft circle the start area as traffic.

## How the physics works

See [`src/physics.js`](src/physics.js). The aircraft is modelled as a 3-DOF
point mass with attitude kinematics:

- **Airspeed** integrates thrust − drag − the gravity component along the
  flight path.
- **Drag** = parasitic (`Cd0`) + induced (`k · Cl²`), plus airbrake.
- **Lift** must supply the load factor demanded by your bank angle and elevator
  pull. When the required lift coefficient exceeds `Clmax`, the wing **stalls**:
  the nose drops and controls go soft.
- **Turns** are coordinated: bank angle curves the flight path at
  `g · tan(φ) / V`.
- **Air density** falls with altitude, so the aircraft (and your controls) feel
  different up high.

## Project layout

```
index.html        UI shell + HUD, loads CesiumJS from CDN
styles.css        HUD and menu styling
src/aircraft.js   Aircraft catalogue + start locations
src/physics.js    Flight dynamics (pure, framework-free)
src/controls.js   Keyboard + pointer-lock mouse input
src/main.js       Cesium world, render loop, camera, AI traffic
```

## Limitations & honest caveats

- It's an arcade-leaning *3-DOF* model, not full 6-DOF rigid-body aero — no
  sideslip dynamics, ground-effect, or detailed engine modelling.
- The aircraft mesh is Cesium's sample `Cesium_Air` model reused/retinted; the
  realism comes from the world and the flight model, not bespoke airframes.
- Photoreal quality and coverage are whatever Google's 3D Tiles provide for a
  given location.
