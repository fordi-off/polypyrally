# PolyPyRally

A low-poly 3D rally game in **Python** - flat-shaded facets, a warm low sun with soft cascaded shadows, tinted
height fog, a stylised sky and a filmic tone-map, so it is a pleasure for the eye. Real vehicle physics: a 6-DOF
rigid body on four spring-damper wheels, a combined-slip tyre model, an engine torque curve, slipping clutch,
6-speed gearbox and an AWD driveline with limited-slip couplings.

```
pip install -r requirements.txt
python -m polypyrally                 # --fullscreen  --vsync  --msaa 8  --shadow 4096  --fps  --seed 12
```

## Stack
| piece | what |
|---|---|
| window, input, audio, HUD | **pygame-ce** (the HUD is drawn with pygame and composited over the 3D scene) |
| rendering | **moderngl** (OpenGL 3.3 core): cascaded shadow maps, MSAA HDR target, bloom, ACES tone-map |
| terrain, meshes, particles | **numpy** (vectorised mesh building; the heightfield the physics drives on *is* the drawn mesh) |
| physics | plain Python at a fixed 240 Hz sub-step (~0.5 ms per frame) |

## Controls
`W/S` gas & brake (hold `S` when stopped to reverse) · `A/D` steer · `Space` handbrake · `E/Q` shift up/down (it
shifts itself too) · `C` camera (chase / far / hood / bumper) · `R` back onto the road · `F3` fps · `F12`
screenshot · `Esc` pause menu.

## Development helpers
```
python -m polypyrally --bench --size 1920x1080         # CPU cost per frame (offscreen)
python -m polypyrally --shot out.png --s 900 --cam 30,14,-25,60   # render one frame without a window
python -m polypyrally --drive out.png --secs 30        # an AI drives, saves a frame with the HUD
python tests/test_rally.py
```

## Code map (`polypyrally/`)
`stage.py` road spline, elevation, heightfield, height queries · `terrain_mesh.py` chunked LOD flat-shaded terrain ·
`scenery.py` instanced trees / rocks · `props.py` arches, spectators · `vehicle.py` the physics · `car_model.py` the
low-poly car · `gl_render.py` + `shaders.py` the renderer · `world.py` streaming · `game.py` state machine ·
`camera.py` · `particles.py` · `hud.py` · `audio.py` (synthesised) · `ai.py` a pace-note driver.
