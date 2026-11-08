# Interface contract: frontend ↔ backend   (version 0.5: single IMU, sensor subsets, sensor weights, val sessions, warm start, delete_model)

The frontend and backend run as **separate processes** and talk over **ZeroMQ on localhost**,
plus shared **files in `data/`**. The serial packet format is internal to the frontend, so the backend never sees it.
Change this document only by proposing it in your `docs/handoff/FROM_<YOU>.md` file, with both humans' approval, and bump the version.

## 0. Ports (all on 127.0.0.1)
| Port | Socket | Direction | Purpose |
|---|---|---|---|
| 5555 | PUB (frontend binds) / SUB | frontend → backend | live synchronized sensor data |
| 5556 | PUB (backend binds) / SUB | backend → frontend | live predictions and gesture events |
| 5557 | REP (backend binds) / REQ | frontend → backend | control commands (train, load, start, stop, status) |

The Python helpers live in `shared/protocol.py`. The frontend writes the first version, both sides import it,
and changes to it follow the same process as changes to this file.

## 1. Message encoding
Every message is a ZeroMQ **multipart** message:
- `[topic: bytes, header: UTF-8 JSON]` for JSON-only messages
- `[topic: bytes, header: UTF-8 JSON, payload: raw bytes]` when an array is attached.

When there is a payload, the header has `"dtype"` (numpy dtype string, little-endian) and `"shape"` (list).
Decode it with `np.frombuffer(payload, dtype).reshape(shape)`.

## 2. Time
- All timestamps are **seconds, float64, on one host clock**: `time.perf_counter()` in the frontend process.
  The backend never reads its own clock for data. It uses the timestamps in the messages.
- **EMG and IMU** are timed by the ESP32 sample counter and mapped to host time by the frontend, using an online
  linear fit of counter against host receive time (take the minimum-latency envelope, because USB adds jitter).
  So consecutive EMG samples are exactly 1 ms apart in host time.
- **Camera** frames are stamped at capture. The frontend subtracts a calibrated camera latency (default 0.08 s).
  The frontend provides a "sync calibration" in which the user does sharp hand taps, and the latency is estimated by
  cross-correlating the IMU acceleration spike with landmark motion.
- Guarantee: EMG and IMU are aligned to within about 1 ms, and the camera to within about one frame (±33 ms) after calibration.

## 3. Sensor stream (port 5555, frontend → backend)
| Topic | Rate | Header fields | Payload |
|---|---|---|---|
| `b"emg"` | chunks every ~20 ms | `t0` (time of first sample), `fs`=1000, `index0` (global sample index of the first row), `dtype`="<u2", `shape`=[n,2] | raw ADC counts, columns [ch1, ch2], 0–4095, rest ≈ 2048. **Unfiltered**: the backend does all filtering |
| `b"imu"` | chunks every ~20 ms | `t0`, `fs`=200, `index0`, `dtype`="<f4", `shape`=[n,6] | per row `[ax, ay, az, gx, gy, gz]` from the single back-of-hand IMU, accel in **g**, gyro in **deg/s**, sensor axes as mounted (see HARDWARE.md). If the IMU is missing or failed, the values are NaN |
| `b"hand"` | every camera frame (~30 Hz) | `t` (latency-corrected), `score` (0–1, 0 = no hand), `handedness` ("Left"/"Right"/null), `img_w`, `img_h`, `dtype`="<f4", `shape`=[21,3] or [0,3] when there's no hand | MediaPipe landmarks, x and y normalised to [0,1] of the image, z = MediaPipe relative depth |
| `b"status"` | 1 Hz | `emg_ok`, `imu_ok`:bool, `cam_ok`, `emg_rate`, `imu_rate`, `cam_fps`, `dropped`, `battery_warning_shown` | none |

`hand` carries the **gesture hand** (the one wearing the IMU). Default: one hand is tracked and published.
In the optional two-hand mode (off by default), the frontend decides which visible hand is which (motion vs IMU, plus MediaPipe's label); the other hand is published on the
frontend-internal topic `b"pointer_hand"` (same header/payload as `hand`, used only for cursor control; the
backend can ignore it, and it deliberately does not start with `hand`, because ZeroMQ topics are prefix matches).

Dropped ESP32 samples: the frontend fills each gap by repeating the last value, keeps the time grid regular, and
counts the gap in `status.dropped`.

## 4. Backend outputs (port 5556, backend → frontend)
| Topic | Rate | Header (JSON only) |
|---|---|---|
| `b"probs"` | 50 Hz | `t` (stream time of the newest frame used), `classes`: [...], `p`: [...] (sums to 1, includes `"none"`) |
| `b"gesture"` | when one happens | `t`, `id` (gesture id), `conf`, `kind`: `"instant"` (fire once), or `"start"`/`"end"` (for hold gestures) |
| `b"status"` | 1 Hz, and on change | `state`: `"idle"`/`"training"`/`"running"`/`"error"`, `model`: name or null, `progress`: 0–1 while training, `message`. Since v0.3 also (optional) `sensors`, `accuracy` while running |

The latency target is under 150 ms from end of motion to the `gesture` event (ideally under 100 ms).

## 5. Control (port 5557, frontend REQ → backend REP, JSON in and out)
| Request | Reply |
|---|---|
| `{"cmd":"status"}` | `{"ok":true, "state":..., "model":..., "classes":[...]}` |
| `{"cmd":"train", "gestures":["swipe_left",...], "sessions":"all" or ["<session_id>",...], "name":"model_name", "sensors":["emg","imu","hand"] (v0.3, optional, default all), "sensor_weights":{"emg":0.2,"imu":1.0,"hand":1.0} (v0.4, optional, 0..1 per sensor; default EMG 0.2, others 1.0), "val_sessions":["<session_id>",...] (v0.4, optional: hold out whole sessions; default = last 20% of each gesture's reps in every session), "warm_start":"<model name>" (v0.5, optional: start from that model's weights and train train.warm_steps instead of train.steps)}` | `{"ok":true}` immediately. Training runs in the background, and progress is published on 5556 `status` |
| `{"cmd":"load", "name":"model_name"}` | `{"ok":true, "classes":[...]}` |
| `{"cmd":"start"}` / `{"cmd":"stop"}` | `{"ok":true}`: start or stop live inference on the 5555 stream |
| `{"cmd":"start", "sensors":["emg","imu"]}` (v0.3, optional field) | `{"ok":true, "sensors":[...], "accuracy":{"precision","recall","fp_per_min","latency_ms"}}`: run the loaded model with only these sensors |
| `{"cmd":"list_models"}` | `{"ok":true, "models":[{"name":..., "classes":[...], "created":..., "val_metrics":{...}, "sensors":[...], "by_sensors":{"emg+imu+hand":{"precision","recall","fp_per_min","latency_ms","per_gesture" (v0.5)}, "imu":{...}, ...}}]}` (`sensors`, `by_sensors` since v0.3) |
| `{"cmd":"delete_model", "name":"model_name"}` (v0.5) | `{"ok":true}`; refuses the loaded model |
| `{"cmd":"calibrate_rest", "seconds":3}` | `{"ok":true}`: the backend recomputes rest-normalisation stats from the live stream (the user relaxes) |

Errors are returned as `{"ok":false, "error":"message"}`. Never let an exception kill the server.

## 6. Files in `data/` (git-ignored, shared)
### `data/gestures.json` (written by the frontend UI, read by the backend)
```json
{"version": 1,
 "gestures": [
   {"id": "swipe_left", "name": "Swipe left", "kind": "instant",
    "action": {"type": "scroll", "dx": 0, "dy": -5}, "created": "2026-11-07T14:00:00"},
   {"id": "pinch_hold", "name": "Pinch and hold", "kind": "hold",
    "action": {"type": "drag", "button": "left"}, "created": "..."}
 ]}
```
- `id`: snake_case, unique, and never reused for a different motion.
- `kind`: `"instant"` (one event per occurrence) or `"hold"` (start and end events, active while held).
- `action` is used **only by the frontend**. The backend ignores it.

### `data/sessions/<session_id>/` (written by the frontend recorder, read by the backend)
`session_id` = `YYYYmmdd_HHMMSS_<tag>`.
| File | Content |
|---|---|
| `meta.json` | `session_id`, `created`, `emg_valid` (bool, v0.3; false = no pads / unusable EMG), `img_w`, `img_h` (v0.3), `fs_emg`=1000, `fs_imu`=200, `cam_latency`, `imu_present`:bool, `electrode_notes`, `imu_notes`, `gestures_snapshot` (copy of gestures.json at record time), `rest_periods`: [[t_start,t_end],...], `protocol_version` |
| `emg.npy`, `emg_t.npy` | uint16 (N,2) raw counts; float64 (N,) host times |
| `imu.npy`, `imu_t.npy` | float32 (M,6) as in §3; float64 (M,) |
| `hand.npy`, `hand_t.npy`, `hand_score.npy` | float32 (K,21,3) with **NaN rows when there's no hand**; float64 (K,); float32 (K,) |
| `labels.csv` | header `t_start,t_end,label,source`. One row per **cue window** (`source=cue`), with `label` = gesture id or `none`. The cue window is generous: the actual motion happens somewhere inside it, and the backend localises it |
| `video.mp4` (optional) | low-res camera video for debugging and relabelling |

### Recording protocol (implemented by the frontend recorder UI)
1. **Rest calibration**: 10 s relaxed, which goes into `meta.rest_periods` and also gets a `none` label.
2. **Cued gesture trials**: for each chosen gesture, N repetitions (default 20) in random order. Each trial is a
   relax gap of 1.5–2.5 s (random) followed by a 2.0 s cue window ("do SWIPE LEFT now"). For `hold` gestures, the cue says
   "hold until the bar ends", and the window is longer (3 s).
3. **"None" free-activity block** (critical for avoiding false triggers): 2–3 min labelled `none`, during which the user
   types, uses a real mouse, moves the hand around, picks things up and gestures while talking.
   Ask for this in every session.

### `data/models/<name>/` (written and read by the backend only)
The backend's own format (weights + `model.json` with classes, feature/normalisation config and validation metrics).
The frontend only uses `list_models` and `load`.

## 7. Simulators (so each side can work alone)
- **Frontend provides** `frontend/tools/replay.py`, which publishes a recorded session on 5555 in real time, and a
  `--sim` mode that synthesises plausible EMG, IMU and hand data with occasional scripted gestures.
- **Backend provides** `backend/tools/fake_backend.py`, which answers control requests and publishes plausible `probs`,
  `gesture` and `status` messages, plus a synthetic-session generator for its own tests.
