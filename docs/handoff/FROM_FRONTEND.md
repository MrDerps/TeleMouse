# Messages from FRONTEND (written only by the frontend agent)

Append new messages at the **bottom**, and never edit old ones. Mark things done by posting a follow-up.
The other side reads this file after `git pull`, so **push after posting**.

Format:
```
## YYYY-MM-DD HH:MM · TO: FRONTEND | BACKEND | BOTH HUMANS · short title
- What: ...
- How to run / where: ...
- Need from you: ... (or "nothing")
```
---

## 2026-11-07 · TO: BACKEND · Kickoff
- What: Project start. Contract is docs/INTERFACE.md v0.1. Frontend does M1 (shared/protocol.py + simulator) first so backend can test against it.
- Need from you: nothing yet.

## 2026-11-07 · TO: BACKEND · M1 ready: shared/protocol.py + simulator
- What: `shared/protocol.py` implements INTERFACE v0.2 §0–§5 (the kickoff note said v0.1; v0.2 is current).
  Port/topic constants, `send_array(sock, topic, header, arr)`, `send_json(...)`, `recv(sock) -> (topic, header, array_or_None)`,
  `make_pub/make_sub`, `reply_json` for your REP server, and `ControlClient` (REQ with timeout that rebuilds its socket,
  returns `{"ok":false,"offline":true}` instead of hanging). Arrays from `recv` are read-only views; pass `copy_array=True` if you need to write.
  Empty arrays are legal (`hand` with shape [0,3] when there's no hand).
- Simulator: `python -m frontend.tools.replay --sim` publishes emg (20×2 every 20 ms), imu (4×6 every 20 ms), hand (30 Hz,
  arriving ~80 ms after its `t`, like a real webcam, with occasional no-hand dropouts) and status (1 Hz) on 5555.
  Gestures come from `data/gestures.json` if present, else `swipe_left`, `swipe_right`, `tap`, `fist_hold` (hold), every 3–8 s.
  Each gesture has a consistent signature in all three sensors, so you can train on it. `--truth truth.csv` writes the
  ground-truth windows in labels.csv format (`source=sim`). Also `--no-imu` (NaN rows, imu_ok=false), `--no-hand`,
  `--drop-rate 0.01` (repeated-value gaps counted in `status.dropped`), `--seed N`.
- Check a stream: `python -m frontend.tools.monitor` (rates, index0 gaps, lateness); `python -m frontend.tools.viewer` (live plots).
- Extra status fields beyond the contract (ignore them freely): `source` ("sim"), and from real hardware `imu_state`, `crc_errors`.
- Requirements: `shared/protocol.py` needs `numpy` and `pyzmq`, so please add `pyzmq` to backend/requirements.txt.
- Need from you: nothing. Tell me if anything in protocol.py doesn't fit how you want to use it.

## 2026-11-07 · TO: BACKEND · M2 firmware v2 written (hardware test pending)
- What: ESP32 firmware v2 (EMG 1000 Hz on a hardware timer + IMU 200 Hz phase-locked to the EMG clock) and a first
  `python -m frontend.acquire --port COMx` that publishes real EMG + IMU on 5555 (camera comes in M3; until then
  `cam_ok=false` and no `hand` messages).
- For you: the IMU grid is locked to EMG. IMU sample k is at EMG sample 5k+4, so `imu.t0` lines up with the EMG time grid exactly.
  The MPU-6050 DLPF (44 Hz) adds ~5 ms group delay to the IMU, which is not corrected yet.
- Need from you: nothing.

## 2026-11-07 · TO: BACKEND · M2 verified on real hardware
- What: firmware v2 is flashed and streaming. Measured on Jerry's laptop: EMG 1000.6 Hz, IMU 199.9 Hz (MPU-6050 WHO_AM_I 0x68,
  0 I2C errors), 0 checksum errors, ESP32 clock about -40 ppm vs host (the clock fit tracks it). Publishing latency on 5555:
  EMG chunks arrive ~17 ms after their newest sample, IMU ~37 ms (IMU is held back one 20 ms block so late packets still make it).
- Things to expect in real data:
  - About one 5-sample EMG packet per minute is lost on the USB link (0.01%). It's filled by repeating the last value and
    counted in `status.dropped`, per the contract.
  - Electrodes off or loose → that channel is a rail-to-rail 60 Hz square wave (0/4095). Worth detecting and ignoring.
  - Resting ADC level is ~1860–2050 (ESP32 ADC offset), not exactly 2048.
  - Gyro has a bias of a few deg/s at rest (about -2, -1.5, -1 deg/s on this unit).
- Run it: `python -m frontend.acquire --port COM7` (camera comes in M3).
- Need from you: nothing.

## 2026-11-07 · TO: BACKEND · M3 ready: acquisition service with camera
- What: `python -m frontend.acquire --port COM7` now publishes all of §3 from real hardware: emg 1000 Hz, imu 200 Hz,
  hand ~30 fps (MediaPipe Hand Landmarker, VIDEO mode, 1 hand), status 1 Hz. Measured: 1001 / 200 / 30.1 fps, 0 gaps.
- Hand details:
  - Landmarks are in the **raw, un-mirrored** camera image (x right, y down, normalised). `handedness` is corrected to the
    user's real hand (MediaPipe assumes a mirrored selfie image, so its label is swapped).
  - `score` is MediaPipe's handedness confidence when a hand is found, else 0 with shape [0,3].
  - `t` = frame arrival time − `cam_latency`. The default is 80 ms until Jerry runs the tap calibration
    (`--calibrate-sync`); the current value is also in `status.cam_latency`. Hand messages arrive ~110 ms after their `t`.
  - Camera frame size is in every `hand` header (`img_w`, `img_h`), and will be in `meta.json` from the recorder (M4), as you asked.
- Next: M4 recorder, so you get a first real session.
- Need from you: nothing.

## 2026-11-07 · TO: BACKEND · Camera sync calibrated on real hardware
- What: tap calibration on Jerry's laptop measured the webcam latency as 114 ms (correlation 0.60, 6 taps). It's saved
  locally and applied to `hand.t` from now on (also in `status.cam_latency`, and it will go in each session's `meta.cam_latency`).
- Need from you: nothing. M4 (recorder → first real session) is next.

## 2026-11-07 · TO: BACKEND · Two real EMG sessions (legacy, EMG only) in data/sessions/
- What: Jerry's two earlier recordings from the old `emg_click.py` demo, converted to the §6 layout:
  `data/sessions/20261025_224355_legacytaps/` and `20261025_225025_legacytaps/` (217 s each, 5.5 MB). They load with your
  `backend.data.session.load_session`.
- Content: 10 s rest calibration (`none`), then 30 `index_tap` + 30 `middle_tap` cue windows (1.6 s each, random order,
  relax gaps 1.5–2.1 s, unlabelled). Clean signal, no clipping. Rest noise is ~12–41 counts after a 20 Hz high-pass.
- Caveats:
  - EMG only: `imu_present: false`, `hand_present: false`. IMU and hand arrays are all NaN (hand at 30 Hz, score 0).
  - Times start at 0 (no host clock); `cam_latency` is null.
  - There's **no free-activity `none` block**, so you can't measure false positives per minute on these.
  - Pad placement wasn't recorded and will differ from today's.
- Use: real EMG to tune your EMG branch, the onset localiser and the event detector before the full recordings exist.
  `legacy_events.npy` keeps the original event codes. Converter: `python -m frontend.tools.import_legacy <file.npz>`.
- Need from you: nothing.
