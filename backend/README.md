# Backend (ML): datasets, model, training, live inference

Setup (repo root, Python 3.12):
```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

| Command | What |
|---|---|
| `python -m backend.tools.synth --n 2` | synthetic sessions (INTERFACE §6 layout) → `backend/_synth/` |
| `python -m backend.tools.fake_backend` | fake 5556/5557 backend for frontend development |
| `python -m backend.tools.fake_client` | smoke-test client for 5556/5557 |
| `python -m backend.data.plot_trial [--session ID]` | example trials (raw, features, activity, localised label) → `backend/_out/trials.png` |
| `python -m backend.baseline [--synth] [--sessions ...] [--gestures ...] [--name N]` | LDA baseline + report → `data/models/N/` |
| `python -m backend.train [--synth] [--sessions ...] [--gestures ...] [--name N] [--warm-start OLD]` | TCN training (~2 min CPU) → `data/models/N/{model.pt, model.json, report.png, curve.png}` |
| `python -m backend.serve [--load NAME --start] [--synth]` | **live server**: SUB 5555 → PUB 5556 (probs/gesture/status), REP 5557 control |
| `python -m backend.tools.record_stream --truth T.csv` | (test tool) record the 5555 sim stream as a session |
| `python -m backend.tools.live_eval --truth T.csv` | (test tool) score live gesture events and latency against sim truth |
| `python -m pytest backend/tests -q` | unit tests |

All hyperparameters are in `backend/config.yaml`, and all paths are relative to the repo root.

## Layout
- `common/`: config loading, and ZeroMQ via `shared/protocol.py`.
- `data/session.py`: loads a session folder.
- `data/features.py`: **one causal streaming pipeline** used by both training and live inference, on a 50 Hz grid:
  - EMG: 20 Hz high-pass + 60 Hz notch, then rest-MAD normalisation, then per-frame RMS/MAV/WL/ZC/SSC × 2 channels (log-compressed).
  - IMU: 20 Hz anti-alias, then acc, gravity-removed acc, gyro and their magnitudes, averaged per frame.
  - Hand: wrist-relative landmarks / palm size, wrist position and velocity, palm size. Held ≤ 100 ms, otherwise zeroed and masked.
  - Each frame also carries a presence mask per modality.
- `data/labels.py`: finds the motion inside each cue window from an activity score (robust z of EMG envelope, |gyro| and hand motion). Ignores ~100 ms edges, and falls back to the window centre (logged) if nothing is found.
- `data/dataset.py`: builds the tensors, plus the train/val split. With ≥2 sessions the newest one is val; otherwise a stratified 20% of trials plus the tail of the free-activity block.
- `eval/events.py`: probabilities → events. EMA, hysteresis with k_on, refractory period; instant fires at the peak, hold sends start/end. **The live server uses the same code.**
- `eval/metrics.py`: event precision/recall, false positives per minute on free activity, median latency, frame confusion, and per-gesture threshold tuning (keeps FP ≤ target/min).
- `baseline.py`: causal 400 ms window stats (mean, std, last−first) → shrinkage LDA. A sanity floor for the TCN.

- `model/tcn.py`: the model.
  - Each modality ([features + mask]) goes through 2 causal dilated conv blocks (32 ch, d=1,2), then is multiplied by its mask.
  - The branches and masks are concatenated and fed to a residual causal TCN (64 ch, k=3, d=1..16), then a per-frame head.
  - About 117k params, receptive field 69 frames (1.38 s), about 2 ms per live forward pass on the last 96 frames.
- `model/augment.py`: augmentation applied to training windows.
  - Time warp ±10%, EMG amplitude ×0.7–1.3 per channel, and ±15° 3D rotation of IMU vectors and landmarks.
  - Gaussian noise, plus modality dropout (EMG 10%, IMU 15%, hand 30%) and random hand-lost gaps.
  - Training windows are taken only from training frames, and half of them are centred on gesture spans.
- `train.py`: training loop.
  - Class-weighted cross-entropy (weights ∝ 1/√freq, `none` dominates), AdamW with a one-cycle schedule, and the best checkpoint by validation loss.
  - Thresholds are tuned on validation, with the same report format as the baseline.

- `serve/engine.py`: the live engine.
  - Raw 5555 messages go through the same `StreamFeaturizer`, then the TCN runs on the last 96 frames, then the same `EventDetector` turns probabilities into events.
  - A test checks that it matches offline output exactly.
- `serve/server.py`: the control server.
  - Training runs in a child process, so a crash there can't kill the server.
  - `calibrate_rest` takes a robust std per 250 ms block and averages the quieter half, so it survives a stray movement.
  - Status adds `lag_ms` (frame age when published) and `emg_warning` (a channel stuck rail-to-rail, i.e. an electrode is off).
- Latency design: hand features for frame T use landmarks up to T − 80 ms (`hand.lag_s`), in training and live alike. So frames never wait for the ~110 ms-late webcam; they're published about 33 ms after their sensor time.

## Choosing sensors (M6)
Any subset of `emg`, `imu`, `hand` can be used:
- **Training:** `python -m backend.train --sensors imu emg`, or `"sensors": [...]` in the `train` command. The other sensors are masked out, exactly as if unplugged.
- **Live:** `{"cmd":"start","sensors":["imu","emg"]}` runs a model with fewer sensors than it was trained with. Modality dropout makes this work.
- **Accuracy per combination:** after training, each model is evaluated with every combination of its sensors, with thresholds re-tuned per combination. Results go to the console table, `sensors.png` and `model.json` → `by_sensors`, and `list_models` returns them. The live server uses the matching thresholds and reports that accuracy in `start` replies and in `status.accuracy`.
- **Automatic masking:** EMG is masked automatically per frame when a channel sits at an ADC rail (electrode off), and for a whole session when `meta.emg_valid` is false.

## Real-data lessons (M6)
- **Users anticipate the cue,** so the motion is searched for from 0.5 s before the cue (`labels.pre_cue_s`).
- **Users repeat gestures or keep going after the cue.** Active frames in the unlabelled relax gaps are ignored (never taught as `none`). Within a cue window, only the strongest burst is the gesture; other strong bursts are ignored, which also covers preparatory phases such as opening the hand before a pinch. Repeats inside a gesture's own cue window are neutral in evaluation.
- **Camera timing:** hand messages arrive about 150 ms late, so `hand.lag_s` = 0.16. For labelling only, the hand activity signal is shifted back by that lag so it lines up with the IMU.
- **Regularisation:** the TCN overfits small real datasets. Defaults are now dropout 0.3, weight decay 0.01 and 400 steps (about 45 s). The checkpoint is chosen by validation event-F1; with fewer than 6 validation gestures, the final weights are kept.

## Why a multi-branch causal TCN (M4)
- **Causal and streaming:** the TCN is causal, so live inference just runs the same convolution over a ~1.5 s history.
- **Fast and stable:** it trains in about 1–3 minutes on a CPU and is stable on small datasets (~20 reps per class). LSTMs train slower. Transformers overfit at this data size.
- **Per-modality encoders:** each sensor keeps its own characteristics, and modality dropout plus presence masks let a missing sensor (hand out of view, IMU unplugged) drop out cleanly instead of breaking the fused representation.

## Synthetic data caveat
Synthetic gestures are deliberately easy, and they exist only for plumbing. Near-perfect baseline scores on them mean nothing about real data.
