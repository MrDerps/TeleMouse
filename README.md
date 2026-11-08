# Axion: a wearable gesture mouse

Axion lets you control your computer with hand gestures. Electrodes on your forearm read muscle signals (EMG), and a
motion sensor (IMU) on the back of your hand reads movement. You teach it your own gestures, give each one an action
(click, scroll, a keyboard shortcut…), and a neural network trained on your laptop recognises them live.

- **No camera needed.** EMG + IMU work with your arm in any position, in any lighting.
- **Your gestures, your actions.** Record a gesture 15 times (about 2 minutes) and it's ready to use.
- **Add or remove gestures any time.** Every recording is kept, so the model just retrains on all of them in a minute or two.
- **Built-in safety.** Actions are off until you turn them on, and **F9** stops everything instantly from any window.

> ⚠️ **Safety:** while electrodes are on a person, power the ESP32 from a **battery, a power bank or an unplugged laptop.
> Never from a computer plugged into the wall.**

---

## How it works

```
 forearm EMG (2 ch, 1 kHz) ─┐
                            ├─ ESP32 ── USB serial ──► acquisition ──► features (50 Hz) ──► neural network ──► gesture
 hand IMU (MPU-6050, 200 Hz)┘                              ▲                                                   │
                                     webcam (optional) ────┘                                                   ▼
                                                                         Axion app (PyQt6) ◄── shows it ── action (click / key / scroll)
```

1. **Hardware:** a hand-built 2-channel EMG amplifier (gain ≈ 1100, 19–480 Hz) and an MPU-6050 IMU feed an ESP32,
   which streams the data to the laptop.
2. **Acquisition** (`frontend/acquire`) reads the serial stream (and the optional webcam with MediaPipe) and
   publishes it over ZeroMQ.
3. **Gesture engine** (`backend/serve`) turns the signals into features on a 50 Hz grid and runs a small causal
   temporal convolutional network (PyTorch). Gravity is removed from the IMU data, so arm direction doesn't matter.
4. **The Axion app** (`frontend/app`) is where you teach gestures, train, watch the live signals and AI output, and
   turn on "Control my computer". Actions are sent with pynput.

---

## Hardware

| Part | Notes |
|---|---|
| ESP32 dev module | EMG ch1 → **GPIO32**, ch2 → **GPIO33** (12-bit ADC, sampled at 1000 Hz) |
| 2-channel EMG front end | per channel: MCP6N11 instrumentation amp (×11) → 19 Hz high-pass → MCP6004 stage (×101, 480 Hz low-pass) → anti-alias filter |
| MPU-6050 IMU | on the back of the hand, I²C: SDA **GPIO21**, SCL **GPIO22**, address 0x68, 200 Hz |
| Electrodes | Ag/AgCl gel pads: one pair per channel on the forearm, plus a reference pad on the elbow bone |
| Webcam (optional) | used for the "pinch to move" pointer mode |

**Electrode placement tip:** put one channel on the **top** of the forearm (muscles that open the hand) and one on the
**bottom** (muscles that close it). The bigger the difference between the two channels, the easier it is to tell
gestures apart. Full circuit details: [docs/HARDWARE.md](docs/HARDWARE.md).

---

## Setup (Windows)

Requires **Python 3.12**. MediaPipe doesn't support newer versions yet.

```powershell
git clone https://github.com/MrDerps/TeleMouse.git
cd TeleMouse
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r frontend/requirements.txt -r backend/requirements.txt
```

### Flash the ESP32

1. Install the Arduino IDE and the **ESP32 core by Espressif**.
2. Open `firmware/emg_imu_stream/emg_imu_stream.ino`, pick your board ("ESP32 Dev Module") and COM port, and upload.
3. Check the stream:
   ```powershell
   python firmware/tools/check_stream.py --list           # find the COM port
   python firmware/tools/check_stream.py --port COM7      # 10 s health check
   ```

The serial format (921600 baud, framed packets with CRC) is described in [firmware/PROTOCOL.md](firmware/PROTOCOL.md).

---

## Running it

Open two terminals with the venv activated.

**1. Start the gesture engine**
```powershell
python -m backend.serve
```

**2. Start the app**
```powershell
python -m frontend.app
```

The camera's hand model downloads automatically the first time you use it.

No hardware? Run `python -m frontend.app --sim` to use simulated sensors.

### Using the app

1. **Dashboard → Connect sensors**: the first time, Settings opens so you can pick the COM port (and camera). Remember the battery rule above.
2. **Calibrate**: relax your hand for 3 seconds so the EMG has a fresh baseline.
3. **Gestures → Teach a new gesture**: name it, choose *instant* (fires once) or *hold* (active while held), pick an
   action, then follow the on-screen cues: 15 repetitions, then some normal activity.
4. The model **retrains automatically**. Watch progress on the **Train** page.
5. **Dashboard → Control my computer**: your gestures now perform their actions. **F9** stops everything.

Useful extras:
- **Switching a gesture off** keeps its recordings and teaches the model to *ignore* that motion.
- **Settings** (gear icon): sensors, EMG influence, IMU feature mode, pointer options, EMG channel swap, and data reset.
- **Data mode** (top right) shows live signals, the camera, model output, and on the Train page a network diagram,
  learning curve and confusion matrix.

---

## Repository layout

```
firmware/        ESP32 sketch (EMG + IMU streamer), serial protocol, stream checker
frontend/
  acquire/       serial + camera acquisition, time sync, EMG channel-swap detection
  record/        guided recording sessions
  actions/       action executor (pynput), pointer control, F9 kill switch
  app/           the Axion desktop app (PyQt6)
  tools/         simulator/replay, signal viewer, monitor, legacy import
backend/
  data/          session loading, features, label localisation
  model/         TCN model, augmentation, save/load
  serve/         live gesture engine (ZeroMQ server, background training)
  eval/          metrics, event detection, reports
  tools/         synthetic sessions, fake backend/client, offline evaluation
  tests/         pytest suite
shared/          message protocol shared by frontend and backend
docs/            INTERFACE.md (frontend ↔ backend contract), HARDWARE.md
legacy/          the original EMG-only click demo (reference)
data/            recordings (sessions/), gestures.json, settings, trained models (models/, not committed)
```

**Ports (ZeroMQ, localhost):** 5555 sensors → engine · 5556 engine → app (probabilities, gestures, status) · 5557 control
(train, load, start, stop…). Full message spec: [docs/INTERFACE.md](docs/INTERFACE.md).

---

## Tests and tools

```powershell
python -m pytest backend -q                      # backend test suite
python -m frontend.tools.replay --sim            # publish simulated sensor data
python -m frontend.tools.viewer                  # watch the raw sensor stream
python -m backend.serve --load <model> --start   # start the engine with a model already running
```

---

## Team and workflow

Built by **Jerry** (hardware, firmware, app) and **William** (model and training) for a hackathon. Each of us worked
with a Claude Code agent on our own laptop, and the two agents coordinated only through this repo: they followed the
contract in `docs/INTERFACE.md` and left each other notes in `docs/handoff/`. The rules for agents working in this
repo are in [CLAUDE.md](CLAUDE.md).
