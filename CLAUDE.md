# EMG Mouse: shared project context

Every Claude Code session in this repo reads this file automatically, so keep it short and true.
Detailed specs live in `docs/`.

## What we're building
This is a gesture-controlled "mouse" for a hackathon. The user wears:
- a **2-channel sEMG** front end on the forearm (DIY analog board → ESP32),
- **one IMU on the back of the hand** (MPU-6050 → same ESP32),
- and a **webcam** watches the hand (MediaPipe hand landmarks on the laptop).

The user can **define any motion** (e.g. "swipe left") and **assign it an action** (e.g. scroll).
They record a few examples, and a neural network is **trained locally** on the 3 fused sensors to
detect when that motion happens, live. Pointer movement is continuous control (IMU/camera), and the
NN handles discrete/held gestures.

## Team: two people, two computers, two Claude Code agents
| Role | Who runs it | Owns (only edits these) | Brief |
|---|---|---|---|
| **FRONTEND** | Jerry, on his laptop (it has the hardware) | `firmware/`, `frontend/`, `shared/`, `docs/handoff/FROM_FRONTEND.md` | ESP32 firmware, serial + camera acquisition, time sync, recording tool, UI, action executor |
| **BACKEND** | William, on their own laptop | `backend/`, `docs/handoff/FROM_BACKEND.md` | Dataset loading, preprocessing, model, training, real-time inference server |

> **Ownership change (2026-11-07, Jerry's decision):** from now on the **FRONTEND agent on Jerry's laptop also owns and
> edits `backend/`**. William's agent: please don't edit `backend/` without checking with Jerry first (propose changes in
> `FROM_BACKEND.md` instead). Rule 2 below now reads "frontend agent: everything except `docs/handoff/FROM_BACKEND.md`".

Jerry has the final say on hardware and on the interface contract. The agents are on **different computers and
different Claude accounts**, so they never see each other's working folder. **The only link between them is the
shared GitHub repo.**

Rules:
1. **At the start of every task, run `git pull --rebase`.** At the end of every milestone, commit and `git push`.
   If the push is rejected, pull with rebase and push again. Never force-push or rewrite history.
2. **Never edit the other side's files.** Commit only your own paths (`git add frontend/ firmware/ shared/ docs/handoff/FROM_FRONTEND.md`,
   or `git add backend/ docs/handoff/FROM_BACKEND.md`). This keeps merge conflicts to near zero.
3. **Messages:** you write only in your own `docs/handoff/FROM_<YOU>.md` (append at the bottom), and you read the other
   side's file after every pull. Messages only arrive after the other side pushes.
4. **`docs/INTERFACE.md` is the contract.** Build exactly to it. To change it, propose the change in your handoff file and
   wait until both humans agree. Then the proposer updates the doc, bumps its version and pushes.
5. Each side must be **testable without the other** (the other side's code may be hours behind): the frontend ships a
   replay/simulated sensor publisher, and the backend ships a synthetic-session generator and a fake event publisher.
6. Before writing a lot of code, write a short plan and get your human's OK.
7. **Real data:** recorded sessions in `data/sessions/` **are committed** (a few MB each), so the backend computer gets
   Jerry's recordings via `git pull`. Models (`data/models/`) are not committed, because they are retrained locally.
8. Final integration runs on **Jerry's laptop** (it has the ESP32 and the camera). The backend must run there with just
   `git pull` and `pip install -r backend/requirements.txt`, with no machine-specific paths.

## Environment
- Jerry's laptop is **Windows**, using PowerShell. Paths look like `C:\Users\jerry\...`.
- Use **Python 3.12** in a repo-root venv: `py -3.12 -m venv .venv` then `.venv\Scripts\activate`.
  The system Python is 3.14, which MediaPipe may not support yet. Don't use it.
- Dependencies: `frontend/requirements.txt`, `backend/requirements.txt`.
- The ESP32 is flashed from the Arduino IDE (ESP32 core by Espressif). It appears as a COM port (e.g. COM7).
- `data/` holds recordings, gesture definitions and models. It is git-ignored and shared by both sides.

## Hardware facts everyone needs (full detail in docs/HARDWARE.md)
- EMG: 2 channels, ESP32 **GPIO32 = ch1, GPIO33 = ch2**, 12-bit ADC, sampled at **1000 Hz**, resting level ≈ 2048.
  The analog chain has already band-limited it to roughly 19–480 Hz, with a total gain of about 1100.
- IMU: **one** MPU-6050 on the back of the hand, I2C (SDA GPIO21, SCL GPIO22), address 0x68. Target rate is 200 Hz.
- Camera: laptop webcam, ~30 fps, MediaPipe Hand Landmarker (21 landmarks).
- **Safety: while electrodes are on a person, the ESP32 must be powered from a battery, a power bank or an
  unplugged laptop. Never from a wall-powered PC.** Mention this in any UI that starts EMG streaming.

## Existing working code (reference, don't edit)
- `legacy/emg_stream.ino`: the current 1 kHz 2-channel EMG streamer (9-byte binary packets, 921600 baud).
- `legacy/emg_click.py`: a working EMG-only left/right click demo. It contains a proven serial parser,
  preprocessing (20 Hz high-pass, 60 Hz notch, rest normalisation, 25 ms RMS envelope), onset detection,
  a 1D CNN and LDA baseline, and a pyqtgraph live UI. Reuse its ideas.
