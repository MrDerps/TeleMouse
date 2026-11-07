# Frontend

Run everything from the repo root with the venv active (`.venv\Scripts\activate`).
Setup once: `py -3.12 -m venv .venv` then `pip install -r frontend/requirements.txt`.

**Safety:** while electrodes are on a person, the ESP32 must be powered from a battery, a power bank
or an unplugged laptop. Never from a wall-powered PC.

| What | Command |
|---|---|
| Protocol self-test | `python -m shared.protocol --selftest` |
| Simulator on 5555 (no hardware) | `python -m frontend.tools.replay --sim` (`--no-imu`, `--no-hand`, `--drop-rate 0.01`, `--truth truth.csv`, `--seed N`) |
| Simulator self-test | `python -m frontend.tools.replay --selftest` |
| Watch rates / gaps on 5555 | `python -m frontend.tools.monitor` (`--events` for the backend's 5556) |
| Live plots of 5555 | `python -m frontend.tools.viewer` |
| Serial ports | `python -m frontend.acquire --list` |
| ESP32 stream check (10 s report) | `python firmware/tools/check_stream.py --port COM7` |
| Acquisition: ESP32 + webcam → 5555 | `python -m frontend.acquire --port COM7` (`--no-camera`, `--no-serial`, `--camera N`) |
| Camera latency (sync) calibration | `python -m frontend.acquire --port COM7 --calibrate-sync` then tap the hand on the desk ~5× |
| Self-tests | `python -m frontend.acquire.serial_v2 --selftest`, `python -m frontend.acquire.serial_reader`, `python -m frontend.acquire.sync --selftest` |

Firmware: `firmware/emg_imu_stream/emg_imu_stream.ino` (board "ESP32 Dev Module", core 3.x).
Serial format: `firmware/PROTOCOL.md`.

Flash from the command line (or open the sketch in the Arduino IDE and press Upload):
```
"C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe" compile --upload -p COM7 --fqbn esp32:esp32:esp32 firmware\emg_imu_stream
```

Machine settings (serial port, camera index, calibrated camera latency) live in `data/frontend_config.json`
(per machine, not committed). The MediaPipe model downloads on first run to `frontend/assets/hand_landmarker.task` (git-ignored).
Only one process can publish on 5555: stop the simulator before running acquire.
