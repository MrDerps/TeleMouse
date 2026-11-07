# Hardware reference

## 1. sEMG front end (built and working)

### Signal chain (per channel)
```
Ag/AgCl gel snap electrodes (E+, E−) ─ alligator clips ─ 10 kΩ series (input protection)
  → MCP6N11-010 instrumentation amp, gain 11 (RF 100k OUT→VFG, RG 10k VFG→VREF), REF pin = VREF
  → 0.1 µF + 82 kΩ to VREF high-pass (≈19 Hz)
  → MCP6004 non-inverting stage, gain 101 (100k / 1k to VREF), 3.3 nF across Rf (≈480 Hz low-pass)
  → 3.3 kΩ + 47 nF anti-alias (≈1 kHz) → ESP32 ADC1 pin
```
- Supply 3.3 V from the ESP32. **VREF = 1.65 V** (100k/100k divider, 10 µF, buffered by MCP6004 op-amp C).
  Op-amp D is tied off.
- Reference electrode (elbow bone) connects through 100 kΩ to VREF.
- Total gain ≈ 1100. Typical forearm sEMG of 0.1–1 mV becomes 0.1–1 V around the 1.65 V midpoint.
- Channel 1 → **GPIO32**, channel 2 → **GPIO33** (both ADC1, so they work with Wi-Fi on). Don't use ADC2 pins.
- ESP32 ADC: 12-bit, `analogSetAttenuation(ADC_11db)` (0–3.3 V range), resting reading ≈ 2048.
  The ADC is non-linear in the last ~0.1 V near each rail. Treat readings near 0 or 4095 as clipping.
- If a hard contraction clips, the fix is to swap that channel's 1k for 5.1k (gain ≈ 21 in stage 2).

### Electrodes and placement
- SparkFun 12969 disposable Ag/AgCl gel snap pads. Pairs are 2–3 cm apart along the muscle, and the reference is on the elbow.
- The electrode placement will not necessarily be known. The NN may have to reset/retrain EMG readings if the pads are removed or changeds
- The signal changes every time the pads go back on, so each session needs a rest calibration and benefits from fresh training data.

### Known preprocessing that works (from legacy/emg_click.py)
- Causal 2nd-order Butterworth high-pass at 20 Hz, plus a 60 Hz notch (Q=30). Mains in Canada is 60 Hz.
- Rest normalisation: per-channel robust std = 1.4826·MAD over a ~3–10 s relaxed period. Signals are then in "× rest noise" units.
- Envelope: causal 25 ms moving RMS.
- Tap onset: when sqrt(env1² + env2²) crosses ~3× its resting median, with a 350 ms refractory period.

### Existing serial protocol v1 (legacy, EMG only)
9-byte packets at 1 kHz, 921600 baud: `0xAA 0x55 | counter u16 LE | ch1 u16 LE | ch2 u16 LE | xor(bytes 2..7)`.
Both channels are read back-to-back in the same 1 ms tick, so each packet is a time-aligned pair.
**The frontend will replace this with protocol v2 (EMG and IMU).**

## 2. IMU (one, on the back of the hand)
- **One Adafruit MPU-6050** (product 3886, STEMMA QT / Qwiic, onboard regulator, 3–5 V input). A second unit exists as a spare.
- Wiring to the ESP32: VIN → 3V3, GND → GND, SDA → **GPIO21**, SCL → **GPIO22**. It's the only device on the bus.
- Address **0x68** (default, AD0 low).
- Placement: flat on the **back of the hand** (over the middle of the metacarpals), strapped firmly.
  Fill in the mounting orientation (which axis points toward the fingertips, which axis points out of the back of the hand)
  once mounted. The backend needs this for augmentation, and the frontend needs it for pointer control.
- Suggested config: accel ±4 g, gyro ±500 °/s, DLPF ~44 Hz, 200 Hz output.
- Long I2C leads to the hand are fragile. Keep them twisted and as short as possible, use 100 kHz if 400 kHz is unreliable,
  and detect and recover from bus errors without crashing the EMG sampling.

## 3. Camera
- Laptop webcam  via OpenCV, ~30 fps, 640×480 is enough.
- MediaPipe **Hand Landmarker** (Tasks API, VIDEO/LIVE_STREAM mode). It gives 21 landmarks: x and y normalised to the image,
  and z as relative depth. It also gives handedness and a presence score.
- Webcam latency is about 50–150 ms, so each frame is timestamped at capture and the latency is calibrated (see INTERFACE.md §2).

## 4. Host
- Windows laptop, Python 3.12 venv. The UI uses PyQt6 + pyqtgraph, which already work on this machine. OS mouse and keyboard are driven by `pynput`.
