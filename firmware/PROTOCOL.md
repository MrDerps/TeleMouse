# Serial protocol v2 (ESP32 → host)

Internal to the frontend: the backend never sees it (it gets INTERFACE.md §3 on port 5555).
Firmware: `firmware/emg_imu_stream/emg_imu_stream.ino`. Host parser: `frontend/acquire/serial_v2.py`.

## Link
921600 baud, 8N1, USB-UART. Opening the port usually resets the board (DTR), and the first
packet after boot is INFO.

## Frame
```
0xA5 0x5A | type u8 | len u8 | payload (len bytes) | crc8
```
- `crc8`: CRC-8, polynomial 0x07, init 0x00, no reflection, no final xor, computed over
  `type`, `len` and `payload` (not over the sync bytes).
- All multi-byte payload fields are **little-endian**, except the raw IMU register bytes,
  which are copied straight from the MPU-6050 (big-endian).
- A parser must skip packets of an unknown `type` by using `len`, so new types can be added.
- The sync bytes differ from v1 (`0xAA 0x55`), so the host can recognise old firmware.

## Types
### 0x01 EMG (200/s, every 5 ms), len = 24
| offset | field | type |
|---|---|---|
| 0 | `counter`: sample index of the first sample | u32 |
| 4 | 5 × (`ch1`, `ch2`) raw 12-bit ADC counts | u16 ×10 |

The counter counts 1 kHz hardware-timer ticks since boot, so it **is** the EMG timebase.
Consecutive packets have `counter` += 5. A bigger step means lost samples (writer overflow or
a corrupted packet).

### 0x02 IMU (200/s), len = 18
| offset | field | type |
|---|---|---|
| 0 | `emg_counter`: EMG sample counter (samples taken so far) at the moment the read started | u32 |
| 4 | MPU-6050 registers 0x3B..0x48: ax, ay, az, temp, gx, gy, gz | i16 **big-endian** ×7 |

IMU reads are triggered by every 5th EMG tick, so `emg_counter` normally steps by 5.
The sample with tag `c` is closest in time to EMG sample index `c - 1`.
Scale: accel ±4 g → `/8192` g; gyro ±500 °/s → `/65.5` °/s; temp °C = `raw/340 + 36.53`.
The MPU's 44 Hz DLPF adds a group delay of about 5 ms, which the host may correct.
No IMU packets are sent while the IMU is missing or failed (the host fills NaN).

### 0x03 STATUS (1/s), len = 38
| offset | field | type |
|---|---|---|
| 0 | `uptime_ms` | u32 |
| 4 | `emg_counter` | u32 |
| 8 | `imu_state`: 0 ok, 1 missing (never worked), 2 failed (worked, then lost; retrying), 3 initialising | u8 |
| 9 | `who_am_i` (0x68 genuine; 0 = no answer) | u8 |
| 10 | `i2c_khz` (400, drops to 100 after a failure) | u16 |
| 12 | `imu_reads`, `imu_errors`, `imu_recoveries` (cumulative) | u32 ×3 |
| 24 | `emg_late_ticks`: timer ticks the sampler handled late (cumulative) | u32 |
| 28 | `emg_overflow`: EMG samples dropped because serial fell behind (cumulative) | u32 |
| 32 | `imu_overflow` (cumulative) | u32 |
| 36 | `max_jitter_us`: worst |interval − 1000 µs| between EMG reads since the last STATUS | u16 |

### 0x04 INFO (at boot, every 5 s, and when the host sends `i`), len = 10 + version length
| offset | field | type |
|---|---|---|
| 0 | `proto_version` (= 2) | u8 |
| 1 | `fs_emg` (1000) | u16 |
| 3 | `fs_imu` (200) | u16 |
| 5 | `emg_batch` (5) | u8 |
| 6 | `accel_fs_g` (4) | u8 |
| 7 | `gyro_fs_dps` (500) | u16 |
| 9 | `dlpf_cfg` (3) | u8 |
| 10 | firmware version, ASCII, no terminator | bytes |

## Host → ESP32
Single bytes: `i` = send INFO now. Everything else is ignored.

## Bandwidth
EMG 29 B × 200/s + IMU 23 B × 200/s + STATUS ≈ 10.5 kB/s, about 11 % of the link.

## Timing design
- A hardware timer interrupt (1 kHz) wakes the top-priority sampler task on core 1, which reads
  GPIO32 then GPIO33 back-to-back. If the task is ever woken late it still takes one sample per tick,
  so the counter never slips; `emg_late_ticks` and `max_jitter_us` report it.
- The IMU task runs on core 0, so I2C stalls (up to the 10 ms Wire timeout) can never delay EMG.
- The serial writer is `loop()` at the lowest priority on core 1. Ring buffers hold about 1 s of EMG.
