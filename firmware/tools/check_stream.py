"""Read the ESP32 for N seconds and report rates, drops, checksum errors and signal sanity.

    python firmware/tools/check_stream.py --list              list serial ports
    python firmware/tools/check_stream.py --port COM7         10 s check
    python firmware/tools/check_stream.py --port COM7 --seconds 30

SAFETY: with electrodes on a person, the ESP32 must be powered from a battery, a power bank
or an UNPLUGGED laptop. Never from a wall-powered PC.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frontend.acquire.serial_v2 import (EmgPacket, ImuPacket, InfoPacket, Parser,  # noqa: E402
                                        StatusPacket)


def list_ports():
    import serial.tools.list_ports as lp
    for p in lp.comports():
        print(f"{p.device:8} {p.description}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list or not a.port:
        list_ports()
        if not a.port:
            print("\nGive --port COMx")
        return 0

    import serial
    print("SAFETY: electrodes on? Then the ESP32 must run from a battery / power bank / unplugged laptop.")
    ser = serial.Serial(a.port, a.baud, timeout=0.05)
    ser.reset_input_buffer()
    ser.write(b"i")
    p = Parser()
    emg_c, emg_t, emg_x = [], [], []
    imu_tag, imu_raw, imu_t = [], [], []
    status: StatusPacket | None = None
    first_status: StatusPacket | None = None
    info: InfoPacket | None = None
    gaps = lost = 0
    last_c = None
    t_start = time.perf_counter()
    t_last_print = t_start
    try:
        while time.perf_counter() - t_start < a.seconds:
            data = ser.read(4096)
            t = time.perf_counter()
            for pk in p.feed(data):
                if isinstance(pk, EmgPacket):
                    if last_c is not None and pk.counter != last_c + 5:
                        gaps += 1
                        lost += max(pk.counter - last_c - 5, 0)
                    last_c = pk.counter
                    emg_c.append(pk.counter)
                    emg_t.append(t)
                    emg_x.append(pk.data)
                elif isinstance(pk, ImuPacket):
                    imu_t.append(t)
                    imu_tag.append(pk.tag)
                    imu_raw.append(pk.raw)
                elif isinstance(pk, StatusPacket):
                    status = pk
                    first_status = first_status or pk
                elif isinstance(pk, InfoPacket):
                    info = pk
            if t - t_last_print >= 1.0:
                n = sum(len(x) for x in emg_x)
                print(f"  {t - t_start:4.1f}s  emg {n:6d}  imu {len(imu_tag):5d}  crc_err {p.crc_errors}"
                      f"  gaps {gaps}" + (f"  imu={status.imu_state_name}" if status else ""), flush=True)
                t_last_print = t
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    el = time.perf_counter() - t_start
    print("\n================ report ================")
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("  PASS  " if cond else "  FAIL  ") + msg)

    if not emg_c:
        if p.v1_hits > 10:
            print("  Old v1 firmware (legacy/emg_stream.ino) is running. Flash firmware/emg_imu_stream first.")
        else:
            print(f"  No v2 packets. {p.skipped_bytes} bytes of noise. Wrong port/baud, or the board isn't running.")
        return 1

    if info:
        print(f"  firmware {info.fw_version}  proto v{info.proto_version}  fs_emg {info.fs_emg}  fs_imu {info.fs_imu}"
              f"  accel +-{info.accel_fs_g} g  gyro +-{info.gyro_fs_dps} dps  dlpf {info.dlpf_cfg}")
    else:
        print("  (no INFO packet seen)")

    emg_c = np.array(emg_c, np.int64)
    emg_t = np.array(emg_t)
    X = np.concatenate(emg_x).astype(np.float64)
    span_c = (emg_c[-1] - emg_c[0]) / 1000.0
    span_t = emg_t[-1] - emg_t[0]
    rate_dev = len(X) / span_t if span_t > 0 else 0.0
    # ESP32 clock vs host: slope of counter vs the lower envelope of receive time
    ppm = float("nan")
    if span_t > 3:
        k = np.polyfit(emg_t, emg_c / 1000.0, 1)[0]
        ppm = (k - 1) * 1e6
    print(f"  EMG: {len(X)} samples in {el:.1f}s  -> {rate_dev:.1f} samples/s received, counter span {span_c:.2f}s"
          f"  (ESP clock {ppm:+.0f} ppm vs host)")
    check(abs(rate_dev - 1000) < 10 or span_t < 2, f"EMG rate ~1000 Hz ({rate_dev:.1f})")
    loss = lost / max(len(X) + lost, 1)
    check(loss < 5e-4, f"EMG counter gaps: {gaps} ({lost} samples lost = {100 * loss:.3f}%, limit 0.05%)")
    if gaps:
        print("        Rare single-packet losses with 0 checksum errors come from the USB link (a 64-byte block\n"
              "        of 0xFF replaces a packet), not the firmware. The reader fills and counts them.")
    check(p.crc_errors == 0, f"checksum errors: {p.crc_errors}, skipped bytes: {p.skipped_bytes}")
    for ch in range(2):
        x = X[:, ch]
        clip = 100 * np.mean((x < 30) | (x > 4065))
        print(f"  ch{ch + 1} (GPIO{32 + ch}): mean {x.mean():6.0f}  std {x.std():6.1f}  min {x.min():4.0f}"
              f"  max {x.max():4.0f}  clipped {clip:.2f}%")
        if clip > 30:
            seg = x[-4000:] - x[-4000:].mean()
            f = np.fft.rfftfreq(len(seg), 1e-3)[np.argmax(np.abs(np.fft.rfft(seg)))]
            check(False, f"ch{ch + 1} is saturated rail-to-rail ({clip:.0f}% clipped, strongest {f:.0f} Hz)")
            if abs(f - 60) < 2:
                print("        -> 60 Hz mains pickup on a floating input: electrodes not on skin, a clip loose,\n"
                      "           or the reference electrode missing. Expected with no electrodes attached.")
            continue
        check(1500 < x.mean() < 2600, f"ch{ch + 1} rests near 2048")
        check(x.std() > 1.0, f"ch{ch + 1} is alive (not stuck)")
        if clip > 1:
            print(f"        ch{ch + 1} clips; if it's only during hard contractions, see HARDWARE.md (1k -> 5.1k)")

    if imu_tag:
        tg = np.array(imu_tag, np.int64)
        R = np.array(imu_raw, np.float64)
        acc, gyr = R[:, 0:3] / 8192.0, R[:, 4:7] / 65.5
        steps = np.diff(tg)
        imu_span = imu_t[-1] - imu_t[0]
        rate = len(tg) / imu_span if imu_span > 0 else 0
        start_delay = (tg[0] - emg_c[0]) / 1000.0
        print(f"  IMU: {len(tg)} samples -> {rate:.1f} Hz; first sample {start_delay:.2f}s after EMG; "
              f"tag step 5 in {100 * np.mean(steps == 5):.1f}% of samples")
        if start_delay > 0.5:
            print("       (the IMU took a while to come up after reset; init retries every 100 ms at boot)")
        print(f"       accel mean g {np.round(acc.mean(0), 3)}  |a| {np.linalg.norm(acc, axis=1).mean():.3f}")
        print(f"       gyro mean dps {np.round(gyr.mean(0), 2)}  std {np.round(gyr.std(0), 2)}")
        print(f"       temp {R[:, 3].mean() / 340 + 36.53:.1f} C")
        check(abs(rate - 200) < 5 or span_t < 2, f"IMU rate ~200 Hz ({rate:.1f})")
        check(0.85 < np.linalg.norm(acc, axis=1).mean() < 1.15, "accel magnitude ~1 g when still")
    else:
        check(False, "IMU: no samples")
    if status:
        print(f"  STATUS: imu {status.imu_state_name} (who_am_i 0x{status.who_am_i:02X}, i2c {status.i2c_khz} kHz,"
              f" reads {status.imu_reads}, errors {status.imu_errors}, recoveries {status.imu_recoveries})")
        print(f"          emg late ticks {status.emg_late_ticks}, overflow {status.emg_overflow},"
              f" imu overflow {status.imu_overflow}, max jitter {status.max_jitter_us} us (last 1 s)")
        if status.imu_state != 0:
            print("          IMU not OK: check VIN->3V3, GND, SDA->GPIO21, SCL->GPIO22, and the STEMMA cable.")
        check(status.emg_overflow == (first_status.emg_overflow if first_status else 0),
              "no firmware-side EMG overflow during the test")
    else:
        print("  (no STATUS packet seen)")
    print("RESULT:", "ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
