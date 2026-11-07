"""ESP32 serial reader: v2 packets -> regular, host-timed EMG and IMU blocks.

Runs one background thread. Produces 20 ms blocks:
    on_block(emg_t0, emg_index0, emg (20,2) uint16, imu_t0, imu_index0, imu (4,6) float32)

Time mapping (INTERFACE §2): host time of EMG sample index i is
    t(i) = offset + i * period
`period` is a least-squares fit of receive time against counter over the last ~60 s
(the ESP32 crystal is off by tens of ppm); `offset` is the lower envelope (minimum of
t_rx - i*period) over the last ~10 s, because USB only ever adds delay.

The IMU is phase-locked to EMG (one read every 5th tick), so IMU sample k is placed at
EMG index 5k+4. Missing IMU samples become NaN rows; the IMU grid stays regular.
Lost EMG samples are filled by repeating the last value and counted in `dropped`.
"""
from __future__ import annotations

import collections
import threading
import time

import numpy as np

from .serial_v2 import EmgPacket, ImuPacket, InfoPacket, Parser, StatusPacket

BLOCK = 20          # EMG samples per published block (20 ms)
IMU_PER_BLOCK = 4


class ClockFit:
    def __init__(self, fs=1000.0):
        self.nominal = 1.0 / fs
        self.period = self.nominal
        self.pts = collections.deque()           # (index_of_last_sample, t_rx), last 60 s
        self.buckets = collections.deque()       # per 0.5 s: the lowest-latency point [key, i, t, r]
        self.offset = None
        self._last_fit = 0.0

    def add(self, idx_last: int, t_rx: float):
        self.pts.append((idx_last, t_rx))
        while t_rx - self.pts[0][1] > 60.0:
            self.pts.popleft()
        if t_rx - self._last_fit > 1.0 and len(self.pts) > 400:
            a = np.array(self.pts)
            if a[-1, 1] - a[0, 1] > 5.0:
                k = np.polyfit(a[:, 0] - a[0, 0], a[:, 1], 1)[0]
                if abs(k / self.nominal - 1) < 500e-6:   # sanity: within 500 ppm
                    self.period = k
            self._last_fit = t_rx
        # lower envelope (USB only adds delay) over the last ~10 s
        key = int(t_rx * 2)
        r = t_rx - idx_last * self.nominal
        if not self.buckets or self.buckets[-1][0] != key:
            self.buckets.append([key, idx_last, t_rx, r])
            while len(self.buckets) > 20:
                self.buckets.popleft()
        elif r < self.buckets[-1][3]:
            self.buckets[-1][1:] = [idx_last, t_rx, r]
        self.offset = min(t - i * self.period for _, i, t, _ in self.buckets)

    def t(self, idx: float) -> float:
        return self.offset + idx * self.period

    def reset(self):
        self.pts.clear()
        self.offset = None
        self.period = self.nominal


class SerialReader:
    def __init__(self, port: str, baud: int = 921600, on_block=None, on_log=print):
        self.port, self.baud = port, baud
        self.on_block = on_block or (lambda *a: None)
        self.log = on_log
        self.parser = Parser()
        self.clock = ClockFit()
        self.info: InfoPacket | None = None
        self.status: StatusPacket | None = None
        self.dropped = 0
        self.emg_count = 0
        self.imu_valid_count = 0
        self.last_rx = 0.0
        self._base = 0                 # added to the ESP counter so indices survive a board reset
        self._last_counter = None
        self._pending = {}             # global EMG index -> (2,) uint16, waiting to fill a block
        self._next_block = None        # global index of the next block to publish
        self._last_row = np.full(2, 2048, np.uint16)
        self._imu = {}                 # imu index k -> (6,) float32
        self._stop = threading.Event()
        self._thread = None
        self.ser = None

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        import serial
        self.ser = serial.Serial(self.port, self.baud, timeout=0.02)
        self.ser.reset_input_buffer()
        self.ser.write(b"i")
        self._thread = threading.Thread(target=self._run, name="serial", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(2)
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ work
    def _run(self):
        while not self._stop.is_set():
            try:
                data = self.ser.read(4096)
            except Exception as e:  # unplugged
                self.log(f"[serial] read failed: {e}")
                break
            if not data:
                continue
            t_rx = time.perf_counter()
            for pk in self.parser.feed(data):
                self._handle(pk, t_rx)

    def _handle(self, pk, t_rx):
        if isinstance(pk, EmgPacket):
            c = pk.counter
            if self._last_counter is not None and c + 5000 < self._last_counter:
                self.log("[serial] ESP32 counter went backwards: board reset, re-syncing clock")
                self._base = self._next_block or 0
                self.clock.reset()
                self._pending.clear()
                self._imu.clear()
                self._next_block = None
            self._last_counter = c
            g = self._base + c
            self.last_rx = t_rx
            self.clock.add(g + len(pk.data) - 1, t_rx)
            if self._next_block is None:
                self._next_block = g - g % BLOCK + (BLOCK if g % BLOCK else 0)
                self._first_block = self._next_block
            for k, row in enumerate(pk.data):
                if g + k >= self._next_block:
                    self._pending[g + k] = row
            self._flush(g + len(pk.data))
        elif isinstance(pk, ImuPacket):
            if self._last_counter is None:
                return
            g_tag = self._base + pk.tag
            # tag = samples taken at read start, normally 5k+5 (read ~ EMG index 5k+4);
            # tolerate the read starting up to 2 ticks late
            k = (g_tag - 3) // 5
            self._imu[k] = pk.scaled()
            self.imu_valid_count += 1
        elif isinstance(pk, StatusPacket):
            self.status = pk
        elif isinstance(pk, InfoPacket):
            if self.info is None:
                self.log(f"[serial] firmware {pk.fw_version}, protocol v{pk.proto_version}")
            self.info = pk

    def _flush(self, have_until: int):
        # publish complete blocks; a block is complete when later samples have arrived
        while self._next_block is not None and have_until >= self._next_block + BLOCK:
            i0 = self._next_block
            blk = np.empty((BLOCK, 2), np.uint16)
            for j in range(BLOCK):
                row = self._pending.pop(i0 + j, None)
                if row is None:
                    row = self._last_row
                    self.dropped += 1
                blk[j] = row
                self._last_row = row
            # IMU for the PREVIOUS block (its packets have had 20 ms to arrive)
            ib = i0 - BLOCK
            imu = np.full((IMU_PER_BLOCK, 6), np.nan, np.float32)
            k0 = ib // 5
            for j in range(IMU_PER_BLOCK):
                v = self._imu.pop(k0 + j, None)
                if v is not None:
                    imu[j] = v
            for old in [k for k in self._imu if k < k0]:
                del self._imu[old]
            self.emg_count += BLOCK
            if ib < self._first_block:
                imu = None                    # nothing before the first block
            self.on_block(self.clock.t(i0), i0, blk, self.clock.t(ib + 4), k0, imu)
            self._next_block += BLOCK


def _selftest():
    """Feed a synthetic packet stream (ESP clock +40 ppm, 1-6 ms USB jitter, one lost EMG
    packet, one lost IMU sample) and check timing, gap filling and IMU alignment."""
    from .serial_v2 import enc_emg, enc_imu, enc_status
    rng = np.random.default_rng(0)
    blocks = []
    rd = SerialReader("none", on_block=lambda *a: blocks.append(a), on_log=lambda m: None)
    T0, drift = 100.0, 40e-6
    lost_emg, lost_imu = 1000, 777          # EMG packet with counter 5000 and IMU sample 777 never arrive
    for m in range(4000):                   # 20 s of 5 ms packets
        c = 5 * m
        t_true_last = T0 + (c + 4) * 1e-3 * (1 + drift)
        t_rx = t_true_last + rng.uniform(0.001, 0.006)
        data = b""
        if m != lost_emg:
            data += enc_emg(c, np.full((5, 2), 2048 + (m % 100), np.uint16))
        if m != lost_imu:
            data += enc_imu(c + 5, [8192, 0, 0, 0, 0, 0, 0])
        if m % 200 == 0:
            data += enc_status(imu_state=0)
        for pk in rd.parser.feed(data):
            rd._handle(pk, t_rx)
    assert rd.dropped == 5, rd.dropped
    t0s = np.array([b[0] for b in blocks])
    i0s = np.array([b[1] for b in blocks])
    assert np.all(np.diff(i0s) == BLOCK)
    true = T0 + i0s * 1e-3 * (1 + drift)
    err = (t0s - true)[len(t0s) // 2:]       # after the fit has converged
    assert np.abs(err).max() < 0.002, np.abs(err).max()
    assert abs(rd.clock.period / 1e-3 - 1 - drift) < 15e-6, rd.clock.period   # 15 ppm = 0.3 ms over 20 s
    imus = [b[5] for b in blocks if b[5] is not None]
    allimu = np.concatenate(imus)
    assert np.isnan(allimu[:, 0]).sum() == 1, np.isnan(allimu[:, 0]).sum()
    assert np.nanmax(np.abs(allimu[:, 0] - 1)) < 1e-6
    ks = [b[4] for b in blocks if b[5] is not None]
    assert all(b[4] * 5 == b[1] - BLOCK for b in blocks if b[5] is not None) and np.all(np.diff(ks) == 4)
    print(f"serial_reader selftest: OK (timing error after convergence max {1000 * np.abs(err).max():.2f} ms)")


if __name__ == "__main__":
    _selftest()
