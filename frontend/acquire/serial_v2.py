"""Parser (and test encoder) for serial protocol v2 - see firmware/PROTOCOL.md.

Pure byte handling, no threads or ports, so it is easy to test:

    python -m frontend.acquire.serial_v2 --selftest
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

SYNC = b"\xa5\x5a"
T_EMG, T_IMU, T_STATUS, T_INFO = 0x01, 0x02, 0x03, 0x04
EMG_BATCH = 5
ACC_LSB_PER_G = 8192.0      # +-4 g
GYRO_LSB_PER_DPS = 65.5     # +-500 deg/s
IMU_STATES = {0: "ok", 1: "missing", 2: "failed", 3: "init"}

_CRC = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = ((_c << 1) ^ 0x07) & 0xFF if _c & 0x80 else (_c << 1) & 0xFF
    _CRC.append(_c)
_CRC = bytes(_CRC)


def crc8(data) -> int:
    c = 0
    for b in data:
        c = _CRC[c ^ b]
    return c


@dataclass
class EmgPacket:
    counter: int              # sample index of the first row
    data: np.ndarray          # (5, 2) uint16, columns ch1, ch2


@dataclass
class ImuPacket:
    tag: int                  # EMG samples taken when the read started (~ EMG index tag-1)
    raw: np.ndarray           # (7,) int16: ax ay az temp gx gy gz

    def scaled(self) -> np.ndarray:
        """-> float32 [ax, ay, az (g), gx, gy, gz (deg/s)]"""
        r = self.raw.astype(np.float32)
        return np.concatenate([r[0:3] / ACC_LSB_PER_G, r[4:7] / GYRO_LSB_PER_DPS]).astype(np.float32)


@dataclass
class StatusPacket:
    uptime_ms: int
    emg_counter: int
    imu_state: int
    who_am_i: int
    i2c_khz: int
    imu_reads: int
    imu_errors: int
    imu_recoveries: int
    emg_late_ticks: int
    emg_overflow: int
    imu_overflow: int
    max_jitter_us: int

    @property
    def imu_state_name(self) -> str:
        return IMU_STATES.get(self.imu_state, f"?{self.imu_state}")


@dataclass
class InfoPacket:
    proto_version: int
    fs_emg: int
    fs_imu: int
    emg_batch: int
    accel_fs_g: int
    gyro_fs_dps: int
    dlpf_cfg: int
    fw_version: str


_STATUS_FMT = "<IIBBHIIIIIIH"     # 38 bytes
_INFO_FMT = "<BHHBBHB"            # 10 bytes + version string


@dataclass
class Parser:
    """feed(bytes) -> list of packets. Counts problems as it goes."""
    buf: bytearray = field(default_factory=bytearray)
    packets: int = 0
    crc_errors: int = 0
    skipped_bytes: int = 0
    unknown_types: int = 0
    v1_hits: int = 0          # valid-looking legacy v1 (0xAA 0x55) packets seen

    def feed(self, data: bytes) -> list:
        b = self.buf
        b += data
        out = []
        i, n = 0, len(b)
        while True:
            j = b.find(SYNC, i)
            if j < 0:
                # keep a trailing 0xA5 (maybe half a sync), but never one we already consumed
                keep = n - 1 if n > i and b[-1] == 0xA5 else n
                self._skipped(b, i, keep)
                i = keep
                break
            if j > i:
                self._skipped(b, i, j)
            if j + 4 > n:
                i = j
                break
            ln = b[j + 3]
            end = j + 5 + ln
            if end > n:
                i = j
                break
            body = bytes(b[j + 2:j + 4 + ln])
            if crc8(body) != b[end - 1]:
                self.crc_errors += 1
                self.skipped_bytes += 1
                i = j + 1
                continue
            pkt = self._decode(body[0], body[2:])
            if pkt is not None:
                out.append(pkt)
            self.packets += 1
            i = end
        del b[:i]
        return out

    def _skipped(self, b, a, z):
        if z <= a:
            return
        self.skipped_bytes += z - a
        seg = bytes(b[a:z])
        k = seg.find(b"\xaa\x55")
        while 0 <= k <= len(seg) - 9:
            p = seg[k:k + 9]
            if p[2] ^ p[3] ^ p[4] ^ p[5] ^ p[6] ^ p[7] == p[8]:
                self.v1_hits += 1
            k = seg.find(b"\xaa\x55", k + 1)

    def _decode(self, t, p):
        try:
            if t == T_EMG:
                c = struct.unpack_from("<I", p)[0]
                d = np.frombuffer(p, "<u2", offset=4).reshape(-1, 2).copy()
                return EmgPacket(c, d)
            if t == T_IMU:
                tag = struct.unpack_from("<I", p)[0]
                raw = np.frombuffer(p, ">i2", offset=4, count=7).astype(np.int16)
                return ImuPacket(tag, raw)
            if t == T_STATUS:
                return StatusPacket(*struct.unpack_from(_STATUS_FMT, p))
            if t == T_INFO:
                f = struct.unpack_from(_INFO_FMT, p)
                return InfoPacket(*f, p[struct.calcsize(_INFO_FMT):].decode("ascii", "replace"))
        except (struct.error, ValueError):
            self.crc_errors += 1
            return None
        self.unknown_types += 1
        return None


# ----------------------------------------------------------------------------- encoder (tests/sim)
def frame(t: int, payload: bytes) -> bytes:
    body = bytes([t, len(payload)]) + payload
    return SYNC + body + bytes([crc8(body)])


def enc_emg(counter: int, data) -> bytes:
    return frame(T_EMG, struct.pack("<I", counter) + np.asarray(data, "<u2").tobytes())


def enc_imu(tag: int, raw7) -> bytes:
    return frame(T_IMU, struct.pack("<I", tag) + np.asarray(raw7, ">i2").tobytes())


def enc_status(**kw) -> bytes:
    f = dict(uptime_ms=0, emg_counter=0, imu_state=0, who_am_i=0x68, i2c_khz=400, imu_reads=0,
             imu_errors=0, imu_recoveries=0, emg_late_ticks=0, emg_overflow=0, imu_overflow=0, max_jitter_us=0)
    f.update(kw)
    return frame(T_STATUS, struct.pack(_STATUS_FMT, *f.values()))


def enc_info(fw="2.0.0") -> bytes:
    return frame(T_INFO, struct.pack(_INFO_FMT, 2, 1000, 200, 5, 4, 500, 3) + fw.encode())


def _selftest():
    rng = np.random.default_rng(0)
    stream = bytearray(enc_info())
    for k in range(400):
        stream += enc_emg(k * 5, rng.integers(0, 4096, (5, 2)))
        stream += enc_imu(k * 5 + 5, [8192, 0, 0, 0, 655, 0, -655])
    stream += enc_status(emg_counter=2000, imu_state=0)
    stream += bytes([0xAA, 0x55, 1, 0, 0, 8, 0, 8, 0 ^ 1 ^ 8 ^ 8]) * 3    # legacy v1 junk
    clean = bytes(stream)
    # corrupt one byte in packet ~100 and inject garbage
    bad = bytearray(clean)
    bad[2000] ^= 0xFF
    bad[5000:5000] = b"\x00\xa5\x13\x37"
    p = Parser()
    pk = []
    for i in range(0, len(bad), 97):          # odd chunk size to split packets
        pk += p.feed(bytes(bad[i:i + 97]))
    emg = [x for x in pk if isinstance(x, EmgPacket)]
    imu = [x for x in pk if isinstance(x, ImuPacket)]
    st = [x for x in pk if isinstance(x, StatusPacket)]
    info = [x for x in pk if isinstance(x, InfoPacket)]
    assert info and info[0].fw_version == "2.0.0" and info[0].fs_emg == 1000
    assert len(emg) + len(imu) == 798, (len(emg), len(imu))      # the two damaged packets are lost
    assert p.crc_errors >= 1 and p.v1_hits == 3, (p.crc_errors, p.v1_hits)
    assert st and st[0].emg_counter == 2000 and st[0].imu_state_name == "ok"
    s = imu[0].scaled()
    assert abs(s[0] - 1) < 1e-6 and abs(s[3] - 10) < 1e-6 and abs(s[5] + 10) < 1e-6, s
    # clean stream parses fully
    p2 = Parser()
    assert len(p2.feed(clean)) == 802 and p2.crc_errors == 0
    print("serial_v2 selftest: OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
