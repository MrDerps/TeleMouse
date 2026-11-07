#!/usr/bin/env python3
"""
emg_click.py - tell an index-finger tap (LEFT click) from a middle-finger tap
(RIGHT click) using 2 EMG channels streamed from the ESP32 (emg_stream.ino).

    python emg_click.py ports                       list serial ports
    python emg_click.py view   --port COM5          live graphs (use this to place pads)
    python emg_click.py record --port COM5          guided recording (~3.5 min)
    python emg_click.py train  recordings/*.npz     train 1D CNN (+ LDA baseline)
    python emg_click.py run    --port COM5          live left/right detection
    python emg_click.py run    --port COM5 --click  ...and actually click the mouse

Use  --port sim  to try any mode without the hardware.
On Mac/Linux the port looks like /dev/cu.usbserial-XXXX or /dev/ttyUSB0.

How it works
  1. Both channels are filtered the same way live and in training:
     20 Hz high-pass + 60 Hz notch, then scaled by the resting noise level
     ("x rest" units), so the numbers mean the same thing every session.
  2. A tap is detected when the combined 25 ms RMS envelope crosses
     THR x its resting level.  That instant (the onset) lines the window up.
  3. The window (50 ms before -> 250 ms after onset, both channels, raw +
     envelope = 4 x 300 samples) goes to the classifier:  rest / left / right.
"""
import argparse
import glob
import os
import random
import struct
import sys
import threading
import time

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

FS = 1000            # samples per second (must match the ESP32 sketch)
PRE, POST = 50, 250  # window around the tap onset, in samples (= ms)
WIN = PRE + POST
JIT = 20             # extra margin kept in training for random time-shift augmentation
ENV = 25             # envelope RMS length (ms)
REFR = 350           # min gap between two taps (ms)
CLASSES = ["rest", "left", "right"]
EV_RELAX, EV_LEFT, EV_RIGHT, EV_CAL, EV_END = 0, 1, 2, 3, 9
COL1, COL2, COLD = "#4ea1ff", "#ff9f43", "#e5ebe6"


# ----------------------------------------------------------------------------- sources
class SerialSource:
    """Reads 9-byte packets from the ESP32 in a background thread."""

    def __init__(self, port, baud):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.buf = bytearray()
        self.out = []
        self.lock = threading.Lock()
        self.last = None
        self.dropped = 0
        self.bad = 0
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            try:
                data = self.ser.read(4096)
            except Exception:
                break
            if data:
                self._parse(data)

    def _parse(self, data):
        b = self.buf
        b += data
        got, i, n = [], 0, len(b)
        while True:
            j = b.find(b"\xaa\x55", i)
            if j < 0:
                i = max(i, n - 1)          # keep a possible half header
                break
            if j + 9 > n:
                i = j                      # incomplete packet, wait for more
                break
            if b[j + 2] ^ b[j + 3] ^ b[j + 4] ^ b[j + 5] ^ b[j + 6] ^ b[j + 7] != b[j + 8]:
                self.bad += 1
                i = j + 1
                continue
            cnt, a, c = struct.unpack_from("<HHH", b, j + 2)
            if self.last is not None:
                gap = (cnt - self.last - 1) & 0xFFFF
                if 0 < gap < 1000:
                    self.dropped += gap
            self.last = cnt
            got.append((a, c))
            i = j + 9
        del b[:i]
        if got:
            with self.lock:
                self.out.extend(got)

    def read(self):
        with self.lock:
            s, self.out = self.out, []
        return np.asarray(s, dtype=np.float32).reshape(-1, 2)

    def close(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass


class SimSource:
    """Fake EMG so you can try everything without hardware.
    Index taps are stronger on ch1, middle taps stronger on ch2."""

    def __init__(self, clock=time.monotonic, seed=None):
        self.clock = clock
        self.rng = np.random.default_rng(seed)
        self.t0 = clock()
        self.n = 0
        self.taps = []
        self.auto = True
        self.next_auto = 5 * FS
        self.dropped = self.bad = 0

    def cue_event(self, code):          # called by the recorder when a cue appears
        self.auto = False
        self._tap(self.n + int(self.rng.uniform(250, 500)), code)

    def _tap(self, start, code):
        g = np.array([1.0, 0.45]) if code == EV_LEFT else np.array([0.5, 1.0])
        g = g * self.rng.uniform(0.8, 1.2, 2) * self.rng.uniform(150, 400)
        self.taps.append((start, int(self.rng.uniform(120, 200)), g))

    def read(self):
        k = int((self.clock() - self.t0) * FS) - self.n
        if k <= 0:
            return np.zeros((0, 2), np.float32)
        idx = np.arange(self.n, self.n + k)
        x = 2048 + self.rng.normal(0, 5, (k, 2)) + 10 * np.sin(2 * np.pi * 60 * idx / FS)[:, None]
        if self.auto and self.n >= self.next_auto:
            self._tap(self.n, int(self.rng.integers(1, 3)))
            self.next_auto = self.n + int(self.rng.uniform(1500, 3000))
        keep = []
        for s, dur, g in self.taps:
            a, b = max(s, self.n), min(s + dur, self.n + k)
            if a < b:
                w = np.hanning(dur)[a - s:b - s]
                x[a - self.n:b - self.n] += self.rng.normal(0, 1, (b - a, 2)) * w[:, None] * g
            if s + dur > self.n + k:
                keep.append((s, dur, g))
        self.taps = keep
        self.n += k
        return np.clip(x, 0, 4095).astype(np.float32)

    def close(self):
        pass


def open_source(port, baud):
    if port == "sim":
        return SimSource()
    if not port:
        sys.exit("Give a port, e.g. --port COM5  (run `python emg_click.py ports`), or --port sim")
    return SerialSource(port, baud)


# ----------------------------------------------------------------------------- signal processing
class Preproc:
    """Causal 20 Hz high-pass + 60 Hz notch on both channels. Streams chunk by chunk."""

    def __init__(self, notch=60.0):
        hp = butter(2, 20, "highpass", fs=FS, output="sos")
        b, a = iirnotch(notch, 30, fs=FS)
        self.sos = np.vstack([hp, tf2sos(b, a)])
        self.zi = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


def moving_rms(y, tail=None):
    """Causal 25 ms RMS envelope. `tail` carries state between chunks."""
    if tail is None:
        tail = np.zeros((ENV - 1, y.shape[1]))
    full = np.vstack([tail, y ** 2])
    c = np.vstack([np.zeros((1, y.shape[1])), np.cumsum(full, axis=0)])
    env = np.sqrt(np.maximum(c[ENV:] - c[:-ENV], 0.0) / ENV)
    return env, full[-(ENV - 1):]


def combined(env, std):
    e = env / std
    return np.hypot(e[:, 0], e[:, 1])


def make_window(yf, env, std):
    """-> (4, L) float32: ch1, ch2 (filtered, x rest) + their envelopes."""
    return np.concatenate([yf / std, env / std], axis=1).T.astype(np.float32)


def rest_stats(yf, env):
    """Resting noise per channel + resting combined envelope. Uses medians so a stray
    twitch during calibration doesn't inflate them."""
    std = np.maximum(1.4826 * np.median(np.abs(yf - np.median(yf, 0)), 0), 1e-3)
    return std, float(np.median(combined(env, std)))


def first_crossing(d, thr, a, b):
    a = max(a, 1)
    seg = d[a - 1:b] >= thr
    idx = np.flatnonzero(seg[1:] & ~seg[:-1])
    return a + int(idx[0]) if len(idx) else None


class LivePipe:
    """Keeps the last 20 s of filtered signal + envelope, indexed by absolute sample number."""

    def __init__(self, keep_s=20):
        self.pp = Preproc()
        self.tail = None
        self.keep = keep_s * FS
        self.yf = np.zeros((0, 2))
        self.env = np.zeros((0, 2))
        self.start = 0
        self.n = 0

    def push(self, raw):
        if len(raw) == 0:
            return
        yf = self.pp(raw)
        env, self.tail = moving_rms(yf, self.tail)
        self.yf = np.vstack([self.yf, yf])
        self.env = np.vstack([self.env, env])
        self.n += len(raw)
        extra = len(self.yf) - self.keep
        if extra > 0:
            self.yf, self.env = self.yf[extra:], self.env[extra:]
            self.start += extra

    def get(self, a, b):
        a = max(a, self.start)
        return self.yf[a - self.start:b - self.start], self.env[a - self.start:b - self.start], a


# ----------------------------------------------------------------------------- models
def make_net(cin=4, ncls=3):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv1d(cin, 16, 9, padding=4), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(3),
        nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(3),
        nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
        nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, ncls))


def lda_feats(X):
    """Hand-made features for the LDA baseline: envelope in 3 time slices, ch1/ch2 ratio, waveform length."""
    y, e = X[:, :2], X[:, 2:]
    segs = np.array_split(np.arange(X.shape[2]), 3)
    f = [np.log1p(e[:, :, s].mean(2)) for s in segs]
    tot = e.mean(2)
    f.append(np.log((tot[:, :1] + 1e-3) / (tot[:, 1:] + 1e-3)))
    f.append(np.log1p(np.abs(np.diff(y, axis=2)).mean(2)))
    return np.concatenate(f, 1)


class ClickModel:
    def __init__(self, path, kind="cnn"):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = ck["cfg"]
        self.kind = kind
        if kind == "cnn":
            self.torch = torch
            self.net = make_net()
            self.net.load_state_dict(ck["cnn"])
            self.net.eval()
        else:
            self.lda = ck["lda"]

    def predict(self, X):
        if self.kind == "cnn":
            with self.torch.no_grad():
                p = self.torch.softmax(self.net(self.torch.from_numpy(X[None])), 1)[0].numpy()
        else:
            p = self.lda.predict_proba(lda_feats(X[None]))[0]
        return p


# ----------------------------------------------------------------------------- training
def load_session(path, thr, rng):
    d = np.load(path)
    raw, ev = d["raw"].astype(np.float64), d["events"]
    cal = [k for k in range(len(ev) - 1) if ev[k][1] == EV_CAL]
    if not cal:
        raise SystemExit(f"{path}: no rest-calibration period found")
    s0, s1 = ev[cal[0]][0] + FS, ev[cal[0] + 1][0]
    yf = Preproc()(raw)
    env, _ = moving_rms(yf)
    std, d_rest = rest_stats(yf[s0:s1], env[s0:s1])
    dd = combined(env, std)
    th = thr * d_rest
    L = WIN + 2 * JIT
    X, y, rest, cues, found = [], [], [], 0, 0
    for k in range(len(ev) - 1):
        s, c = int(ev[k][0]), int(ev[k][1])
        e = int(ev[k + 1][0])
        if c in (EV_LEFT, EV_RIGHT):
            cues += 1
            on = first_crossing(dd, th, s + 150, e)
            if on is not None and on - PRE - JIT >= 0 and on + POST + JIT <= len(raw):
                a = on - PRE - JIT
                X.append(make_window(yf[a:a + L], env[a:a + L], std))
                y.append(c)
                found += 1
        elif c in (EV_RELAX, EV_CAL):
            lo, hi = s + (FS if c == EV_CAL else 300), e - L
            if hi > lo:
                rest += list(rng.integers(lo, hi, 6 if c == EV_CAL else 1))
    nmax = max(y.count(EV_LEFT), y.count(EV_RIGHT), 1)
    for a in rng.permutation(rest)[:nmax]:
        X.append(make_window(yf[a:a + L], env[a:a + L], std))
        y.append(0)
    rate = found / max(cues, 1)
    print(f"  {os.path.basename(path)}: {found}/{cues} cued taps detected ({rate:.0%}), "
          f"{y.count(0)} rest windows")
    if rate < 0.8:
        print("    ! many taps missed - tap firmer, or train with a lower --thr (e.g. 2.5)")
    return X, y


def augment(Xb, rng):
    n = len(Xb)
    off = rng.integers(0, 2 * JIT + 1, n)
    out = np.stack([Xb[i, :, o:o + WIN] for i, o in enumerate(off)]).copy()
    sc = rng.uniform(0.75, 1.25, (n, 2))[:, :, None]
    out[:, :2] *= sc
    out[:, 2:] *= sc
    out[:, :2] += rng.normal(0, 0.15, out[:, :2].shape)
    return out.astype(np.float32)


def cmd_train(args):
    import torch
    import torch.nn as nn
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_score, train_test_split

    files = sorted({f for pat in args.files for f in glob.glob(pat)})
    if not files:
        sys.exit("No recordings found.")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    print("Loading recordings:")
    X, y = [], []
    for f in files:
        a, b = load_session(f, args.thr, rng)
        X += a
        y += b
    X, y = np.stack(X), np.array(y)
    counts = [int((y == k).sum()) for k in range(3)]
    print(f"Dataset: rest {counts[0]}, left {counts[1]}, right {counts[2]}")
    if min(counts) < 8:
        sys.exit("Too few examples per class - record more taps.")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    centre = lambda A: np.ascontiguousarray(A[:, :, JIT:JIT + WIN])
    Xva_c = torch.from_numpy(centre(Xva))
    yva_t = torch.from_numpy(yva).long()

    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    best, best_state = (-1, 0), None
    print(f"Training 1D CNN for {args.epochs} epochs ...")
    for ep in range(args.epochs):
        net.train()
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            if len(idx) < 2:
                continue
            xb = torch.from_numpy(augment(Xtr[idx], rng))
            loss = lossf(net(xb), torch.from_numpy(ytr[idx]).long())
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            out = net(Xva_c)
            acc = (out.argmax(1) == yva_t).float().mean().item()
            vl = lossf(out, yva_t).item()
        if (acc, -vl) > best:
            best = (acc, -vl)
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep + 1:3d}  val acc {acc:.0%}")
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(Xva_c).argmax(1).numpy()
    cnn_acc = float((pred == yva).mean())

    lda = LinearDiscriminantAnalysis()
    Fall = lda_feats(centre(X))
    lda.fit(lda_feats(centre(Xtr)), ytr)
    lda_acc = float(lda.score(lda_feats(centre(Xva)), yva))
    lda_cv = float(cross_val_score(LinearDiscriminantAnalysis(), Fall, y, cv=5).mean())
    lda.fit(Fall, y)

    torch.save({"cnn": best_state, "lda": lda,
                "cfg": {"thr": args.thr, "fs": FS, "pre": PRE, "post": POST}}, args.out)

    print("\nResults on held-out taps:")
    print(f"  1D CNN : {cnn_acc:.0%}")
    print(f"  LDA    : {lda_acc:.0%}   (5-fold CV {lda_cv:.0%})")
    print("Confusion (rows = true rest/left/right, cols = predicted):")
    print(confusion_matrix(yva, pred, labels=[0, 1, 2]))
    if lda_acc > cnn_acc + 0.02:
        print("-> LDA did better on this data; try:  python emg_click.py run --kind lda ...")
    rep = os.path.splitext(args.out)[0] + "_report.png"
    save_report(centre(X), y, confusion_matrix(yva, pred, labels=[0, 1, 2]), cnn_acc, lda_acc, rep)
    print(f"\nSaved model -> {args.out}\nSaved report -> {rep}")


def save_report(X, y, cm, cnn_acc, lda_acc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(WIN) - PRE
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    cols = {0: "#888888", 1: COL1, 2: COL2}
    for ch in range(2):
        for k in range(3):
            e = X[y == k, 2 + ch]
            m, s = e.mean(0), e.std(0)
            ax[ch].plot(t, m, color=cols[k], label=CLASSES[k])
            ax[ch].fill_between(t, m - s, m + s, color=cols[k], alpha=0.15)
        ax[ch].axvline(0, color="k", lw=0.8, ls=":")
        ax[ch].set_title(f"Channel {ch + 1} envelope (mean ± sd)")
        ax[ch].set_xlabel("ms from tap onset")
        ax[ch].set_ylabel("x rest noise")
        ax[ch].legend()
    ax[2].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax[2].text(j, i, cm[i, j], ha="center", va="center")
    ax[2].set_xticks(range(3), CLASSES)
    ax[2].set_yticks(range(3), CLASSES)
    ax[2].set_xlabel("predicted")
    ax[2].set_ylabel("true")
    ax[2].set_title(f"CNN held-out: {cnn_acc:.0%}  (LDA {lda_acc:.0%})")
    fig.suptitle("Left (index) vs right (middle): the blue and orange curves should differ "
                 "between channel 1 and channel 2 - if they look the same, move the pads")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


# ----------------------------------------------------------------------------- recorder
class Recorder:
    def __init__(self, trials, src, outdir):
        order = [EV_LEFT] * trials + [EV_RIGHT] * trials
        random.shuffle(order)
        self.phases = [(EV_CAL, 10.0)]
        for c in order:
            self.phases += [(EV_RELAX, 1.5 + random.random() * 0.6), (c, 1.6)]
        self.phases.append((EV_END, 0))
        self.ntrials = len(order)
        self.src, self.outdir = src, outdir
        self.i = -1
        self.active = self.done = False
        self.chunks, self.events = [], []
        self.path = None
        if hasattr(src, "auto"):
            src.auto = False                   # simulator: only tap when cued

    def start(self, n_abs, t):
        self.active = True
        self.start_abs = n_abs
        self._next(n_abs, t)

    def _next(self, n_abs, t):
        self.i += 1
        code, dur = self.phases[self.i]
        self.events.append((n_abs - self.start_abs, code))
        self.t_end = t + dur
        if code in (EV_LEFT, EV_RIGHT) and hasattr(self.src, "cue_event"):
            self.src.cue_event(code)
        if code == EV_END:
            self.save()

    def tick(self, n_abs, t):
        if self.active and not self.done and t >= self.t_end:
            self._next(n_abs, t)

    @property
    def phase(self):
        return self.phases[self.i][0] if self.i >= 0 else None

    @property
    def trial(self):
        return sum(1 for c, _ in self.phases[:self.i + 1] if c in (EV_LEFT, EV_RIGHT))

    def save(self, n_abs=None):
        if self.done or not self.chunks:
            return
        if self.events[-1][1] != EV_END:
            self.events.append((n_abs - self.start_abs, EV_END))
        self.done = True
        self.active = False
        os.makedirs(self.outdir, exist_ok=True)
        self.path = os.path.join(self.outdir, time.strftime("session_%Y%m%d_%H%M%S.npz"))
        np.savez_compressed(self.path, raw=np.concatenate(self.chunks).astype(np.uint16),
                            events=np.array(self.events, dtype=np.int64), fs=FS)
        print(f"Saved {self.path}")


# ----------------------------------------------------------------------------- GUI
def run_gui(mode, args):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    src = open_source(args.port, args.baud)
    model = ClickModel(args.model, args.kind) if mode == "run" else None
    thr0 = args.thr if args.thr else (model.cfg["thr"] if model else 3.0)
    mouse = None
    if mode == "run" and args.click:
        from pynput.mouse import Button, Controller
        mouse = (Controller(), Button)

    app = pg.mkQApp("EMG click")
    pg.setConfigOptions(background="#111513", foreground="#c9d1cb", antialias=False)

    class Win(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"EMG click - {mode}")
            self.resize(1280, 760)
            self.setStyleSheet("QWidget{background:#111513;color:#e5ebe6;font-size:14px}"
                               "QPushButton{background:#232b27;border:1px solid #3a463f;padding:8px;border-radius:6px}"
                               "QProgressBar{border:1px solid #3a463f;border-radius:4px;text-align:center;height:18px}"
                               "QProgressBar::chunk{background:#5cc27a}")
            self.pipe = LivePipe()
            self.std = np.ones(2)
            self.d_mean = None
            self.cal = None
            self.marks = []
            self.pending = []
            self.scan = 0
            self.last_on = -10 ** 9
            self.rate_hist = []
            self.rec = Recorder(args.trials, src, args.out) if mode == "record" else None

            h = QtWidgets.QHBoxLayout(self)
            glw = pg.GraphicsLayoutWidget()
            h.addWidget(glw, 3)
            side = QtWidgets.QVBoxLayout()
            h.addLayout(side, 1)

            def mkplot(row, title):
                p = glw.addPlot(row=row, col=0, title=title)
                p.showGrid(x=True, y=True, alpha=0.15)
                p.setDownsampling(auto=True, mode="peak")
                p.setClipToView(True)
                return p
            p1 = mkplot(0, "Channel 1 · D32  (index → LEFT)")
            p2 = mkplot(1, "Channel 2 · D33  (middle → RIGHT)")
            p3 = mkplot(2, "Envelope (25 ms RMS) · dashed = tap threshold")
            p2.setXLink(p1)
            p3.setXLink(p1)
            p3.setLabel("bottom", "seconds")
            self.c1 = p1.plot(pen=pg.mkPen(COL1, width=1))
            self.c2 = p2.plot(pen=pg.mkPen(COL2, width=1))
            self.e1 = p3.plot(pen=pg.mkPen(COL1, width=2), name="ch1")
            self.e2 = p3.plot(pen=pg.mkPen(COL2, width=2), name="ch2")
            self.dline = p3.plot(pen=pg.mkPen(COLD, width=1))
            self.thr_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#e5ebe6", style=QtCore.Qt.PenStyle.DashLine))
            p3.addItem(self.thr_line)
            self.scatter = pg.ScatterPlotItem(size=14)
            p3.addItem(self.scatter)
            self.plots = (p1, p2, p3)

            self.big = QtWidgets.QLabel("…")
            self.big.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.big.setWordWrap(True)
            self.big.setMinimumHeight(170)
            self.big.setStyleSheet("font-size:30px;font-weight:700;border-radius:10px;background:#1b221e")
            side.addWidget(self.big)
            self.sub = QtWidgets.QLabel("")
            self.sub.setWordWrap(True)
            side.addWidget(self.sub)

            if mode == "run":
                self.bars = []
                for k, name in enumerate(CLASSES):
                    lab = QtWidgets.QLabel(name)
                    bar = QtWidgets.QProgressBar()
                    bar.setRange(0, 100)
                    side.addWidget(lab)
                    side.addWidget(bar)
                    self.bars.append(bar)
                self.hist = QtWidgets.QListWidget()
                self.hist.setMaximumHeight(170)
                side.addWidget(self.hist)
                self.clickbox = QtWidgets.QCheckBox("Send real mouse clicks")
                self.clickbox.setEnabled(mouse is not None)
                if mouse is None:
                    self.clickbox.setToolTip("start with --click to enable")
                side.addWidget(self.clickbox)

            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("Threshold ×rest"))
            self.thr = QtWidgets.QDoubleSpinBox()
            self.thr.setRange(1.5, 15)
            self.thr.setSingleStep(0.5)
            self.thr.setValue(thr0)
            row.addWidget(self.thr)
            side.addLayout(row)
            b = QtWidgets.QPushButton("Recalibrate rest (relax 3 s)")
            b.clicked.connect(self.start_cal)
            side.addWidget(b)
            if self.rec:
                self.recbtn = QtWidgets.QPushButton("Start recording  (Space)")
                self.recbtn.clicked.connect(self.start_rec)
                side.addWidget(self.recbtn)
            side.addStretch(1)
            self.status = QtWidgets.QLabel("")
            self.status.setStyleSheet("color:#9ba69f;font-size:12px")
            side.addWidget(self.status)

            self.start_cal()
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.tick)
            self.timer.start(33)

        # --- actions
        def start_cal(self):
            self.cal = (self.pipe.n + 500, self.pipe.n + 3500)

        def start_rec(self):
            if self.rec and not self.rec.active and not self.rec.done:
                self.rec.start(self.pipe.n, time.monotonic())
                self.recbtn.setEnabled(False)

        def keyPressEvent(self, e):
            if e.key() == QtCore.Qt.Key.Key_Space:
                self.start_rec()

        def closeEvent(self, e):
            if self.rec and self.rec.active:
                self.rec.save(self.pipe.n)
            src.close()
            e.accept()

        def setbig(self, text, bg="#1b221e", fg="#e5ebe6", size=30):
            self.big.setText(text)
            self.big.setStyleSheet(f"font-size:{size}px;font-weight:700;border-radius:10px;"
                                   f"background:{bg};color:{fg}")

        # --- main loop
        def tick(self):
            raw = src.read()
            self.pipe.push(raw)
            if self.rec and self.rec.active and len(raw):
                self.rec.chunks.append(raw)
            n, now = self.pipe.n, time.monotonic()
            self.rate_hist = [(t, k) for t, k in self.rate_hist if now - t < 2] + [(now, n)]

            calibrating = self.cal is not None
            if calibrating and n >= self.cal[1]:
                yf, env, _ = self.pipe.get(*self.cal)
                if len(yf) > 500:
                    self.std, self.d_mean = rest_stats(yf, env)
                self.cal = None
                self.scan = n
                calibrating = False

            if mode == "run" and not calibrating and self.d_mean:
                self.detect(n)
            if self.rec:
                self.rec.tick(n, now)

            self.update_text(calibrating, now)
            self.update_plots(n)

        def detect(self, n):
            thr = self.thr.value() * self.d_mean
            if n > self.scan + 1:
                _, env, a = self.pipe.get(self.scan - 1, n)
                d = combined(env, self.std)
                above = d >= thr
                for i in np.flatnonzero(above[1:] & ~above[:-1]) + 1:
                    on = a + int(i)
                    if on - self.last_on >= REFR:
                        self.pending.append(on)
                        self.last_on = on
                self.scan = n
            for on in list(self.pending):
                if n >= on + POST:
                    self.pending.remove(on)
                    yf, env, a = self.pipe.get(on - PRE, on + POST)
                    if len(yf) == WIN:
                        self.classify(on, make_window(yf, env, self.std))

        def classify(self, on, X):
            p = model.predict(X)
            k = int(np.argmax(p))
            for i, bar in enumerate(self.bars):
                bar.setValue(int(round(p[i] * 100)))
            ok = k != 0 and p[k] >= args.conf
            self.marks.append((on / FS, k if ok else 0))
            if ok:
                name = "LEFT CLICK" if k == 1 else "RIGHT CLICK"
                self.setbig(f"{name}\n{p[k]:.0%}", bg=COL1 if k == 1 else COL2, fg="#111513", size=34)
                self.flash_until = time.monotonic() + 0.6
                self.hist.insertItem(0, f"{time.strftime('%H:%M:%S')}  {name}  {p[k]:.0%}")
                if mouse and self.clickbox.isChecked():
                    mouse[0].click(mouse[1].left if k == 1 else mouse[1].right)
            else:
                self.hist.insertItem(0, f"{time.strftime('%H:%M:%S')}  ignored "
                                        f"({CLASSES[k]} {p[k]:.0%})")
            while self.hist.count() > 12:
                self.hist.takeItem(12)

        def update_text(self, calibrating, now):
            rate = 0
            if len(self.rate_hist) > 1:
                (t0, n0), (t1, n1) = self.rate_hist[0], self.rate_hist[-1]
                rate = (n1 - n0) / max(t1 - t0, 1e-6)
            self.status.setText(f"{rate:5.0f} samples/s · dropped {src.dropped} · bad {src.bad}"
                                + ("" if rate > 900 or rate == 0 else "  ← should be ~1000"))
            if calibrating:
                self.setbig("RELAX\ncalibrating…", size=26)
                return
            if self.rec:
                ph = self.rec.phase
                if self.rec.done:
                    self.setbig("Done ✓", bg="#1f3a28", size=30)
                    self.sub.setText(f"Saved {self.rec.path}\nClose the window, then:\n"
                                     f"python emg_click.py train {self.rec.outdir}/*.npz")
                elif ph is None:
                    self.setbig("Press SPACE\nto start", size=28)
                    self.sub.setText(f"{self.rec.ntrials} cues (half index, half middle). When a cue shows, "
                                     "tap that finger ONCE, firmly, like a real click. Keep the wrist and "
                                     "other fingers still. Starts with 10 s of rest.")
                elif ph == EV_CAL:
                    self.setbig("RELAX\nhand still", size=28)
                    self.sub.setText(f"measuring rest noise… {max(0, self.rec.t_end - now):.0f} s")
                elif ph == EV_RELAX:
                    self.setbig("relax", size=26)
                    self.sub.setText(f"trial {self.rec.trial}/{self.rec.ntrials}")
                elif ph == EV_LEFT:
                    self.setbig("TAP INDEX\n← LEFT", bg=COL1, fg="#111513", size=32)
                elif ph == EV_RIGHT:
                    self.setbig("TAP MIDDLE\nRIGHT →", bg=COL2, fg="#111513", size=32)
                return
            if mode == "view":
                _, env, _ = self.pipe.get(self.pipe.n - FS, self.pipe.n)
                if len(env):
                    m = (env / self.std).max(0)
                    lead = "ch1" if m[0] > m[1] * 1.3 else "ch2" if m[1] > m[0] * 1.3 else "even"
                    self.setbig(f"peak last 1 s\nch1 {m[0]:.0f}   ch2 {m[1]:.0f}", size=24)
                    self.sub.setText(f"ratio ch1/ch2 = {m[0] / max(m[1], 1e-6):.1f}  ({lead})\n\n"
                                     "Tap index a few times, then middle. You want index → ch1 clearly "
                                     "bigger, middle → ch2 clearly bigger. Move pads until that's true.")
            elif mode == "run":
                if now > getattr(self, "flash_until", 0):
                    self.setbig("ready", size=26)
                self.sub.setText(f"conf ≥ {args.conf:.0%} to count · model: {args.kind.upper()}")

        def update_plots(self, n):
            yf, env, a = self.pipe.get(n - 3 * FS, n)
            if len(yf) < 2:
                return
            t = np.arange(a, a + len(yf)) / FS
            yn, en = yf / self.std, env / self.std
            self.c1.setData(t, yn[:, 0])
            self.c2.setData(t, yn[:, 1])
            self.e1.setData(t, en[:, 0])
            self.e2.setData(t, en[:, 1])
            self.dline.setData(t, np.hypot(en[:, 0], en[:, 1]))
            if self.d_mean:
                self.thr_line.setValue(self.thr.value() * self.d_mean)
            unit = "× rest" if self.d_mean else "ADC counts"
            for p in self.plots:
                p.setLabel("left", unit)
            self.marks = [m for m in self.marks if m[0] > t[0]]
            cols = {0: "#888888", 1: COL1, 2: COL2}
            ymax = float(np.hypot(en[:, 0], en[:, 1]).max())
            self.scatter.setData([{"pos": (tm, ymax * 1.05), "brush": cols[k], "pen": None}
                                  for tm, k in self.marks])

    w = Win()
    w.show()
    if getattr(args, "screenshot", None):          # used for automated testing
        def snap():
            w.grab().save(args.screenshot)
            app.quit()
        QtCore.QTimer.singleShot(int(args.seconds * 1000), snap)
    pg.exec()


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("ports", help="list serial ports")

    def live(name, hlp):
        p = sp.add_parser(name, help=hlp)
        p.add_argument("--port", help="e.g. COM5, /dev/ttyUSB0, or sim")
        p.add_argument("--baud", type=int, default=921600)
        p.add_argument("--thr", type=float, default=None, help="tap threshold, × resting envelope")
        p.add_argument("--screenshot", help=argparse.SUPPRESS)
        p.add_argument("--seconds", type=float, default=8, help=argparse.SUPPRESS)
        return p
    live("view", "live graphs")
    r = live("record", "guided recording")
    r.add_argument("--trials", type=int, default=30, help="taps per finger")
    r.add_argument("--out", default="recordings")
    r = live("run", "live classification")
    r.add_argument("--model", default="model.pt")
    r.add_argument("--kind", choices=["cnn", "lda"], default="cnn")
    r.add_argument("--conf", type=float, default=0.7, help="min confidence to count a click")
    r.add_argument("--click", action="store_true", help="allow sending real mouse clicks")
    t = sp.add_parser("train", help="train on recordings")
    t.add_argument("files", nargs="+")
    t.add_argument("--out", default="model.pt")
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--thr", type=float, default=3.0)
    args = ap.parse_args()

    if args.cmd == "ports":
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(f"{p.device:20s} {p.description}")
    elif args.cmd == "train":
        cmd_train(args)
    else:
        if args.cmd != "record":
            args.trials, args.out = 0, None
        run_gui(args.cmd, args)


if __name__ == "__main__":
    main()
