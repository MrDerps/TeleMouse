"""Plot example trials (raw signals, features, activity score, localised label) for visual sanity.

    python -m backend.data.plot_trial                         # newest session, one trial per gesture
    python -m backend.data.plot_trial --session <id> --out backend/_out/trials.png
"""
from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from backend.common.config import load_config, repo_path  # noqa: E402
from backend.data.dataset import gesture_catalog, build_session  # noqa: E402
from backend.data.features import EmgFilter, LAYOUT  # noqa: E402
from backend.data.session import load_session, resolve_sessions, load_truth  # noqa: E402


def plot_trials(sess, d, classes, out, per_gesture=1, pad=1.0):
    picks = []
    for g in classes[1:]:
        picks += [t for t in d.trials if t["label"] == g][:per_gesture]
    if not picks:
        raise ValueError("no trials to plot")
    truth = load_truth(sess.path)
    filt = EmgFilter({"emg": {**_cfg["emg"]}})
    emg_f = filt(sess.emg.astype(np.float64)) / d.rest_std
    fig, axes = plt.subplots(6, len(picks), figsize=(4.2 * len(picks), 13), squeeze=False, sharex="col")
    for c, tr in enumerate(picks):
        a, b = tr["cue"][0] - pad, tr["cue"][1] + pad
        me = (sess.emg_t >= a) & (sess.emg_t < b)
        mi = (sess.imu_t >= a) & (sess.imu_t < b)
        mh = (sess.hand_t >= a) & (sess.hand_t < b)
        mf = (d.T >= a) & (d.T < b)
        t0 = tr["cue"][0]
        ax = axes[:, c]
        ax[0].plot(sess.emg_t[me] - t0, emg_f[me, 0], lw=0.4, label="ch1")
        ax[0].plot(sess.emg_t[me] - t0, emg_f[me, 1], lw=0.4, label="ch2", alpha=0.7)
        ax[0].set_title(f"{tr['label']}  ({tr['kind']}{', FALLBACK' if tr['fallback'] else ''})")
        ax[0].set_ylabel("EMG filt (x rest)")
        ax[1].plot(d.T[mf] - t0, d.X[mf, LAYOUT["emg"]][:, 0], label="log rms1")
        ax[1].plot(d.T[mf] - t0, d.X[mf, LAYOUT["emg"]][:, 1], label="log rms2")
        ax[1].set_ylabel("EMG feat")
        ax[2].plot(sess.imu_t[mi] - t0, sess.imu[mi, 3:], lw=0.7)
        ax[2].set_ylabel("gyro deg/s")
        ax[3].plot(sess.imu_t[mi] - t0, sess.imu[mi, :3], lw=0.7)
        ax[3].set_ylabel("accel g")
        ax[4].plot(sess.hand_t[mh] - t0, sess.hand[mh, 0, 0], ".", ms=2, label="wrist x")
        ax[4].plot(sess.hand_t[mh] - t0, sess.hand[mh, 0, 1], ".", ms=2, label="wrist y")
        tip = np.linalg.norm(sess.hand[mh, 4, :2] - sess.hand[mh, 8, :2], axis=1)
        ax[4].plot(sess.hand_t[mh] - t0, tip * 3, ".", ms=2, label="thumb-index x3")
        ax[4].set_ylabel("hand (image)")
        ax[5].plot(d.T[mf] - t0, d.score[mf], "k", lw=0.8, label="activity")
        yl = d.y[mf]
        ax[5].fill_between(d.T[mf] - t0, 0, np.where(yl > 0, d.score[mf].max(), 0), color="tab:green", alpha=0.3,
                           step="mid", label="labelled gesture")
        ax[5].fill_between(d.T[mf] - t0, 0, np.where(yl < 0, d.score[mf].max(), 0), color="tab:gray", alpha=0.3,
                           step="mid", label="ignore")
        ax[5].set_ylabel("activity z")
        ax[5].set_xlabel("s from cue start")
        for x in ax:
            x.axvspan(0, tr["cue"][1] - t0, color="tab:orange", alpha=0.08)
            for g in truth:
                if g["t_start"] >= a and g["t_end"] <= b:
                    x.axvline(g["t_start"] - t0, color="r", ls=":", lw=0.8)
                    x.axvline(g["t_end"] - t0, color="r", ls=":", lw=0.8)
        if c == 0:
            for x in ax:
                x.legend(fontsize=6, loc="upper left")
    fig.suptitle(f"{sess.id}: orange = cue window, green = localised label, red dotted = synth truth")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


_cfg = None


def main(argv=None):
    global _cfg
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _cfg = cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None, help="session id or path (default: newest)")
    ap.add_argument("--out", default="backend/_out/trials.png")
    ap.add_argument("--per-gesture", type=int, default=1)
    a = ap.parse_args(argv)
    roots = [repo_path(cfg["paths"]["sessions"]), repo_path(cfg["paths"]["synth"])]
    paths = resolve_sessions([a.session] if a.session else "all", roots)
    sess = load_session(paths[-1])
    cat = gesture_catalog(cfg, [sess])
    gl = [g for g in cat if any(r["label"] == g for r in sess.labels)]
    classes = ["none"] + gl
    d = build_session(sess, classes, {g: cat[g].get("kind", "instant") for g in gl}, cfg)
    print("wrote", plot_trials(sess, d, classes, repo_path(a.out), a.per_gesture))


if __name__ == "__main__":
    main()
