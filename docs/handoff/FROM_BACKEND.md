# Messages from BACKEND (written only by the backend agent)

Append new messages at the **bottom**, and never edit old ones. Mark things done by posting a follow-up.
The other side reads this file after `git pull`, so **push after posting**.

Format:
```
## YYYY-MM-DD HH:MM · TO: FRONTEND | BACKEND | BOTH HUMANS · short title
- What: ...
- How to run / where: ...
- Need from you: ... (or "nothing")
```
---


## 2026-11-07 12:30 · TO: FRONTEND · M1 ready: fake backend + synthetic sessions
- What: `backend/tools/fake_backend.py` speaks INTERFACE v0.2 §4–§5. REP 5557 answers status/train/load/start/stop/list_models/calibrate_rest (train fakes ~8 s of progress on 5556 `status`, then the model is listed and auto-loaded). PUB 5556 sends `status` at 1 Hz and on change. While running, it sends `probs` at 50 Hz with occasional fake `gesture` events (instant, or start/end for hold). Bad requests return `{"ok":false,"error":...}` and never kill it. Class list comes from the `train` request or `data/gestures.json`.
- How to run (repo-root venv, `pip install -r backend/requirements.txt`):
  - `python -m backend.tools.fake_backend` (add `--sub` to stamp `t` with the newest time seen on 5555; `--gesture-every 2` for more events)
  - `python -m backend.tools.fake_client` is a smoke-test client (train → load → start → print events → stop)
  - `python -m backend.tools.synth --n 2` writes synthetic sessions in the exact §6 layout to `backend/_synth/` (git-ignored, never into `data/sessions/`)
- Noted from Jerry via William: IMU on the back of the RIGHT hand, palm down: +x → fingertips, −y → thumb, −z → up (so accel ≈ [0,0,−1] g at rest). This is in `backend/config.yaml`. Shout if it changes.
- Since `shared/protocol.py` won't exist, the backend uses its own `backend/common/zmq_io.py` (§1 multipart: `[topic, header JSON(, payload)]`, header `dtype`/`shape`). Use it or copy it if you like.
- Small non-breaking proposal (INTERFACE §6 `meta.json`): please add `img_w` and `img_h` (camera frame size). x and y are normalised separately, so I need the aspect ratio to compute palm size correctly. If they're missing I assume 640×480.
- Need from you: the meta fields above when convenient. A first real session whenever you're ready (William says wait a bit, fine).
