"""Webcam + MediaPipe Hand Landmarker (Tasks API, VIDEO mode).

Two threads so that slow inference never delays timestamps or lets frames pile up:
  * capture: grabs frames as fast as the camera delivers them and stamps each one with
    time.perf_counter() the moment it arrives. Keeps only the newest frame.
  * up to `num_hands` hands per frame (default 2: the right hand gestures, the left hand points).
  * inference (N workers, default 2, each with its own landmarker): a free worker takes the
    newest frame. Two workers keep up at 30 fps even when Windows throttles the CPU on battery
    (one inference is 13-40 ms there). Results older than the last published one are dropped,
    so output stays in time order. Each worker calls
        on_hands(t_raw, hands, img_w, img_h)   hands = [{"lm": (21,3), "handedness": "Left"/"Right", "score"}]
    t_raw is the arrival stamp; the caller subtracts the calibrated camera latency.

Landmarks are in the raw (un-mirrored) camera image. MediaPipe labels handedness as if
the image were mirrored (selfie view), so the label is swapped here to give the user's
actual hand.
"""
from __future__ import annotations

import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "hand_landmarker.task"


def ensure_model(log=print) -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    log(f"[camera] downloading the MediaPipe hand model (~8 MB) to {MODEL_PATH} ...")
    tmp = MODEL_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL_PATH)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the hand model ({e}). If you're offline, copy hand_landmarker.task "
            f"from another machine to {MODEL_PATH} (source: {MODEL_URL}).") from e
    return MODEL_PATH


class Camera:
    def __init__(self, index=0, width=640, height=480, fps=30, on_hands=None, log=print, keep_frames=True,
                 workers=2, num_hands=2):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.on_hands = on_hands or (lambda *a: None)
        self.num_hands = num_hands
        self.log = log
        self.keep_frames = keep_frames        # keep the newest frame + landmarks for a preview
        self.workers = workers
        self.landmarkers = []
        self._pub_lock = threading.Lock()
        self._last_pub_t = -1.0
        self.frames_out_of_order = 0
        self.cap = None
        self._cond = threading.Condition()
        self._frame = None                    # (t_raw, frame)
        self._seq = 0
        self._stop = threading.Event()
        self._threads = []
        self.frames_captured = 0
        self.frames_processed = 0
        self.frames_skipped = 0               # captured but replaced before inference got to them
        self.last_frame_t = 0.0
        self.infer_ms = 0.0
        self.preview = None                   # (frame BGR, hands list) for UIs
        self.img_w, self.img_h = width, height

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model = ensure_model(self.log)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO, num_hands=self.num_hands,
            min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4, min_tracking_confidence=0.4)
        self.landmarkers = [vision.HandLandmarker.create_from_options(opts) for _ in range(self.workers)]
        self._mp = mp

        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self._threads = [threading.Thread(target=self._capture, name="cam-capture", daemon=True)]
        self._threads += [threading.Thread(target=self._infer, args=(lmk,), name=f"cam-infer{i}", daemon=True)
                          for i, lmk in enumerate(self.landmarkers)]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        for t in self._threads:
            t.join(2)
        if self.cap is not None:
            self.cap.release()
        for lmk in self.landmarkers:
            lmk.close()

    # ------------------------------------------------------------------ threads
    def _capture(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                if self._frame is not None:
                    self.frames_skipped += 1
                self._frame = (t, frame)
                self._seq += 1
                self.frames_captured += 1
                self.last_frame_t = t
                self._cond.notify()

    def _infer(self, landmarker):
        import cv2
        last_ms = -1
        while not self._stop.is_set():
            with self._cond:
                while self._frame is None and not self._stop.is_set():
                    self._cond.wait(0.2)
                if self._stop.is_set():
                    break
                t_raw, frame = self._frame
                self._frame = None
            h, w = frame.shape[:2]
            self.img_w, self.img_h = w, h
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(int(t_raw * 1000), last_ms + 1)          # VIDEO mode needs increasing ms
            last_ms = ts
            t0 = time.perf_counter()
            try:
                res = landmarker.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
            except Exception as e:  # never let one bad frame kill the thread
                self.log(f"[camera] landmarker error: {e}")
                continue
            self.infer_ms = 0.9 * self.infer_ms + 0.1 * (1000 * (time.perf_counter() - t0))
            hands = []
            for i, pts in enumerate(res.hand_landmarks or []):
                cat = res.handedness[i][0]
                hands.append({"lm": np.array([[p.x, p.y, p.z] for p in pts], np.float32),
                              "handedness": {"Left": "Right", "Right": "Left"}.get(cat.category_name),  # un-mirror
                              "score": float(cat.score)})
            with self._pub_lock:                              # keep output in time order
                if t_raw <= self._last_pub_t:
                    self.frames_out_of_order += 1
                    continue
                self._last_pub_t = t_raw
                self.frames_processed += 1
                if self.keep_frames:
                    self.preview = (frame, hands)
                self.on_hands(t_raw, hands, w, h)
