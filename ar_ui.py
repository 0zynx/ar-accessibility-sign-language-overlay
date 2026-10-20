"""AR overlay pieces: One Euro smoothing, speech bubble that follows the hand, sentence bar,
background text-to-speech, and a low-latency camera reader."""
import math
import queue
import threading
import time

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
STRIP_H = 56


class OneEuro:
    """Low jitter when still, low lag when moving. Works on scalars or vectors."""

    def __init__(self, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t_prev = self.x_prev = self.dx_prev = None

    @staticmethod
    def _alpha(dt, cutoff):
        return 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x, np.zeros_like(x)
            return x
        dt = max(t - self.t_prev, 1e-4)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        a = self._alpha(dt, self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat)))
        x_hat = a * x + (1 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return x_hat


class LatestFrame:
    """Reads the camera on a thread and keeps only the newest frame (no lag on phone streams)."""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame, self.ok, self.lock, self.stop = None, False, threading.Lock(), False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            ok, f = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, f
            if not ok:
                time.sleep(0.05)

    def read(self):
        with self.lock:
            return self.ok, (None if self.frame is None else self.frame.copy())

    def release(self):
        self.stop = True
        time.sleep(0.1)
        self.cap.release()


class Speaker:
    """Text-to-speech on a background thread so video never stalls."""

    def __init__(self, enabled=True):
        self.q, self.enabled = queue.Queue(), enabled
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
        except Exception as e:
            print(f"[voice disabled] {e}")
            return
        while True:
            text = self.q.get()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass

    def say(self, text):
        if self.enabled and self.q.qsize() < 3:
            self.q.put(text.lower())


# ------------------------------------------------------------------ drawing
def ease_out_back(t):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def rrect(img, x, y, w, h, r, color):
    x, y, w, h = int(x), int(y), int(w), int(h)
    r = int(max(1, min(r, w // 2, h // 2)))
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1, cv2.LINE_AA)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def conf_color(c):  # BGR
    return (96, 210, 90) if c >= 0.8 else (40, 190, 245) if c >= 0.6 else (70, 80, 240)


def measure_bubble(text, cstr, s, max_w):
    """Bubble geometry. Long phrases shrink their font to fit max_w."""
    r, pad = 26 * s, 12 * s
    scale, th = 0.9 * s, max(1, int(round(2 * s)))
    (tw, thh), _ = cv2.getTextSize(text, FONT, scale, th)
    while tw > max_w and scale > 0.35:
        scale *= 0.92
        (tw, thh), _ = cv2.getTextSize(text, FONT, scale, th)
    cscale = 0.55 * s
    (cw, _), _ = cv2.getTextSize(cstr, FONT, cscale, 1)
    return dict(r=r, pad=pad, scale=scale, th=th, tw=tw, thh=thh, cscale=cscale, cw=cw,
                bw=pad + 2 * r + pad + max(tw, cw) + pad, bh=2 * r + 2 * pad)


def conf_text(text, conf):
    return "watching" if text == "..." else f"{int(conf * 100)}%"


def draw_bubble(frame, text, conf, center, side, tip, s, alpha, progress, max_w):
    """Speech bubble: sign icon, confidence ring + %, countdown bar, tail pointing at the hand."""
    if alpha < 0.02 or s < 0.05:
        return
    m = measure_bubble(text, conf_text(text, conf), s, max_w)
    r, pad, bw, bh = m["r"], m["pad"], m["bw"], m["bh"]
    ov = frame.copy()
    bx, by = center[0] - bw / 2, center[1] - bh / 2
    ex = bx + bw if side < 0 else bx                      # bubble edge facing the hand
    tail = np.array([[ex, center[1] - 9 * s], [ex, center[1] + 9 * s], [tip[0], tip[1]]], np.int32)
    cv2.fillConvexPoly(ov, tail, (255, 255, 255), cv2.LINE_AA)
    rrect(ov, bx - 2, by - 2, bw + 4, bh + 4, 20 * s + 2, (230, 230, 230))
    rrect(ov, bx, by, bw, bh, 20 * s, (40, 32, 28))
    ic = (int(bx + pad + r), int(by + bh / 2))
    cv2.circle(ov, ic, int(r), (70, 58, 50), -1, cv2.LINE_AA)
    ch = text[0] if text and text[0].isalnum() else "?"
    (iw, ih), _ = cv2.getTextSize(ch, FONT, 1.0 * s, m["th"])
    cv2.putText(ov, ch, (ic[0] - iw // 2, ic[1] + ih // 2), FONT, 1.0 * s, (255, 255, 255), m["th"], cv2.LINE_AA)
    if text != "...":
        cv2.ellipse(ov, ic, (int(r), int(r)), -90, 0, 360 * conf, conf_color(conf), max(2, int(4 * s)), cv2.LINE_AA)
    tx = int(bx + pad + 2 * r + pad)
    cv2.putText(ov, text, (tx, int(by + pad + m["thh"] + 2 * s)), FONT, m["scale"], (255, 255, 255), m["th"], cv2.LINE_AA)
    cv2.putText(ov, conf_text(text, conf), (tx, int(by + bh - pad)), FONT, m["cscale"], (200, 200, 200), 1, cv2.LINE_AA)
    if progress > 0:
        cv2.rectangle(ov, (int(bx + 14 * s), int(by + bh - 7 * s)),
                      (int(bx + 14 * s + (bw - 28 * s) * min(progress, 1.0)), int(by + bh - 4 * s)),
                      (96, 210, 90), -1, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def place_bubble(bbox, W, H, bw, bh, side, s):
    """Put the bubble beside the hand box (flipping sides near screen edges). Returns (centre, side, tail tip)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    if cx > 0.62 * W:
        side = -1
    elif cx < 0.38 * W:
        side = 1
    gap = 26 * s
    if side == 1 and x2 + gap + bw > W:
        side = -1
    elif side == -1 and x1 - gap - bw < 0:
        side = 1
    tx = (x2 + gap + bw / 2) if side == 1 else (x1 - gap - bw / 2)
    ty = y1 + 0.28 * (y2 - y1)
    tx = float(np.clip(tx, bw / 2 + 6, max(bw / 2 + 6, W - bw / 2 - 6)))
    ty = float(np.clip(ty, bh / 2 + 8, max(bh / 2 + 8, H - STRIP_H - bh / 2 - 8)))
    tip = (x2 - 6 if side == 1 else x1 + 6, y1 + 0.4 * (y2 - y1))
    tip = (float(np.clip(tip[0], 0, W)), float(np.clip(tip[1], 0, H - STRIP_H)))   # keep the tail on-screen
    return np.array([tx, ty]), side, tip


def draw_sentence_strip(frame, words):
    H, W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, H - STRIP_H), (W, H), (30, 24, 20), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    words = list(words)
    text = "  |  ".join(words) if words else "sign a word or phrase..."
    color = (255, 255, 255) if words else (150, 150, 150)
    while cv2.getTextSize(text, FONT, 0.9, 2)[0][0] > W - 30 and len(words) > 1:
        words = words[1:]
        text = "...  |  " + "  |  ".join(words)
    cv2.putText(frame, text, (15, H - 18), FONT, 0.9, color, 2, cv2.LINE_AA)
