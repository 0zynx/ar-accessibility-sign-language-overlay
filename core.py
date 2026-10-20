"""Shared core for SignBridge Words (ISL words + phrases -> English).

Pipeline per clip:   video frames -> pose+hands landmarks -> trim to the signing part
                     -> resample to T frames -> body-relative features -> classifier

Only Tracker needs MediaPipe/OpenCV. Everything else is plain numpy so it is easy to test.
Landmarks are in "square pixel" units (x*W, y*H, z*W) so the aspect ratio does not matter.
"""
import re
import time
import uuid
from collections import deque, namedtuple
from pathlib import Path

import joblib
import numpy as np

T_FRAMES = 24                       # every sign is resampled to this many frames
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]   # nose, L/R shoulder, L/R elbow, L/R wrist
N_PTS = 7 + 21 + 21                 # pose subset + left hand + right hand
LH, RH = slice(7, 28), slice(28, 49)
FRAME_DIM = 9 + 70 + 70             # 149 numbers per frame
FrameData = namedtuple("FrameData", "P flags active")   # P (49,3) | flags [pose, lh, rh] | bool


# ------------------------------------------------------------------ tracker
class Tracker:
    """Pose + hands from one BGR frame. Use the SAME (un-mirrored) frames for training and live use."""

    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.prev_sh = None            # (shoulder centre, shoulder width) from the last good pose

    def reset(self):
        """Forget tracking state (call between unrelated videos)."""
        self.prev_sh = None
        for m in (self.pose, self.hands):
            try:
                m.reset()
            except Exception:
                pass

    def process(self, bgr):
        import cv2
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pr, hr = self.pose.process(rgb), self.hands.process(rgb)
        P = np.zeros((N_PTS, 3), np.float32)
        flags = np.zeros(3, bool)

        if pr.pose_landmarks:
            lm = pr.pose_landmarks.landmark
            P[:7] = [[lm[i].x * W, lm[i].y * H, lm[i].z * W] for i in POSE_IDX]
            if lm[11].visibility > 0.5 and lm[12].visibility > 0.5:
                flags[0] = True
                self.prev_sh = ((P[1] + P[2]) / 2, max(float(np.linalg.norm((P[1] - P[2])[:2])), 1.0))
        sh = self.prev_sh

        hands = []
        if hr.multi_hand_landmarks:
            for h in hr.multi_hand_landmarks[:2]:
                hands.append(np.array([[p.x * W, p.y * H, p.z * W] for p in h.landmark], np.float32))
        # Assign to person-left / person-right slots by position (independent of MediaPipe's
        # handedness label, which flips with mirroring). Facing the camera, the person's right
        # hand appears on the image-left.
        ref_x = sh[0][0] if sh is not None else W / 2
        if len(hands) == 2:
            hands.sort(key=lambda h: h[0, 0])
            P[RH], P[LH] = hands[0], hands[1]
            flags[1] = flags[2] = True
        elif len(hands) == 1:
            if hands[0][0, 0] < ref_x:
                P[RH], flags[2] = hands[0], True
            else:
                P[LH], flags[1] = hands[0], True

        active = False                 # "signing" = a hand is up in front of the upper body
        for slot, fl in ((LH, 1), (RH, 2)):
            if flags[fl]:
                wy = P[slot][0, 1]
                if sh is None or wy < sh[0][1] + 1.6 * sh[1]:
                    active = True
        return FrameData(P, flags, active)


HAND_LINKS = None


def draw_skeleton(frame, fd, mirror=False):
    """Draw hands/arms from FrameData (so it also works on a mirrored display)."""
    import cv2
    global HAND_LINKS
    if HAND_LINKS is None:
        import mediapipe as mp
        HAND_LINKS = list(mp.solutions.hands.HAND_CONNECTIONS)
    W = frame.shape[1]
    pt = lambda p: (int(W - p[0]) if mirror else int(p[0]), int(p[1]))
    P, fl = fd.P, fd.flags
    if fl[0]:
        for a, b in ((1, 2), (1, 3), (3, 5), (2, 4), (4, 6)):
            cv2.line(frame, pt(P[a]), pt(P[b]), (200, 160, 60), 2, cv2.LINE_AA)
    for slot, f, col in ((LH, 1, (80, 220, 120)), (RH, 2, (80, 160, 255))):
        if fl[f]:
            h = P[slot]
            for a, b in HAND_LINKS:
                cv2.line(frame, pt(h[a]), pt(h[b]), col, 2, cv2.LINE_AA)
            for p in h:
                cv2.circle(frame, pt(p), 3, (255, 255, 255), -1, cv2.LINE_AA)


# ------------------------------------------------------------------ features
def _body_ref(P, flags):
    """Shoulder centre + width (median over the clip) = origin and unit for body-relative coords."""
    ok = flags[:, 0]
    if ok.any():
        sc = np.median((P[ok, 1] + P[ok, 2]) / 2, axis=0)
        sw = float(np.median(np.linalg.norm((P[ok, 1] - P[ok, 2])[:, :2], axis=1)))
        return sc, max(sw, 1.0)
    hv = flags[:, 1] | flags[:, 2]
    if hv.any():                                   # fallback: no pose -> use hand size
        wr = np.where(flags[hv, 1][:, None], P[hv, 7], P[hv, 28])
        hs = np.median(np.linalg.norm((np.where(flags[hv, 1][:, None], P[hv, 16], P[hv, 37]) - wr)[:, :2], axis=1))
        return np.median(wr, axis=0), max(3.5 * float(hs), 1.0)
    return np.zeros(3, np.float32), 100.0


def _hand_block(h, present, sc, sw):
    out = np.zeros(70, np.float32)
    if not present:
        return out
    out[0:3] = (h[0] - sc) / sw                       # wrist position relative to body
    out[3:6] = (h[8] - sc) / sw                       # index fingertip relative to body
    hs = max(float(np.linalg.norm((h[9] - h[0])[:2])), 1e-3)
    out[6:69] = ((h - h[0]) / hs).reshape(-1)         # hand shape, scale/position invariant
    out[69] = 1.0
    return out


def frame_features(P, fl, sc, sw):
    f = np.zeros(FRAME_DIM, np.float32)
    if fl[0]:
        f[0:9] = ((P[[0, 3, 4]] - sc) / sw).reshape(-1)          # nose, elbows
    f[9:79] = _hand_block(P[LH], fl[1], sc, sw)
    f[79:149] = _hand_block(P[RH], fl[2], sc, sw)
    return f


def trim_range(active, margin=2, min_len=6):
    idx = np.where(active)[0]
    if len(idx) < min_len:
        return 0, len(active) - 1
    return max(0, idx[0] - margin), min(len(active) - 1, idx[-1] + margin)


def clip_features(P, flags, active, T=T_FRAMES, aug=False, rng=None):
    """One clip -> one fixed-length feature vector (T*149 + velocities)."""
    P = np.asarray(P, np.float32)
    flags = np.asarray(flags, bool)
    lo, hi = trim_range(np.asarray(active, bool))
    if aug and rng is not None:                       # random crop jitter + coordinate noise
        span = hi - lo + 1
        j = int(span * 0.12)
        if j > 0 and span - 2 * j >= 6:
            lo += int(rng.integers(0, j + 1))
            hi -= int(rng.integers(0, j + 1))
    P, flags = P[lo:hi + 1], flags[lo:hi + 1]
    sc, sw = _body_ref(P, flags)
    if aug and rng is not None:
        P = P + rng.normal(0, 0.01 * sw, P.shape).astype(np.float32)
        sc = sc + rng.normal(0, 0.02 * sw, 3)
    idx = np.round(np.linspace(0, len(P) - 1, T)).astype(int)
    feats = np.stack([frame_features(P[i], flags[i], sc, sw) for i in idx])
    vel = []
    for w0, pres in ((9, 78), (79, 148)):             # wrist velocity of each hand
        d = np.diff(feats[:, w0:w0 + 3], axis=0)
        d *= (feats[1:, pres] * feats[:-1, pres])[:, None]
        vel.append(d.ravel())
    return np.concatenate([feats.ravel()] + vel).astype(np.float32)


FEATURE_DIM = T_FRAMES * FRAME_DIM + 2 * (T_FRAMES - 1) * 3


def mirror_clip(P, flags):
    """Mirror the signer left<->right (so left-handed signers / both hands are covered)."""
    ok = flags[:, 0]
    cx = float(np.median((P[ok, 1, 0] + P[ok, 2, 0]) / 2)) if ok.any() else float(P[:, :, 0][P[:, :, 0] != 0].mean())
    Pm = P.copy()
    Pm[:, :, 0] = 2 * cx - Pm[:, :, 0]
    for a, b in ((1, 2), (3, 4), (5, 6)):
        Pm[:, [a, b]] = Pm[:, [b, a]]
    tmp = Pm[:, LH].copy()
    Pm[:, LH], Pm[:, RH] = Pm[:, RH], tmp
    fm = flags.copy()
    fm[:, [1, 2]] = fm[:, [2, 1]]
    return Pm, fm


# ------------------------------------------------------------------ clip storage
def clean_label(name):
    """'12. Thank You' -> 'thank you'."""
    s = re.sub(r"^\s*\d+\s*[\.\-_)]*\s*", "", str(name))
    return re.sub(r"\s+", " ", s.replace("_", " ")).strip().lower()


def save_clip(root, label, P, flags, active):
    d = Path(root) / re.sub(r"[^\w\- ]", "", label)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.pkl"   # unique even across parallel workers
    joblib.dump({"label": label, "P": np.asarray(P, np.float32), "flags": np.asarray(flags, bool),
                 "active": np.asarray(active, bool)}, path)
    return path


def load_clips(roots):
    clips = []
    for r in roots:
        for f in sorted(Path(r).rglob("*.pkl")):
            c = joblib.load(f)
            c["file"] = str(f)
            clips.append(c)
    return clips


# ------------------------------------------------------------------ live segmentation
class Segmenter:
    """Cuts the live stream into signs: starts when a hand comes up, ends when hands rest/leave
    for `pause` seconds. SPACE (manual mode) overrides it."""

    def __init__(self, pause=0.5, start_frames=3, max_dur=8.0, min_active=6):
        self.pause, self.start_frames, self.max_dur, self.min_active = pause, start_frames, max_dur, min_active
        self.reset()

    def reset(self):
        self.state, self.manual = "idle", False
        self.pre = deque(maxlen=self.start_frames + 2)
        self.frames, self.act_run, self.inactive_since, self.t0 = [], 0, None, 0.0

    @property
    def signing(self):
        return self.state == "signing"

    def _begin(self, frames, now):
        self.state, self.frames, self.t0 = "signing", frames, now
        self.inactive_since, self.act_run = None, 0
        self.pre.clear()

    def _finish(self):
        frames = self.frames
        self.reset()
        if sum(f.active for f in frames) < self.min_active:
            return None
        return (np.stack([f.P for f in frames]), np.stack([f.flags for f in frames]),
                np.array([f.active for f in frames]))

    def toggle_manual(self, now):
        if self.state == "idle":
            self._begin([], now)
            self.manual = True
            return None
        return self._finish()

    def update(self, fd, now):
        """Feed one FrameData. Returns (P, flags, active) when a sign just finished, else None."""
        if self.state == "idle":
            self.pre.append(fd)
            self.act_run = self.act_run + 1 if fd.active else 0
            if self.act_run >= self.start_frames:
                self._begin(list(self.pre), now)
            return None
        self.frames.append(fd)
        if now - self.t0 > self.max_dur:
            return self._finish()
        if self.manual:
            return None
        if fd.active:
            self.inactive_since = None
        elif self.inactive_since is None:
            self.inactive_since = now
        elif now - self.inactive_since >= self.pause:
            return self._finish()
        return None


# ------------------------------------------------------------------ model wrapper
class WordModel:
    def __init__(self, path):
        b = joblib.load(path)
        self.model, self.classes, self.name = b["model"], list(b["classes"]), b.get("name", "model")
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1

    def predict_topk(self, feat, k=3):
        p = self.model.predict_proba(feat[None])[0]
        order = np.argsort(p)[::-1][:k]
        return [(self.classes[int(self.model.classes_[i])] if hasattr(self.model, "classes_") else self.classes[i],
                 float(p[i])) for i in order]
