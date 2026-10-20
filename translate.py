"""SignBridge Words - live ISL words/phrases -> English with an AR bubble that follows your hand.

    python translate.py                                   # webcam
    python translate.py --source http://PHONE_IP:8080/video --mirror off    # Android phone camera
    python translate.py --model models/words.joblib

How it works: raise your hand(s) and sign. When you lower them / pause (~0.5 s) the sign is
recognised, shown in the bubble beside your hand, added to the sentence bar and spoken.
No model yet? It still runs ("SIGN") so you can test hand tracking and the AR bubble first.

Keys: SPACE manual start/stop (use if auto-detect misbehaves) | c clear | s speak sentence
      BACKSPACE undo last | h skeleton | v voice on/off | q quit
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ar_ui import (FONT, STRIP_H, LatestFrame, OneEuro, Speaker, draw_bubble, draw_sentence_strip,
                   ease_out_back, measure_bubble, conf_text, place_bubble)
from core import Segmenter, Tracker, WordModel, clip_features, draw_skeleton

RESULT_TTL = 3.0         # seconds the recognised word stays in the bubble


def hand_box_and_size(fd, W, mirror):
    """Bounding box (display coordinates) + size of the visible hands, or None."""
    pts, size = [], 0.0
    for sl, f in ((slice(7, 28), 1), (slice(28, 49), 2)):
        if fd.flags[f]:
            h = fd.P[sl]
            pts.append(h[:, :2])
            size = max(size, float(np.linalg.norm((h[9] - h[0])[:2])))
    if not pts:
        return None, 0.0
    p = np.vstack(pts)
    x1, y1, x2, y2 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    pad = 0.08 * max(x2 - x1, y2 - y1)
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    if mirror:
        x1, x2 = W - x2, W - x1
    return np.array([x1, y1, x2, y2]), size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="models/words.joblib")
    ap.add_argument("--mirror", choices=["auto", "on", "off"], default="auto", help="mirror the DISPLAY only")
    ap.add_argument("--pause", type=float, default=0.5, help="seconds of rest that end a sign")
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--no-voice", action="store_true")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    mirror = isinstance(src, int) if args.mirror == "auto" else args.mirror == "on"
    model = None
    if Path(args.model).exists():
        model = WordModel(args.model)
        print(f"Loaded {model.name} model: {len(model.classes)} words/phrases")
    else:
        print(f"[no model at {args.model}] demo mode: shows 'SIGN' so you can test tracking + AR")

    cam, trk, seg = LatestFrame(src), Tracker(), Segmenter(pause=args.pause)
    speaker = Speaker(enabled=not args.no_voice)
    f_box, f_size = OneEuro(1.2, 0.02), OneEuro(0.8, 0.01)
    bubble_c, side, vis = None, 1, 0.0
    result = None                       # dict(text, conf, t)
    shown, pop_start, last_hand_t, last_t = None, 0.0, 0.0, time.time()
    words, show_skel, fps = [], True, 0.0

    def finish(clip):
        nonlocal result
        if clip is None:
            return
        P, flags, act = clip
        if model is None:
            result = dict(text="SIGN", conf=1.0, t=time.time())
            return
        top = model.predict_topk(clip_features(P, flags, act), 3)
        print("  " + "   ".join(f"{l} {p*100:.0f}%" for l, p in top))
        (label, conf) = top[0]
        if conf >= args.min_conf:
            result = dict(text=label.upper(), conf=conf, t=time.time())
            words.append(label.upper())
            speaker.say(label)
        else:
            result = dict(text="NOT SURE", conf=conf, t=time.time())

    print("Running. q in the video window quits.")
    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            cv2.waitKey(10)
            continue
        H, W = frame.shape[:2]
        now = time.time()
        dt = max(now - last_t, 1e-3)
        last_t = now
        fps = 0.9 * fps + 0.1 / dt

        fd = trk.process(frame)                       # tracking uses the un-mirrored frame
        finish(seg.update(fd, now))
        disp = cv2.flip(frame, 1) if mirror else frame
        if show_skel:
            draw_skeleton(disp, fd, mirror)

        box, hsize = hand_box_and_size(fd, W, mirror)
        if box is not None:
            if now - last_hand_t > 0.6:
                f_box.reset(); f_size.reset(); bubble_c = None
            last_hand_t = now
            f_box(box, now); f_size(hsize, now)

        # what does the bubble say right now?
        text, conf, progress = None, 0.0, 0.0
        if seg.signing:
            text = "..."
        elif result and now - result["t"] < RESULT_TTL:
            text, conf = result["text"], result["conf"]
            progress = 1.0 - (now - result["t"]) / RESULT_TTL
        want = text is not None and f_box.x_prev is not None
        vis = min(1.0, vis + dt * 8) if want else max(0.0, vis - dt * 5)

        if text is not None and f_box.x_prev is not None and vis > 0.02:
            bb = f_box.x_prev
            hs = float(f_size.x_prev) if f_size.x_prev is not None else 110.0
            s = float(np.clip(hs / 110.0, 0.65, 1.6))      # nearer hand -> bigger bubble
            if text != shown:
                if shown in (None, "...") or text != "...":
                    pop_start = now
                shown = text
            pop = max(ease_out_back(min(1.0, (now - pop_start) / 0.28)), 0.01)
            max_w = min(0.5 * W, 420 * s)
            m = measure_bubble(text, conf_text(text, conf), s, max_w)
            target, side, tip = place_bubble(bb, W, H, m["bw"], m["bh"], side, s)
            bubble_c = target if bubble_c is None else bubble_c + 0.35 * (target - bubble_c)
            draw_bubble(disp, text, conf, bubble_c, side, tip, s * pop, vis, progress, max_w)
        elif not want:
            shown = None

        if seg.signing:
            cv2.circle(disp, (30, 70), 10, (0, 0, 255), -1)
            cv2.putText(disp, "signing..." + ("  (manual)" if seg.manual else ""), (50, 78),
                        FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        draw_sentence_strip(disp, words)
        cv2.putText(disp, f"{fps:3.0f} FPS", (W - 110, 56), FONT, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(disp, "SPACE manual | c clear | s speak | bksp undo | h skeleton | v voice | q quit",
                    (12, 24), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.imshow("SignBridge Words", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" "):
            finish(seg.toggle_manual(now))
        elif k == ord("c"):
            words.clear()
        elif k == ord("s") and words:
            speaker.say(" . ".join(words))
        elif k in (8, 127) and words:
            words.pop()
        elif k == ord("h"):
            show_skel = not show_skel
        elif k == ord("v"):
            speaker.enabled = not speaker.enabled

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
