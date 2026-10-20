"""Record your own words/phrases (best for phrases, and to add your own signer to a dataset).

    python record_words.py --words "hello,thank you,how are you,i need help"
    python record_words.py --words "water" --source http://PHONE_IP:8080/video     # phone camera

Each word/phrase is ONE class. For every one record 25+ clips: vary speed, distance, position,
clothes, lighting - and get friends to sign too. More variety = better recognition.

Keys:  SPACE start (2 s countdown) / stop a clip | n next word | p previous word
       u delete last clip of this word | h skeleton | q quit
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ar_ui import FONT, LatestFrame
from core import Tracker, clean_label, draw_skeleton, save_clip, trim_range


def count_clips(root, label):
    d = Path(root) / label
    return len(list(d.glob("*.pkl"))) if d.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", required=True, help='comma separated, e.g. "hello,thank you"')
    ap.add_argument("--out", default="clips_own")
    ap.add_argument("--source", default="0")
    ap.add_argument("--mirror", choices=["on", "off"], default="on", help="mirror the DISPLAY only")
    args = ap.parse_args()

    words = [clean_label(w) for w in args.words.split(",") if w.strip()]
    src = int(args.source) if args.source.isdigit() else args.source
    mirror = args.mirror == "on"
    cam, trk = LatestFrame(src), Tracker()
    wi, state, t_press, buf, skeleton = 0, "idle", 0.0, [], True   # state: idle | countdown | rec
    last_saved = {}

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            cv2.waitKey(10)
            continue
        fd = trk.process(frame)               # always process the UN-mirrored frame
        H, W = frame.shape[:2]
        disp = cv2.flip(frame, 1) if mirror else frame
        if skeleton:
            draw_skeleton(disp, fd, mirror)
        now = time.time()
        label = words[wi]

        if state == "countdown":
            left = 2.0 - (now - t_press)
            if left <= 0:
                state, buf = "rec", []
            else:
                cv2.putText(disp, str(int(np.ceil(left))), (W // 2 - 40, H // 2 + 40), FONT, 5, (0, 255, 255), 8, cv2.LINE_AA)
        if state == "rec":
            buf.append(fd)
            cv2.circle(disp, (40, 90), 14, (0, 0, 255), -1)
            cv2.putText(disp, f"REC {now - t_press - 2.0:0.1f}s  (SPACE to stop)", (65, 98), FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            if now - t_press - 2.0 > 8:       # safety stop
                state = "stop"

        if state == "stop":
            P, flags, act = (np.stack([f.P for f in buf]), np.stack([f.flags for f in buf]),
                             np.array([f.active for f in buf]))
            lo, hi = trim_range(act)
            if act.sum() < 6:
                print("  too short / no hands seen - discarded, try again")
            else:
                last_saved[label] = save_clip(args.out, label, P, flags, act)
                print(f"  saved '{label}' ({hi - lo + 1} frames) -> {count_clips(args.out, label)} clips")
            state, buf = "idle", []

        n = count_clips(args.out, label)
        cv2.putText(disp, f"WORD {wi + 1}/{len(words)}:  {label.upper()}", (20, 40), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"clips: {n}  {'(aim for 25+)' if n < 25 else '- enough, or add more variety'}",
                    (20, H - 60), FONT, 0.7, (80, 255, 120) if n >= 25 else (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, "SPACE rec/stop | n next | p prev | u undo | h skeleton | q quit",
                    (20, H - 25), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.imshow("SignBridge recorder", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" "):
            if state == "idle":
                state, t_press = "countdown", now
            elif state == "rec":
                state = "stop"
        elif k == ord("n") and state == "idle":
            wi = (wi + 1) % len(words)
        elif k == ord("p") and state == "idle":
            wi = (wi - 1) % len(words)
        elif k == ord("u") and state == "idle" and label in last_saved and last_saved[label].exists():
            last_saved[label].unlink()
            print(f"  deleted last clip of '{label}'")
        elif k == ord("h"):
            skeleton = not skeleton

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
