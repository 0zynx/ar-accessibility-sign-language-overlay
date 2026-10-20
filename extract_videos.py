"""Convert a downloaded sign-VIDEO dataset into landmark clips for training.

Expected layout: any folders, with the WORD as the video's parent folder, e.g.
    INCLUDE/Adjectives/1. loud/MVI_0001.MOV
    INCLUDE/Greetings/hello/clip3.mp4
The label is the parent folder name, cleaned ("1. Loud" -> "loud").

    python extract_videos.py --data INCLUDE --out clips_include
    python extract_videos.py --data INCLUDE --out clips_include --classes hello,thank you,help,water,food
    python extract_videos.py --data INCLUDE --out clips_include --max-per-class 30 --workers 4
    python extract_videos.py --data INCLUDE --out clips_include --top-n 40      # 40 words with most videos
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from core import Tracker, clean_label, save_clip

VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}
_trk = None
_cfg = {}


def _init(out, target_fps):
    global _trk, _cfg
    _trk = Tracker()
    _cfg = {"out": out, "fps": target_fps}


def _work(job):
    path, label = job
    try:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(fps / _cfg["fps"])))
        _trk.reset()
        t = _trk
        P, F, A, i = [], [], [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                fd = t.process(frame)
                P.append(fd.P); F.append(fd.flags); A.append(fd.active)
            i += 1
        cap.release()
        if sum(A) < 6:
            return label, False, str(path)
        save_clip(_cfg["out"], label, np.stack(P), np.stack(F), np.array(A))
        return label, True, str(path)
    except Exception as e:  # keep going on a bad file
        return label, False, f"{path}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="clips_include")
    ap.add_argument("--classes", default="", help="comma separated subset of words (cleaned names)")
    ap.add_argument("--top-n", type=int, default=0, help="use only the N words that have the most videos")
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--fps", type=float, default=15, help="frames per second to analyse")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) // 2))
    args = ap.parse_args()

    wanted = {clean_label(c) for c in args.classes.split(",") if c.strip()}
    per_class = {}
    for f in sorted(Path(args.data).rglob("*")):
        if f.suffix.lower() in VIDEO_EXT:
            lab = clean_label(f.parent.name)
            if wanted and lab not in wanted:
                continue
            per_class.setdefault(lab, []).append(f)
    if not per_class:
        raise SystemExit("No videos found. Check --data and --classes.")

    if args.top_n:
        keep = sorted(per_class, key=lambda l: -len(per_class[l]))[:args.top_n]
        per_class = {l: per_class[l] for l in keep}
    jobs = []
    for lab, files in per_class.items():
        if args.max_per_class:
            files = files[:args.max_per_class]
        jobs += [(f, lab) for f in files]
    print(f"{len(per_class)} words, {len(jobs)} videos, {args.workers} workers")

    kept, bad = {}, []
    with mp.Pool(args.workers, initializer=_init, initargs=(args.out, args.fps)) as pool:
        for lab, ok, info in tqdm(pool.imap_unordered(_work, jobs), total=len(jobs)):
            if ok:
                kept[lab] = kept.get(lab, 0) + 1
            else:
                bad.append(info)
    print(f"\nSaved clips for {len(kept)} words -> {args.out}")
    for lab in sorted(kept):
        print(f"  {lab:<20} {kept[lab]}")
    if bad:
        print(f"\nSkipped {len(bad)} videos (no hands found / unreadable). First few: {bad[:3]}")


if __name__ == "__main__":
    main()
