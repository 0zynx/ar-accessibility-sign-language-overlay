import argparse
import os
import tempfile
import zipfile
from collections import defaultdict
from pathlib import PurePosixPath

import numpy as np

from core import clean_label, save_clip

VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}


def open_zip(src):
    if src.lower().startswith("http"):
        try:
            from remotezip import RemoteZip
        except ImportError:
            raise SystemExit("Remote mode needs:  pip install remotezip")
        return RemoteZip(src)
    return zipfile.ZipFile(src)


def group_videos(z):
    groups = defaultdict(list)
    for name in z.namelist():
        p = PurePosixPath(name)
        if name.endswith("/") or "__MACOSX" in name or p.name.startswith("._"):
            continue
        if p.suffix.lower() in VIDEO_EXT:
            groups[clean_label(p.parent.name)].append(name)
    return groups


def video_to_clip(data, suffix, trk, target_fps):
    import cv2
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    cap = None
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        cap = cv2.VideoCapture(tmp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(fps / target_fps)))
        trk.reset()
        P, F, A, i = [], [], [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                fdta = trk.process(frame)
                P.append(fdta.P); F.append(fdta.flags); A.append(fdta.active)
            i += 1
        if sum(A) < 6:
            return None
        return np.stack(P), np.stack(F), np.array(A)
    finally:
        if cap is not None:
            cap.release()
        try:
            os.remove(tmp)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="zip URL or local .zip path")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="clips_include")
    ap.add_argument("--classes", default="")
    ap.add_argument("--top-n", type=int, default=0)
    ap.add_argument("--max-per-class", type=int, default=15)
    ap.add_argument("--fps", type=float, default=15)
    args = ap.parse_args()

    with open_zip(args.source) as z:
        groups = group_videos(z)
        if not groups:
            raise SystemExit("No videos found in that zip.")
        wanted = {clean_label(c) for c in args.classes.split(",") if c.strip()}
        if wanted:
            groups = {k: v for k, v in groups.items() if k in wanted}
        if args.top_n:
            keep = sorted(groups, key=lambda k: -len(groups[k]))[:args.top_n]
            groups = {k: groups[k] for k in keep}

        print(f"{len(groups)} words, {sum(len(v) for v in groups.values())} videos in this zip")
        for k in sorted(groups):
            print(f"  {k:<24}{len(groups[k]):>4} videos")
        if args.list:
            return

        from tqdm import tqdm
        from core import Tracker
        trk = Tracker()
        jobs = [(k, n) for k, names in groups.items() for n in sorted(names)[:args.max_per_class]]
        saved, skipped = defaultdict(int), 0
        for label, name in tqdm(jobs, desc="videos"):
            try:
                clip = video_to_clip(z.read(name), PurePosixPath(name).suffix, trk, args.fps)
            except Exception as e:
                print(f"  skipped {name}: {e}")
                skipped += 1
                continue
            if clip is None:
                skipped += 1
                continue
            save_clip(args.out, label, *clip)
            saved[label] += 1

    print(f"\nSaved clips -> {args.out}")
    for k in sorted(saved):
        print(f"  {k:<24}{saved[k]:>4} clips")
    if skipped:
        print(f"Skipped {skipped} videos (no hands found / unreadable)")


if __name__ == "__main__":
    main()