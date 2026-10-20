"""Train the word/phrase recognizer from recorded or extracted clips.

    python train_words.py --data clips_own clips_include
    python train_words.py --data clips_own --models svm mlp rf --aug 4

Evaluation is held out BY CLIP (before any augmentation), so the reported accuracy is honest.
Classes with fewer than --min-clips clips are dropped (with a warning).
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from core import T_FRAMES, clean_label, clip_features, load_clips, mirror_clip


def build_models(n_samples, n_feat):
    comps = max(2, min(200, n_samples - 1, n_feat))
    return {
        "svm": lambda: make_pipeline(StandardScaler(), PCA(comps, whiten=True, random_state=0),
                                     SVC(C=10, gamma="scale", probability=True, random_state=0)),
        "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=300, early_stopping=True, random_state=0)),
        "rf": lambda: RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0),
    }


def topk_acc(model, X, y, k=3):
    p = model.predict_proba(X)
    top = np.argsort(p, axis=1)[:, ::-1][:, :k]
    cls = model.classes_
    return float(np.mean([y[i] in cls[top[i]] for i in range(len(y))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["clips_own"])
    ap.add_argument("--models", nargs="+", default=["svm", "mlp"], choices=["svm", "mlp", "rf"])
    ap.add_argument("--aug", type=int, default=3, help="random augmented copies per clip (plus mirror)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--min-clips", type=int, default=3)
    ap.add_argument("--out", default="models/words.joblib")
    args = ap.parse_args()

    clips = load_clips(args.data)
    for c in clips:
        c["label"] = clean_label(c["label"])
    counts = Counter(c["label"] for c in clips)
    keep = sorted(l for l, n in counts.items() if n >= args.min_clips)
    for l, n in sorted(counts.items()):
        if l not in keep:
            print(f"[drop] '{l}' has only {n} clips (< {args.min_clips})")
    if len(keep) < 2:
        raise SystemExit("Need at least 2 words with enough clips. Record/extract more data.")
    print(f"{len(clips)} clips, {len(keep)} classes: {keep[:15]}{' ...' if len(keep) > 15 else ''}")

    rng = np.random.default_rng(0)
    train, test = [], []
    for l in keep:
        group = [c for c in clips if c["label"] == l]
        rng.shuffle(group)
        n_test = max(1, round(args.test_frac * len(group))) if len(group) >= 5 else 0
        test += group[:n_test]
        train += group[n_test:]
    cid = {l: i for i, l in enumerate(keep)}

    t0 = time.time()
    Xtr, ytr = [], []
    for c in train:
        P, fl, act = c["P"], c["flags"], c["active"]
        Pm, fm = mirror_clip(P, fl)
        for (p, f) in ((P, fl), (Pm, fm)):
            Xtr.append(clip_features(p, f, act)); ytr.append(cid[c["label"]])
            for _ in range(args.aug):
                Xtr.append(clip_features(p, f, act, aug=True, rng=rng)); ytr.append(cid[c["label"]])
    Xtr, ytr = np.array(Xtr, np.float32), np.array(ytr)
    Xte = np.array([clip_features(c["P"], c["flags"], c["active"]) for c in test], np.float32)
    yte = np.array([cid[c["label"]] for c in test])
    print(f"train {Xtr.shape} (from {len(train)} clips)   test {Xte.shape}   features built in {time.time()-t0:.0f}s")

    builders = build_models(len(Xtr), Xtr.shape[1])
    results, trained = {}, {}
    for name in args.models:
        t = time.time()
        model = builders[name]().fit(Xtr, ytr)
        if len(Xte):
            a1, a3 = float(np.mean(model.predict(Xte) == yte)), topk_acc(model, Xte, yte, 3)
            results[name] = (a1, a3)
            print(f"[{name.upper():>3}] top-1 {a1*100:5.1f}%   top-3 {a3*100:5.1f}%   ({time.time()-t:.0f}s)")
        else:
            results[name] = (0.0, 0.0)
            print(f"[{name.upper():>3}] trained (no test split: some words have <5 clips)")
        trained[name] = model

    best = max(results, key=lambda n: results[n][0])
    if len(Xte):
        pred = trained[best].predict(Xte)
        wrong = Counter((keep[a], keep[b]) for a, b in zip(yte, pred) if a != b)
        if wrong:
            print("\nMost confused (true -> predicted):")
            for (a, b), n in wrong.most_common(6):
                print(f"  {a} -> {b}  x{n}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # final model: refit best on ALL clips (train + test) so no data is wasted
    if len(Xte):
        Xall = np.vstack([Xtr, np.array([clip_features(c["P"], c["flags"], c["active"]) for c in test]),
                          np.array([clip_features(mirror_clip(c["P"], c["flags"])[0], mirror_clip(c["P"], c["flags"])[1], c["active"]) for c in test])])
        yall = np.concatenate([ytr, yte, yte])
        final = build_models(len(Xall), Xall.shape[1])[best]().fit(Xall, yall)
    else:
        final = trained[best]
    joblib.dump({"model": final, "classes": keep, "T": T_FRAMES, "name": best.upper()}, args.out)
    print(f"\nBest: {best.upper()}.  Saved {args.out}  ({len(keep)} words/phrases)")


if __name__ == "__main__":
    main()
