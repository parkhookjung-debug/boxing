"""
Train a +/-7 frame LIM punch sequence classifier.

The older model looks mostly at a single punch frame plus short motion summaries.
This trainer uses the labeled punch frame and seven frames before/after it, so
rear hooks, recovery motion, and uppercuts can be separated by the whole motion.

Run from D:\\Code\\boxing:
    python LIM_train_sequence.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
from collections import Counter

import numpy as np

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit("scikit-learn and numpy are required: pip install scikit-learn numpy") from exc


KEYPOINTS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]

HALF_WINDOW = 7
WINDOW = HALF_WINDOW * 2 + 1


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def xy(row: dict[str, str], name: str) -> tuple[float, float]:
    return float(row[f"{name}_x"]), float(row[f"{name}_y"])


def shoulder_width(row: dict[str, str]) -> float:
    lx, ly = xy(row, "left_shoulder")
    rx, ry = xy(row, "right_shoulder")
    return math.hypot(rx - lx, ry - ly) + 1e-6


def angle3(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    den = math.hypot(bax, bay) * math.hypot(bcx, bcy) + 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, (bax * bcx + bay * bcy) / den))))


def wrist_displacement(rows: list[dict[str, str]], idx: int, wrist: str, lookback: int = 6) -> float:
    lo = max(0, idx - lookback)
    sub = rows[lo : idx + 1]
    scale = shoulder_width(rows[idx])
    pts = [xy(r, wrist) for r in sub]
    if len(pts) < 2:
        return 0.0
    return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts))) / scale


def assign_side(rows: list[dict[str, str]], idx: int) -> str:
    left = wrist_displacement(rows, idx, "left_wrist")
    right = wrist_displacement(rows, idx, "right_wrist")
    return "lead" if left >= right else "rear"


def side_names(side: str) -> tuple[str, str, str, str, str, str]:
    if side == "lead":
        return "left_wrist", "left_elbow", "left_shoulder", "right_wrist", "right_elbow", "right_shoulder"
    return "right_wrist", "right_elbow", "right_shoulder", "left_wrist", "left_elbow", "left_shoulder"


def normalized_points(row: dict[str, str]) -> dict[str, tuple[float, float]]:
    scale = shoulder_width(row)
    lsx, lsy = xy(row, "left_shoulder")
    rsx, rsy = xy(row, "right_shoulder")
    cx, cy = (lsx + rsx) * 0.5, (lsy + rsy) * 0.5
    return {name: ((xy(row, name)[0] - cx) / scale, (xy(row, name)[1] - cy) / scale) for name in KEYPOINTS}


def frame_metrics(row: dict[str, str], side: str) -> list[float]:
    wr, el, sh, owr, oel, osh = side_names(side)
    scale = shoulder_width(row)
    wx, wy = xy(row, wr)
    ex, ey = xy(row, el)
    sx, sy = xy(row, sh)
    owx, owy = xy(row, owr)
    oex, oey = xy(row, oel)
    osx, osy = xy(row, osh)
    return [
        math.hypot(wx - sx, wy - sy) / scale,
        angle3((sx, sy), (ex, ey), (wx, wy)) / 180.0,
        math.hypot(wx - ex, wy - ey) / scale,
        (wy - sy) / scale,
        (ey - sy) / scale,
        abs(ex - sx) / scale,
        math.hypot(owx - osx, owy - osy) / scale,
        angle3((osx, osy), (oex, oey), (owx, owy)) / 180.0,
    ]


def make_sequence_features(rows: list[dict[str, str]], center_idx: int, side: str) -> list[float]:
    idxs = [min(len(rows) - 1, max(0, center_idx + off)) for off in range(-HALF_WINDOW, HALF_WINDOW + 1)]
    features: list[float] = []

    prev_punch_wr = None
    prev_opp_wr = None
    punch_wr, punch_el, punch_sh, opp_wr, opp_el, opp_sh = side_names(side)
    punch_wr_x, punch_wr_y = [], []
    punch_ext, punch_ew, punch_angle = [], [], []

    for idx in idxs:
        row = rows[idx]
        points = normalized_points(row)
        ordered = [punch_wr, punch_el, punch_sh, opp_wr, opp_el, opp_sh, "nose", "left_hip", "right_hip"]
        for name in ordered:
            features.extend(points[name])

        metrics = frame_metrics(row, side)
        features.extend(metrics)

        pwr = points[punch_wr]
        owr = points[opp_wr]
        if prev_punch_wr is None:
            features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([
                pwr[0] - prev_punch_wr[0],
                pwr[1] - prev_punch_wr[1],
                owr[0] - prev_opp_wr[0],
                owr[1] - prev_opp_wr[1],
            ])
        prev_punch_wr = pwr
        prev_opp_wr = owr
        punch_wr_x.append(pwr[0])
        punch_wr_y.append(pwr[1])
        punch_ext.append(metrics[0])
        punch_angle.append(metrics[1] * 180.0)
        punch_ew.append(metrics[2])

    features.extend(
        [
            1.0 if side == "rear" else 0.0,
            max(punch_ext),
            min(punch_ext),
            max(punch_ext) - min(punch_ext),
            max(punch_ext) - punch_ext[0],
            max(punch_ext) - punch_ext[-1],
            min(punch_ew),
            max(punch_ew),
            max(punch_ew) - min(punch_ew),
            min(punch_angle) / 180.0,
            max(punch_angle) / 180.0,
            max(punch_wr_x) - min(punch_wr_x),
            max(punch_wr_y) - min(punch_wr_y),
            punch_wr_y[0] - min(punch_wr_y),
            max(punch_wr_y) - punch_wr_y[0],
        ]
    )
    return features


def build_dataset(base_dir: str):
    x, y, meta = [], [], []
    for video_id in range(1, 9):
        data_path = os.path.join(base_dir, f"LIM_full_data{video_id}.csv")
        label_path = os.path.join(base_dir, f"LIM{video_id}_labels.csv")
        if not os.path.exists(data_path) or not os.path.exists(label_path):
            continue
        rows = read_rows(data_path)
        frame_to_idx = {int(r["frame_number"]): i for i, r in enumerate(rows)}
        labels = read_rows(label_path)
        for label in labels:
            frame = int(label["frame_number"])
            punch = label["punch_type"].strip().lower()
            if frame not in frame_to_idx:
                continue
            idx = frame_to_idx[frame]
            side = assign_side(rows, idx)
            x.append(make_sequence_features(rows, idx, side))
            y.append(punch)
            meta.append({"video": video_id, "frame": frame, "side": side, "punch": punch})
    return np.array(x, dtype=np.float32), np.array(y), meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--out", default="lim_punch_seq_model.pkl")
    args = parser.parse_args()

    x, y, meta = build_dataset(args.base_dir)
    if len(x) == 0:
        raise SystemExit("No training samples found.")

    print(f"Samples: {len(x)}")
    print(f"Features: {x.shape[1]}")
    print(f"Class counts: {Counter(y)}")
    print(f"Side counts: {Counter((m['punch'], m['side']) for m in meta)}")

    rf = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=7,
        n_jobs=-1,
    )
    clf = VotingClassifier(
        estimators=[("rf", rf), ("et", et)],
        voting="soft",
        weights=[1, 1],
        n_jobs=-1,
    )
    pipe = Pipeline([("scale", StandardScaler()), ("model", clf)])

    if min(Counter(y).values()) >= 4:
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=3)
        pred = cross_val_predict(pipe, x, y, cv=cv, n_jobs=-1)
        print("\nCross-val report:")
        print(classification_report(y, pred, digits=3))
        print("Confusion matrix labels:", sorted(set(y)))
        print(confusion_matrix(y, pred, labels=sorted(set(y))))

    pipe.fit(x, y)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(args.base_dir, args.out)
    bundle = {
        "model": pipe,
        "classes": sorted(set(y)),
        "half_window": HALF_WINDOW,
        "window": WINDOW,
        "feature_count": int(x.shape[1]),
        "keypoints": KEYPOINTS,
        "meta": {
            "samples": len(x),
            "class_counts": dict(Counter(y)),
            "side_counts": {f"{k[0]}:{k[1]}": v for k, v in Counter((m["punch"], m["side"]) for m in meta).items()},
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
