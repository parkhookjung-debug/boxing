"""
Build boxing style profiles from pose CSV files.

The fighter names are treated as source datasets for style archetypes:
- Bivol  -> Soviet Out-Boxing
- Canelo -> Mexican Pressure Counter
- Garcia -> American Speed Boxer
- LIM    -> LIM Adaptive Hybrid

Run from D:\\Code\\boxing:
    python style_profile.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from statistics import mean, pstdev


STYLE_SOURCES = {
    "soviet_outboxing": {
        "display_name": "Soviet Out-Boxing",
        "source_fighter": "Bivol",
        "pattern": "bivol_full_data*.csv",
        "description": "Distance control, straight punches, measured rhythm, and stable posture.",
        "training_focus": [
            "Jab control and step-out exits",
            "Straight punch recovery",
            "Long-range stance stability",
        ],
    },
    "mexican_pressure_counter": {
        "display_name": "Mexican Pressure Counter",
        "source_fighter": "Canelo",
        "pattern": "canelo_full_data*.csv",
        "description": "Forward pressure, compact guard, body/head level changes, and counters.",
        "training_focus": [
            "Compact hook mechanics",
            "Pressure steps after defense",
            "Body-to-head punch transitions",
        ],
    },
    "american_speed_boxer": {
        "display_name": "American Speed Boxer",
        "source_fighter": "Garcia",
        "pattern": "garcia_full_data*.csv",
        "description": "Hand speed, rhythm changes, sharp entries, and fast punch release.",
        "training_focus": [
            "Explosive first punch",
            "Tempo changes",
            "Fast hand return after combinations",
        ],
    },
    "lim_adaptive_hybrid": {
        "display_name": "LIM Adaptive Hybrid",
        "source_fighter": "LIM",
        "pattern": "LIM_full_data*.csv",
        "description": "Personalized LIM reference style built from the user's own labeled sessions.",
        "training_focus": [
            "Personal baseline refinement",
            "Punch recognition consistency",
            "Style blend optimization",
        ],
    },
}

KP_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

FEATURE_ORDER = [
    "guard_lead_height",
    "guard_rear_height",
    "guard_symmetry",
    "head_height",
    "head_stability",
    "lean_forward",
    "lean_variability",
    "stance_width",
    "stance_variability",
    "lead_hand_activity",
    "rear_hand_activity",
    "hand_activity_balance",
    "lead_arm_extension",
    "rear_arm_extension",
    "extension_balance",
    "lead_elbow_compactness",
    "rear_elbow_compactness",
    "wrist_speed_mean",
    "wrist_speed_burst",
    "rhythm_variability",
    "vertical_motion",
    "lateral_motion",
]


@dataclass
class PoseFrame:
    raw: dict[str, str]

    def xy(self, name: str) -> tuple[float, float]:
        return float(self.raw[f"{name}_x"]), float(self.raw[f"{name}_y"])

    def score(self, name: str) -> float:
        key = f"{name}_v"
        try:
            return float(self.raw[key])
        except (KeyError, ValueError):
            return 1.0


def read_csv_rows(path: str) -> list[PoseFrame]:
    with open(path, newline="", encoding="utf-8") as f:
        return [PoseFrame(row) for row in csv.DictReader(f)]


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle3(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    den = math.hypot(bax, bay) * math.hypot(bcx, bcy) + 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, (bax * bcx + bay * bcy) / den))))


def shoulder_width(frame: PoseFrame) -> float:
    return dist(frame.xy("left_shoulder"), frame.xy("right_shoulder")) + 1e-6


def center(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
    return vals[idx]


def extract_features(frames: list[PoseFrame]) -> dict[str, float]:
    guard_l, guard_r = [], []
    head_y, lean_x, stance = [], [], []
    l_ext, r_ext = [], []
    l_compact, r_compact = [], []
    l_speed, r_speed = [], []
    l_dx, r_dx, l_dy_up, r_dy_up = [], [], [], []
    hand_speed_all = []

    prev: PoseFrame | None = None
    for frame in frames:
        sw = shoulder_width(frame)
        ls, rs = frame.xy("left_shoulder"), frame.xy("right_shoulder")
        le, re = frame.xy("left_elbow"), frame.xy("right_elbow")
        lw, rw = frame.xy("left_wrist"), frame.xy("right_wrist")
        lh, rh = frame.xy("left_hip"), frame.xy("right_hip")
        nose = frame.xy("nose")
        sh_c = center(ls, rs)
        hip_c = center(lh, rh)

        guard_l.append((lw[1] - ls[1]) / sw)
        guard_r.append((rw[1] - rs[1]) / sw)
        head_y.append((nose[1] - sh_c[1]) / sw)
        lean_x.append((sh_c[0] - hip_c[0]) / sw)
        stance.append(dist(lh, rh) / sw)

        l_ext.append(dist(lw, ls) / sw)
        r_ext.append(dist(rw, rs) / sw)
        l_angle = angle3(ls, le, lw)
        r_angle = angle3(rs, re, rw)
        l_compact.append(1.0 - min(1.0, l_angle / 180.0))
        r_compact.append(1.0 - min(1.0, r_angle / 180.0))

        if prev is not None:
            psw = shoulder_width(prev)
            plw, prw = prev.xy("left_wrist"), prev.xy("right_wrist")
            ld = dist(lw, plw) / psw
            rd = dist(rw, prw) / psw
            l_speed.append(ld)
            r_speed.append(rd)
            hand_speed_all.extend([ld, rd])
            l_dx.append(abs(lw[0] - plw[0]) / psw)
            r_dx.append(abs(rw[0] - prw[0]) / psw)
            l_dy_up.append(max(0.0, plw[1] - lw[1]) / psw)
            r_dy_up.append(max(0.0, prw[1] - rw[1]) / psw)
        prev = frame

    lead_activity = safe_mean(l_speed)
    rear_activity = safe_mean(r_speed)
    total_activity = lead_activity + rear_activity + 1e-6
    lead_ext = safe_mean(l_ext)
    rear_ext = safe_mean(r_ext)

    features = {
        "guard_lead_height": -safe_mean(guard_l),
        "guard_rear_height": -safe_mean(guard_r),
        "guard_symmetry": -abs(safe_mean(guard_l) - safe_mean(guard_r)),
        "head_height": -safe_mean(head_y),
        "head_stability": -safe_std(head_y),
        "lean_forward": safe_mean(lean_x),
        "lean_variability": safe_std(lean_x),
        "stance_width": safe_mean(stance),
        "stance_variability": safe_std(stance),
        "lead_hand_activity": lead_activity,
        "rear_hand_activity": rear_activity,
        "hand_activity_balance": 1.0 - abs(lead_activity - rear_activity) / total_activity,
        "lead_arm_extension": lead_ext,
        "rear_arm_extension": rear_ext,
        "extension_balance": 1.0 - abs(lead_ext - rear_ext) / (lead_ext + rear_ext + 1e-6),
        "lead_elbow_compactness": safe_mean(l_compact),
        "rear_elbow_compactness": safe_mean(r_compact),
        "wrist_speed_mean": safe_mean(hand_speed_all),
        "wrist_speed_burst": percentile(hand_speed_all, 0.90),
        "rhythm_variability": safe_std(hand_speed_all),
        "vertical_motion": safe_mean(l_dy_up + r_dy_up),
        "lateral_motion": safe_mean(l_dx + r_dx),
    }
    return {k: round(float(features[k]), 6) for k in FEATURE_ORDER}


def build_profiles(base_dir: str) -> dict:
    profiles = {}
    for style_id, info in STYLE_SOURCES.items():
        paths = sorted(glob.glob(os.path.join(base_dir, info["pattern"])))
        frames: list[PoseFrame] = []
        for path in paths:
            frames.extend(read_csv_rows(path))
        if not frames:
            continue
        features = extract_features(frames)
        profiles[style_id] = {
            "display_name": info["display_name"],
            "source_fighter": info["source_fighter"],
            "description": info["description"],
            "training_focus": info["training_focus"],
            "source_files": [os.path.basename(path) for path in paths],
            "frame_count": len(frames),
            "features": features,
        }
    return {
        "schema_version": 1,
        "feature_order": FEATURE_ORDER,
        "profiles": profiles,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--out", default="style_profiles.json")
    args = parser.parse_args()

    result = build_profiles(args.base_dir)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(args.base_dir, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
    for style_id, profile in result["profiles"].items():
        print(f"- {style_id}: {profile['frame_count']} frames, {len(profile['source_files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
