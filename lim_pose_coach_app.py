"""
LIM pose-based boxing coach app.

Uses the existing LIM CSV-derived assets in this folder:
- lim_punch_model.pkl: punch classifier trained from LIM_full_data*.csv + labels.
- punch_templates.csv: normalized pose templates for front/side punch matching.
- LIM_punch_DNA_all.csv and LIM_DNA.csv: reference posture and punch metrics.

Run:
    python lim_pose_coach_app.py
    python lim_pose_coach_app.py --source "LIM 7.mp4"
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Iterable

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KP_NOSE = 0
KP_L_SH, KP_R_SH = 5, 6
KP_L_EL, KP_R_EL = 7, 8
KP_L_WR, KP_R_WR = 9, 10
KP_L_HI, KP_R_HI = 11, 12
KP_L_KN, KP_R_KN = 13, 14
KP_L_AN, KP_R_AN = 15, 16

NEEDED = [KP_NOSE, KP_L_SH, KP_R_SH, KP_L_EL, KP_R_EL, KP_L_WR, KP_R_WR]
VIS_MIN = 0.30

SKELETON = [
    (KP_L_SH, KP_R_SH),
    (KP_L_SH, KP_L_EL),
    (KP_L_EL, KP_L_WR),
    (KP_R_SH, KP_R_EL),
    (KP_R_EL, KP_R_WR),
    (KP_L_SH, KP_L_HI),
    (KP_R_SH, KP_R_HI),
    (KP_L_HI, KP_R_HI),
    (KP_L_HI, KP_L_KN),
    (KP_L_KN, KP_L_AN),
    (KP_R_HI, KP_R_KN),
    (KP_R_KN, KP_R_AN),
    (KP_NOSE, KP_L_SH),
    (KP_NOSE, KP_R_SH),
]

PUNCH_COLORS = {
    "jab": (0, 220, 255),
    "cross": (255, 150, 40),
    "hook": (210, 70, 255),
    "uppercut": (80, 240, 90),
}


def load_ui_font(size: int):
    if ImageFont is None:
        return None
    for path in (
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


UI_FONTS = {size: load_ui_font(size) for size in (18, 20, 22, 24, 28, 32)}

FEATURE_NAMES = [
    "arm_ext",
    "elbow_angle",
    "elbow_y",
    "elbow_lat",
    "wrist_y",
    "ew",
    "arm_dy_rise",
    "wr_dx_abs",
    "wr_dy_min",
    "wr_dy_rise",
    "wr_peak_v",
    "opp_wr_dx_abs",
]

TEMPLATE_KPS = [
    ("left_wrist", KP_L_WR),
    ("right_wrist", KP_R_WR),
    ("left_elbow", KP_L_EL),
    ("right_elbow", KP_R_EL),
    ("left_shoulder", KP_L_SH),
    ("right_shoulder", KP_R_SH),
    ("nose", KP_NOSE),
    ("left_hip", KP_L_HI),
    ("right_hip", KP_R_HI),
]

POSE_DEFAULTS = {
    "guard_l_ydiff": -0.15,
    "guard_r_ydiff": -0.10,
    "head_y_ratio": -0.85,
    "lean_forward": 0.05,
    "stance_3d_ratio": 1.20,
}


@dataclass
class Template:
    punch_type: str
    side: str
    view: str
    vec: np.ndarray


@dataclass
class PunchResult:
    punch_type: str
    side: str
    confidence: float
    source: str
    feedback: str


def parse_source(value: str):
    if value.isdigit():
        return int(value)
    return value


def load_classifier(path: str):
    if not os.path.exists(path):
        return None, [], 8
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle.get("model"), bundle.get("features", FEATURE_NAMES), int(bundle.get("window", 8))


def load_sequence_classifier(path: str):
    if not os.path.exists(path):
        return None, 0, 0
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle.get("model"), int(bundle.get("half_window", 7)), int(bundle.get("feature_count", 0))


def load_templates(path: str) -> list[Template]:
    if not os.path.exists(path):
        return []
    templates: list[Template] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vec = []
            for name, _idx in TEMPLATE_KPS:
                vec.append(float(row[f"{name}_dx"]))
                vec.append(float(row[f"{name}_dy"]))
            templates.append(
                Template(
                    punch_type=row["punch_type"].strip().lower(),
                    side=row["side"].strip().lower(),
                    view=row["view"].strip().lower(),
                    vec=np.array(vec, dtype=float),
                )
            )
    return templates


def load_pose_dna(path: str) -> dict[str, float]:
    dna = dict(POSE_DEFAULTS)
    if not os.path.exists(path):
        return dna
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in dna:
                try:
                    dna[key] = float(row[key])
                except (KeyError, TypeError, ValueError):
                    pass
            break
    return dna


def shoulder_width(kp: np.ndarray) -> float:
    return float(np.linalg.norm(kp[KP_R_SH, :2] - kp[KP_L_SH, :2]) + 1e-6)


def robust_scale(kp: np.ndarray) -> float:
    sw = shoulder_width(kp)
    hip = float(np.linalg.norm(kp[KP_R_HI, :2] - kp[KP_L_HI, :2]) + 1e-6)
    return max(sw, hip, 1e-6)


def angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_v = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos_v))


def side_to_indices(side: str, orthodox: bool) -> tuple[int, int, int]:
    if side == "lead":
        return (KP_L_WR, KP_L_EL, KP_L_SH) if orthodox else (KP_R_WR, KP_R_EL, KP_R_SH)
    return (KP_R_WR, KP_R_EL, KP_R_SH) if orthodox else (KP_L_WR, KP_L_EL, KP_L_SH)


def sequence_indices(side: str, orthodox: bool) -> tuple[int, int, int, int, int, int]:
    wr, el, sh = side_to_indices(side, orthodox)
    owr, oel, osh = side_to_indices(opposite_side(side), orthodox)
    return wr, el, sh, owr, oel, osh


def opposite_side(side: str) -> str:
    return "rear" if side == "lead" else "lead"


def template_vector(kp: np.ndarray, scale: float) -> np.ndarray:
    center = (kp[KP_L_SH, :2] + kp[KP_R_SH, :2]) * 0.5
    values = []
    for _name, idx in TEMPLATE_KPS:
        xy = (kp[idx, :2] - center) / scale
        values.extend([float(xy[0]), float(xy[1])])
    return np.array(values, dtype=float)


def infer_view(kp: np.ndarray) -> str:
    sw = shoulder_width(kp)
    hip = float(np.linalg.norm(kp[KP_R_HI, :2] - kp[KP_L_HI, :2]) + 1e-6)
    ratio = sw / hip
    return "side" if ratio < 0.82 else "front"


def match_template(kp: np.ndarray, side: str, view: str, templates: list[Template]) -> tuple[str | None, float, float]:
    if not templates:
        return None, float("inf"), 0.0
    vec = template_vector(kp, robust_scale(kp))
    candidates = [t for t in templates if t.side == side and (view == "auto" or t.view == view)]
    if not candidates:
        candidates = [t for t in templates if t.side == side]
    if not candidates:
        candidates = templates
    best = min(candidates, key=lambda t: float(np.linalg.norm(vec - t.vec)))
    dist = float(np.linalg.norm(vec - best.vec))
    conf = float(max(0.0, min(1.0, math.exp(-dist * 0.75))))
    return best.punch_type, dist, conf


def make_feature(hist: Deque[np.ndarray], opp_hist: Deque[np.ndarray], side: str, orthodox: bool) -> list[float] | None:
    if len(hist) < 3:
        return None
    kp = hist[-1]
    wr, el, sh = side_to_indices(side, orthodox)
    owr, _oel, _osh = side_to_indices(opposite_side(side), orthodox)
    scale = shoulder_width(kp)

    wrist = kp[wr]
    elbow = kp[el]
    shoulder = kp[sh]
    arm_ext = float(np.linalg.norm(wrist[:2] - shoulder[:2]) / scale)
    elbow_angle = angle3(shoulder, elbow, wrist)
    elbow_y = float((elbow[1] - shoulder[1]) / scale)
    elbow_lat = float(abs(elbow[0] - shoulder[0]) / scale)
    wrist_y = float((wrist[1] - shoulder[1]) / scale)
    ew = float(np.linalg.norm(elbow[:2] - wrist[:2]) / scale)

    wrists = np.array([frame[wr, :2] for frame in hist], dtype=float)
    opp_wrists = np.array([frame[owr, :2] for frame in opp_hist], dtype=float)
    diff = np.diff(wrists, axis=0)
    opp_diff = np.diff(opp_wrists, axis=0) if len(opp_wrists) >= 2 else np.zeros((0, 2))
    y = wrists[:, 1]
    wr_dx_abs = float(np.abs(diff[:, 0]).sum() / scale) if len(diff) else 0.0
    wr_dy_min = float(((y - wrist[1]) / scale).min()) if len(y) else 0.0
    wr_dy_rise = float((y[0] - y.min()) / scale) if len(y) else 0.0
    wr_peak_v = float((np.linalg.norm(diff, axis=1) / scale).max()) if len(diff) else 0.0
    opp_wr_dx_abs = float((np.linalg.norm(opp_diff, axis=1) / scale).sum()) if len(opp_diff) else 0.0
    arm_dy_rise = float((shoulder[1] - y.min()) / scale) if len(y) else 0.0

    return [
        arm_ext,
        elbow_angle,
        elbow_y,
        elbow_lat,
        wrist_y,
        ew,
        arm_dy_rise,
        wr_dx_abs,
        wr_dy_min,
        wr_dy_rise,
        wr_peak_v,
        opp_wr_dx_abs,
    ]


def normalized_kp_points(kp: np.ndarray) -> dict[int, tuple[float, float]]:
    scale = shoulder_width(kp)
    center = (kp[KP_L_SH, :2] + kp[KP_R_SH, :2]) * 0.5
    indices = [KP_NOSE, KP_L_SH, KP_R_SH, KP_L_EL, KP_R_EL, KP_L_WR, KP_R_WR, KP_L_HI, KP_R_HI]
    return {idx: (float((kp[idx, 0] - center[0]) / scale), float((kp[idx, 1] - center[1]) / scale)) for idx in indices}


def sequence_frame_metrics(kp: np.ndarray, side: str, orthodox: bool) -> list[float]:
    wr, el, sh, owr, oel, osh = sequence_indices(side, orthodox)
    scale = shoulder_width(kp)
    return [
        float(np.linalg.norm(kp[wr, :2] - kp[sh, :2]) / scale),
        angle3(kp[sh], kp[el], kp[wr]) / 180.0,
        float(np.linalg.norm(kp[wr, :2] - kp[el, :2]) / scale),
        float((kp[wr, 1] - kp[sh, 1]) / scale),
        float((kp[el, 1] - kp[sh, 1]) / scale),
        float(abs(kp[el, 0] - kp[sh, 0]) / scale),
        float(np.linalg.norm(kp[owr, :2] - kp[osh, :2]) / scale),
        angle3(kp[osh], kp[oel], kp[owr]) / 180.0,
    ]


def make_sequence_features_from_hist(hist: Deque[np.ndarray], side: str, orthodox: bool, half_window: int = 7) -> list[float] | None:
    window = half_window * 2 + 1
    if len(hist) < window:
        return None
    frames = list(hist)[-window:]
    wr, el, sh, owr, oel, osh = sequence_indices(side, orthodox)
    ordered = [wr, el, sh, owr, oel, osh, KP_NOSE, KP_L_HI, KP_R_HI]

    features: list[float] = []
    prev_punch_wr = None
    prev_opp_wr = None
    punch_wr_x, punch_wr_y = [], []
    punch_ext, punch_ew, punch_angle = [], [], []

    for frame in frames:
        points = normalized_kp_points(frame)
        for idx in ordered:
            features.extend(points[idx])

        metrics = sequence_frame_metrics(frame, side, orthodox)
        features.extend(metrics)

        pwr = points[wr]
        opp = points[owr]
        if prev_punch_wr is None:
            features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([
                pwr[0] - prev_punch_wr[0],
                pwr[1] - prev_punch_wr[1],
                opp[0] - prev_opp_wr[0],
                opp[1] - prev_opp_wr[1],
            ])
        prev_punch_wr = pwr
        prev_opp_wr = opp

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


def hook_gate(feat: list[float]) -> bool:
    """True when the wrist stays close to the elbow, which is the main hook cue."""
    arm_ext, elbow_angle, _el_y, _el_lat, _wr_y, ew, _arm_dy_rise, wr_dx_abs, _dy_min, wr_dy_rise, _peak, _opp = feat
    close_elbow_wrist = ew < 0.48
    not_fully_straight = not (arm_ext > 1.05 and elbow_angle > 138)
    enough_motion = wr_dx_abs > 0.16 or wr_dy_rise > 0.08
    not_clear_uppercut = wr_dy_rise < 0.42
    return close_elbow_wrist and not_fully_straight and enough_motion and not_clear_uppercut


def uppercut_gate(feat: list[float]) -> bool:
    """True when the wrist has a clear y-axis rise."""
    arm_ext, elbow_angle, _el_y, _el_lat, _wr_y, ew, arm_dy_rise, wr_dx_abs, _dy_min, wr_dy_rise, _peak, _opp = feat
    vertical_rise = wr_dy_rise > 0.28 or arm_dy_rise > 0.34
    not_straight_cross = not (arm_ext > 1.15 and elbow_angle > 145)
    compact_enough = ew < 0.75 or elbow_angle < 145
    return vertical_rise and not_straight_cross and compact_enough


def straight_gate(feat: list[float]) -> bool:
    arm_ext, elbow_angle, *_rest = feat
    return arm_ext > 0.88 and elbow_angle > 128


def straight_score(kp: np.ndarray, side: str, orthodox: bool) -> float:
    wr, el, sh = side_to_indices(side, orthodox)
    scale = shoulder_width(kp)
    arm_ext = float(np.linalg.norm(kp[wr, :2] - kp[sh, :2]) / scale)
    elbow_angle = angle3(kp[sh], kp[el], kp[wr])
    return arm_ext * 0.65 + max(0.0, elbow_angle - 100.0) / 90.0 * 0.35


def straight_frame_gate(kp: np.ndarray, side: str, orthodox: bool) -> bool:
    wr, el, sh = side_to_indices(side, orthodox)
    scale = shoulder_width(kp)
    arm_ext = float(np.linalg.norm(kp[wr, :2] - kp[sh, :2]) / scale)
    elbow_angle = angle3(kp[sh], kp[el], kp[wr])
    return arm_ext > 0.92 and elbow_angle > 132


def compact_frame_gate(kp: np.ndarray, side: str, orthodox: bool) -> bool:
    wr, el, sh = side_to_indices(side, orthodox)
    scale = shoulder_width(kp)
    arm_ext = float(np.linalg.norm(kp[wr, :2] - kp[sh, :2]) / scale)
    ew = float(np.linalg.norm(kp[wr, :2] - kp[el, :2]) / scale)
    elbow_angle = angle3(kp[sh], kp[el], kp[wr])
    return ew < 0.50 and not (arm_ext > 1.08 and elbow_angle > 138)


def best_extension_frame(hist: Deque[np.ndarray], side: str, orthodox: bool) -> np.ndarray:
    if not hist:
        raise ValueError("empty history")
    recent = list(hist)[-6:]
    return max(recent, key=lambda frame: straight_score(frame, side, orthodox))


def best_compact_frame(hist: Deque[np.ndarray], side: str, orthodox: bool) -> np.ndarray:
    if not hist:
        raise ValueError("empty history")
    recent = list(hist)[-6:]

    def compact_score(frame: np.ndarray) -> float:
        wr, el, sh = side_to_indices(side, orthodox)
        scale = shoulder_width(frame)
        ew = float(np.linalg.norm(frame[wr, :2] - frame[el, :2]) / scale)
        elbow_angle = angle3(frame[sh], frame[el], frame[wr])
        return ew + max(0.0, elbow_angle - 120.0) / 180.0

    return min(recent, key=compact_score)


def event_motion_summary(hist: Deque[np.ndarray], side: str, orthodox: bool) -> dict[str, float]:
    recent = list(hist)[-8:]
    if len(recent) < 3:
        return {"extension_gain": 0.0, "extension_drop": 0.0, "x_range": 0.0, "y_rise": 0.0, "min_ew": 9.0}

    wr, el, sh = side_to_indices(side, orthodox)
    ext, ew, xs, ys = [], [], [], []
    for frame in recent:
        scale = shoulder_width(frame)
        ext.append(float(np.linalg.norm(frame[wr, :2] - frame[sh, :2]) / scale))
        ew.append(float(np.linalg.norm(frame[wr, :2] - frame[el, :2]) / scale))
        xs.append(float(frame[wr, 0] / scale))
        ys.append(float(frame[wr, 1] / scale))

    return {
        "extension_gain": max(ext) - ext[0],
        "extension_drop": max(ext) - ext[-1],
        "x_range": max(xs) - min(xs),
        "y_rise": ys[0] - min(ys),
        "min_ew": min(ew),
    }


def looks_like_retraction(summary: dict[str, float]) -> bool:
    compact_lateral = summary["min_ew"] < 0.50 and summary["x_range"] > 0.24
    vertical_attack = summary["y_rise"] > 0.28
    extending_attack = summary["extension_gain"] > 0.08
    clear_drop = summary["extension_drop"] > 0.14 and summary["extension_gain"] < 0.06
    return clear_drop and not compact_lateral and not vertical_attack and not extending_attack


def classify_rule(feat: list[float], side: str) -> tuple[str, float]:
    arm_ext, elbow_angle, _el_y, _el_lat, _wr_y, ew, arm_dy_rise, wr_dx_abs, _dy_min, wr_dy_rise, _peak, _opp = feat
    if hook_gate(feat):
        return "hook", 0.82
    if uppercut_gate(feat):
        return "uppercut", 0.72
    if straight_gate(feat):
        return ("jab" if side == "lead" else "cross"), 0.78
    return ("jab" if side == "lead" else "cross"), 0.45


def combine_prediction(
    model,
    seq_model,
    seq_half_window: int,
    templates: list[Template],
    kp: np.ndarray,
    hist: Deque[np.ndarray],
    opp_hist: Deque[np.ndarray],
    side: str,
    view: str,
    orthodox: bool,
) -> tuple[str | None, float, str]:
    feat = make_feature(hist, opp_hist, side, orthodox)
    votes: Counter[str] = Counter()
    details = []

    seq_feat = make_sequence_features_from_hist(hist, side, orthodox, seq_half_window or 7) if seq_model is not None else None
    if seq_feat is not None:
        try:
            pred = str(seq_model.predict([seq_feat])[0]).lower()
            proba = 0.70
            if hasattr(seq_model, "predict_proba"):
                probs = seq_model.predict_proba([seq_feat])[0]
                proba = float(np.max(probs))
            votes[pred] += max(4, int(round(proba * 9)))
            details.append(f"seq:{pred}:{proba:.2f}")
        except Exception as exc:
            details.append(f"seq-error:{type(exc).__name__}")

    if feat is not None and hook_gate(feat):
        votes["hook"] += 7 if side == "rear" else 5
        details.append(f"ew-hook:{side}")

    if feat is not None and model is not None:
        try:
            pred = str(model.predict([feat])[0]).lower()
            proba = 0.58
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([feat])[0]
                proba = float(np.max(probs))
            if pred == "hook" and not hook_gate(feat):
                if straight_gate(feat):
                    pred = "jab" if side == "lead" else "cross"
                    votes[pred] += 2
                    details.append(f"ml->straight:{pred}:{proba:.2f}")
                else:
                    details.append(f"ml:weak-hook:{proba:.2f}")
            elif pred == "uppercut" and not uppercut_gate(feat):
                if straight_gate(feat):
                    pred = "jab" if side == "lead" else "cross"
                    votes[pred] += 2
                    details.append(f"ml->straight:{pred}:{proba:.2f}")
                else:
                    details.append(f"ml:skip-upper:{proba:.2f}")
            else:
                votes[pred] += max(1, int(round(proba * 4)))
                details.append(f"ml:{pred}:{proba:.2f}")
        except Exception:
            pass

    t_pred, _dist, t_conf = match_template(kp, side, view, templates)
    if t_pred:
        if feat is not None and t_pred == "hook" and not hook_gate(feat):
            details.append(f"tpl:skip-hook:{t_conf:.2f}")
        elif feat is not None and t_pred == "uppercut" and not uppercut_gate(feat):
            details.append(f"tpl:skip-upper:{t_conf:.2f}")
        else:
            tpl_weight = max(1, int(round(t_conf * (2 if t_pred == "hook" else 3))))
            votes[t_pred] += tpl_weight
            details.append(f"tpl:{t_pred}:{t_conf:.2f}")

    if feat is not None and hook_gate(feat):
        details.append("frame-straight:blocked-by-hook")
    elif straight_frame_gate(kp, side, orthodox):
        straight = "jab" if side == "lead" else "cross"
        votes[straight] += 3
        details.append(f"frame-straight:{straight}")

    if feat is not None:
        r_pred, r_conf = classify_rule(feat, side)
        votes[r_pred] += max(1, int(round(r_conf * 3)))
        details.append(f"rule:{r_pred}:{r_conf:.2f}")

    if not votes:
        return None, 0.0, "no-valid-punch"

    punch, weight = votes.most_common(1)[0]
    if feat is not None:
        if punch == "hook" and not hook_gate(feat):
            return None, 0.0, "reject-hook-gate"
        if punch == "uppercut" and not uppercut_gate(feat):
            return None, 0.0, "reject-upper-gate"
        if punch in ("jab", "cross") and not (straight_frame_gate(kp, side, orthodox) or straight_gate(feat)):
            return None, 0.0, "reject-weak-straight"

    confidence = min(0.98, 0.40 + weight / max(8.0, sum(votes.values()) * 1.25))
    return punch, confidence, " + ".join(details)


def analyse_feedback(kp: np.ndarray, punch_type: str, side: str, orthodox: bool) -> str:
    scale = shoulder_width(kp)
    wr, el, sh = side_to_indices(side, orthodox)
    wrist, elbow, shoulder = kp[wr], kp[el], kp[sh]
    elbow_angle = angle3(shoulder, elbow, wrist)
    arm_ext = float(np.linalg.norm(wrist[:2] - shoulder[:2]) / scale)
    elbow_y = float((elbow[1] - shoulder[1]) / scale)

    if punch_type in ("jab", "cross"):
        if arm_ext < 1.00 or elbow_angle < 145:
            return "팔을 끝까지 펴고 어깨 뒤에서 밀어주세요."
        if abs(kp[KP_NOSE, 0] - (kp[KP_L_SH, 0] + kp[KP_R_SH, 0]) * 0.5) / scale > 0.45:
            return "머리가 많이 빠집니다. 중심선을 조금 더 지켜주세요."
        return "직선 펀치 좋아요. 회수까지 빠르게 가져가세요."
    if punch_type == "hook":
        if elbow_angle > 135:
            return "훅은 팔을 너무 펴지 말고 팔꿈치 각도를 유지하세요."
        if elbow_y > 0.35:
            return "팔꿈치가 낮습니다. 어깨 높이에 가깝게 올려주세요."
        return "훅 궤도 좋아요. 반대손 가드만 유지하세요."
    if elbow_angle > 140:
        return "어퍼는 팔꿈치를 접고 아래에서 위로 짧게 올려주세요."
    return "어퍼 각도 좋아요. 턱은 당기고 몸통 회전을 더하세요."


def posture_score(kp: np.ndarray, sc: np.ndarray, dna: dict[str, float], orthodox: bool) -> tuple[int, list[str]]:
    scale = shoulder_width(kp)
    messages = []
    score = 100

    lead_wr, _lead_el, lead_sh = side_to_indices("lead", orthodox)
    rear_wr, _rear_el, rear_sh = side_to_indices("rear", orthodox)
    lead_guard = (kp[lead_wr, 1] - kp[lead_sh, 1]) / scale
    rear_guard = (kp[rear_wr, 1] - kp[rear_sh, 1]) / scale
    if lead_guard > dna["guard_l_ydiff"] + 0.22:
        score -= 14
        messages.append("앞손 가드 올리기")
    if rear_guard > dna["guard_r_ydiff"] + 0.22:
        score -= 14
        messages.append("뒷손 가드 올리기")

    shoulder_center = (kp[KP_L_SH, :2] + kp[KP_R_SH, :2]) * 0.5
    head_offset = float((kp[KP_NOSE, 1] - shoulder_center[1]) / scale)
    if abs(head_offset - dna["head_y_ratio"]) > 0.35:
        score -= 10
        messages.append("턱 당기고 시선 고정")

    if sc[KP_L_HI] > VIS_MIN and sc[KP_R_HI] > VIS_MIN:
        hip_center = (kp[KP_L_HI, :2] + kp[KP_R_HI, :2]) * 0.5
        lean = float((shoulder_center[0] - hip_center[0]) / scale)
        if abs(lean - dna["lean_forward"]) > 0.35:
            score -= 10
            messages.append("상체 기울기 안정")

    if not messages:
        messages.append("기본 자세 안정적")
    return max(0, min(100, score)), messages[:3]


def draw_text(img, text: str, xy: tuple[int, int], scale=0.65, color=(245, 245, 245), thick=1):
    if Image is not None and any(ord(ch) > 127 for ch in text):
        font_size = min(UI_FONTS, key=lambda size: abs(size - int(30 * scale)))
        font = UI_FONTS[font_size]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        shadow = (0, 0, 0)
        draw.text((xy[0] + 2, xy[1] - font_size + 2), text, font=font, fill=shadow)
        draw.text((xy[0], xy[1] - font_size), text, font=font, fill=rgb)
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_skeleton(img, kp: np.ndarray, sc: np.ndarray):
    for a, b in SKELETON:
        if sc[a] > VIS_MIN and sc[b] > VIS_MIN:
            cv2.line(img, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), (180, 180, 190), 2)
    for i in NEEDED:
        if sc[i] > VIS_MIN:
            cv2.circle(img, tuple(kp[i, :2].astype(int)), 4, (80, 230, 255), -1)


class PunchDetector:
    def __init__(
        self,
        model,
        seq_model,
        seq_half_window: int,
        templates: list[Template],
        orthodox: bool,
        view_mode: str,
        window: int,
    ):
        self.model = model
        self.seq_model = seq_model
        self.seq_half_window = seq_half_window or 7
        self.templates = templates
        self.orthodox = orthodox
        self.view_mode = view_mode
        self.window = max(8, window + 3, self.seq_half_window * 2 + 1)
        self.hist = deque(maxlen=self.window)
        self.side_hist = {"lead": deque(maxlen=self.window), "rear": deque(maxlen=self.window)}
        self.prev_wr = {"lead": None, "rear": None}
        self.active = {"lead": False, "rear": False}
        self.peak = {"lead": 0.0, "rear": 0.0}
        self.start_time = {"lead": 0.0, "rear": 0.0}
        self.last_fire = {"lead": 0.0, "rear": 0.0}
        self.last_event = 0.0
        self.last_result: PunchResult | None = None
        self.counts = Counter()
        self.pending_events = deque()
        self.rearm_required = {"lead": False, "rear": False}
        self.settle_frames = {"lead": 0, "rear": 0}

    def classify_event(self, side: str, now: float) -> PunchResult | None:
        motion = event_motion_summary(self.side_hist[side], side, self.orthodox)
        if looks_like_retraction(motion):
            return None

        compact_kp = best_compact_frame(self.side_hist[side], side, self.orthodox)
        extension_kp = best_extension_frame(self.side_hist[side], side, self.orthodox)
        use_compact = compact_frame_gate(compact_kp, side, self.orthodox)
        classify_kp = compact_kp if use_compact else extension_kp
        view = infer_view(classify_kp) if self.view_mode == "auto" else self.view_mode
        punch, conf, source = combine_prediction(
            self.model,
            self.seq_model,
            self.seq_half_window,
            self.templates,
            classify_kp,
            self.side_hist[side],
            self.side_hist[opposite_side(side)],
            side,
            view,
            self.orthodox,
        )
        if punch is None:
            return None

        result = PunchResult(punch, side, conf, source, analyse_feedback(classify_kp, punch, side, self.orthodox))
        self.counts[punch] += 1
        self.last_result = result
        self.last_event = now
        return result

    def process_pending_events(self, now: float) -> PunchResult | None:
        if not self.pending_events:
            return None
        for event in list(self.pending_events):
            event["frames_left"] -= 1
            if event["frames_left"] <= 0:
                self.pending_events.remove(event)
                result = self.classify_event(event["side"], now)
                if result is not None:
                    return result
        return None

    def update(self, kp: np.ndarray, now: float) -> PunchResult | None:
        self.hist.append(kp.copy())
        for side in ("lead", "rear"):
            self.side_hist[side].append(kp.copy())
        if len(self.hist) < 3:
            return None

        speeds = {}
        scale = shoulder_width(kp)
        for side in ("lead", "rear"):
            wr, _el, _sh = side_to_indices(side, self.orthodox)
            cur = kp[wr, :2].copy()
            prev = self.prev_wr[side]
            self.prev_wr[side] = cur
            speeds[side] = 0.0 if prev is None else float(np.linalg.norm(cur - prev) / scale)

        threshold = 0.038
        for side in ("lead", "rear"):
            if self.rearm_required[side]:
                if speeds[side] < threshold * 0.55:
                    self.settle_frames[side] += 1
                else:
                    self.settle_frames[side] = 0
                if self.settle_frames[side] >= 5:
                    self.rearm_required[side] = False
                    self.settle_frames[side] = 0

        pending_result = self.process_pending_events(now)
        if pending_result is not None:
            return pending_result
        if self.pending_events:
            return None

        fired: list[str] = []
        fired_peak = {}
        for side in ("lead", "rear"):
            if self.rearm_required[side]:
                self.active[side] = False
                self.peak[side] = 0.0
                continue
            speed = speeds[side]
            other = speeds[opposite_side(side)]
            cooldown_ok = now - self.last_fire[side] > 0.62 and now - self.last_event > 0.48
            if speed > threshold and cooldown_ok:
                if not self.active[side]:
                    self.start_time[side] = now
                self.active[side] = True
                self.peak[side] = max(self.peak[side], speed)
            timed_out = self.active[side] and now - self.start_time[side] > 0.20
            falling = self.active[side] and speed < threshold * 0.80
            if self.active[side] and (falling or timed_out):
                if self.peak[side] > max(0.058, other * 0.95):
                    fired.append(side)
                    fired_peak[side] = self.peak[side]
                self.active[side] = False
                self.peak[side] = 0.0

        if not fired:
            return None
        side = max(fired, key=lambda s: fired_peak.get(s, 0.0))
        motion = event_motion_summary(self.side_hist[side], side, self.orthodox)
        if looks_like_retraction(motion):
            return None
        self.last_fire[side] = now
        self.last_event = now
        self.rearm_required[side] = True
        self.settle_frames[side] = 0
        delay = self.seq_half_window if self.seq_model is not None else 0
        if delay > 0:
            self.pending_events.append({"side": side, "frames_left": delay})
            return None
        return self.classify_event(side, now)


def build_pose_model():
    try:
        from rtmlib import RTMO
    except ImportError as exc:
        raise SystemExit("rtmlib가 필요합니다. 현재 환경에서 `pip install rtmlib onnxruntime` 후 다시 실행하세요.") from exc

    try:
        import onnxruntime as ort

        device = "dml" if "DmlExecutionProvider" in ort.get_available_providers() else "cpu"
    except Exception:
        device = "cpu"
    url = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
        "rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip"
    )
    print(f"Loading RTMO pose model ({device})...")
    return RTMO(url, backend="onnxruntime", device=device)


def main() -> int:
    parser = argparse.ArgumentParser(description="LIM pose boxing coach")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--stance", choices=["orthodox", "southpaw"], default="orthodox")
    parser.add_argument("--view", choices=["auto", "front", "side"], default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    model, _features, model_window = load_classifier(os.path.join(BASE_DIR, "lim_punch_model.pkl"))
    seq_model, seq_half_window, seq_feature_count = load_sequence_classifier(os.path.join(BASE_DIR, "lim_punch_seq_model.pkl"))
    templates = load_templates(os.path.join(BASE_DIR, "punch_templates.csv"))
    dna = load_pose_dna(os.path.join(BASE_DIR, "LIM_DNA.csv"))

    print(f"Punch model: {'loaded' if model is not None else 'missing, rule/template only'}")
    print(
        "Sequence model: "
        + (f"loaded (+/-{seq_half_window}, {seq_feature_count} features)" if seq_model is not None else "missing")
    )
    print(f"Templates: {len(templates)}")
    print(f"View: {args.view}, stance: {args.stance}")

    pose_model = build_pose_model()
    cap = cv2.VideoCapture(parse_source(args.source))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    orthodox = args.stance == "orthodox"
    detector = PunchDetector(model, seq_model, seq_half_window, templates, orthodox, args.view, model_window)
    win = "LIM Boxing Coach"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    flash_until = 0.0
    fps_t = time.time()
    fps = 0.0
    frame_n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        frame_n += 1
        if now - fps_t >= 0.5:
            fps = frame_n / (now - fps_t)
            fps_t = now
            frame_n = 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kps, scores = pose_model(rgb)
        display = frame.copy()

        if len(kps) > 0:
            kp = kps[0].astype(float)
            sc = scores[0].astype(float)
            pose_ok = all(sc[i] > VIS_MIN for i in NEEDED)
            if pose_ok:
                result = detector.update(kp, now)
                if result:
                    flash_until = now + 0.45
                draw_skeleton(display, kp, sc)
                p_score, p_messages = posture_score(kp, sc, dna, orthodox)
                view_now = infer_view(kp) if args.view == "auto" else args.view

                draw_text(display, f"POSTURE {p_score}/100  VIEW {view_now.upper()}  FPS {fps:.1f}", (24, 36), 0.72)
                for i, msg in enumerate(p_messages):
                    draw_text(display, msg, (24, 68 + i * 28), 0.62, (210, 245, 210))
            else:
                draw_text(display, "포즈를 더 잘 보이게 서주세요", (24, 42), 0.75, (70, 210, 255))
        else:
            draw_text(display, "사람을 찾는 중", (24, 42), 0.75, (70, 210, 255))

        if detector.last_result:
            r = detector.last_result
            color = PUNCH_COLORS.get(r.punch_type, (255, 255, 255))
            label = f"{r.punch_type.upper()}  {r.side}  {r.confidence * 100:.0f}%"
            if now < flash_until:
                cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), color, 10)
            draw_text(display, label, (24, display.shape[0] - 102), 1.0, color, 2)
            draw_text(display, r.feedback, (24, display.shape[0] - 64), 0.68, (245, 245, 245))

        x = display.shape[1] - 330
        y = 32
        cv2.rectangle(display, (x - 16, y - 28), (display.shape[1] - 20, y + 156), (20, 20, 24), -1)
        draw_text(display, "COUNT", (x, y), 0.70)
        for i, name in enumerate(["jab", "cross", "hook", "uppercut"]):
            draw_text(display, f"{name.upper():8s} {detector.counts[name]:3d}", (x, y + 34 + i * 28), 0.62, PUNCH_COLORS[name])
        draw_text(display, "Q quit  R reset  D stance", (24, display.shape[0] - 22), 0.55, (200, 200, 205))

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            detector.counts.clear()
            detector.last_result = None
        if key == ord("d"):
            orthodox = not orthodox
            detector.orthodox = orthodox

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
