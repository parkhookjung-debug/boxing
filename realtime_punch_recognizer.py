from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import Counter, deque

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

import lim_punch_features as F


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

VIS_MIN = 0.25
NEEDED = [F.NOSE, F.L_SH, F.R_SH, F.L_EL, F.R_EL, F.L_WR, F.R_WR, F.L_HI, F.R_HI]
LEFT_LEAD = True

PUNCH_LABELS = {
    "jab": "JAB",
    "cross": "CROSS",
    "hook": "HOOK",
    "uppercut": "UPPER",
}

CALIB_LABELS = {
    "jab": "JAB",
    "cross": "CROSS",
    "lead_hook": "LEFT HOOK",
    "rear_hook": "RIGHT HOOK",
    "lead_uppercut": "LEFT UPPER",
    "rear_uppercut": "RIGHT UPPER",
}

DETAIL_TO_PUNCH = {
    "jab": "jab",
    "cross": "cross",
    "lead_hook": "hook",
    "rear_hook": "hook",
    "lead_uppercut": "uppercut",
    "rear_uppercut": "uppercut",
}

DETAIL_EXPECTED_SIDE = {
    "jab": "lead",
    "cross": "rear",
    "lead_hook": "lead",
    "rear_hook": "rear",
    "lead_uppercut": "lead",
    "rear_uppercut": "rear",
}

PUNCH_COLORS = {
    "jab": (0, 220, 255),
    "cross": (255, 145, 40),
    "hook": (220, 80, 255),
    "uppercut": (90, 235, 90),
}

SKELETON = [
    (F.NOSE, F.L_SH), (F.NOSE, F.R_SH),
    (F.L_SH, F.R_SH), (F.L_SH, F.L_EL), (F.L_EL, F.L_WR),
    (F.R_SH, F.R_EL), (F.R_EL, F.R_WR),
    (F.L_SH, F.L_HI), (F.R_SH, F.R_HI), (F.L_HI, F.R_HI),
    (F.L_HI, 13), (13, 15), (F.R_HI, 14), (14, 16),
]

LR_SWAP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def swap_left_right(kp, scores):
    """Swap COCO left/right labels while keeping screen coordinates unchanged."""
    return kp[LR_SWAP_IDX].copy(), scores[LR_SWAP_IDX].copy()


def load_font(size: int):
    if ImageFont is None:
        return None
    for path in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONTS = {size: load_font(size) for size in (14, 16, 18, 22, 30, 48)}


def draw_texts(frame, texts):
    if not texts:
        return
    if Image is None:
        for text, xy, size, color in texts:
            cv2.putText(frame, text, (int(xy[0]), int(xy[1] + size)),
                        cv2.FONT_HERSHEY_SIMPLEX, size / 28, color, 1, cv2.LINE_AA)
        return
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(pil)
    for text, xy, size, color in texts:
        font = FONTS.get(size) or FONTS[16]
        rgb = (color[2], color[1], color[0])
        drawer.text((xy[0] + 2, xy[1] + 2), text, font=font, fill=(0, 0, 0))
        drawer.text(xy, text, font=font, fill=rgb)
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def panel(frame, x, y, w, h, alpha=0.58):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 18, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 76), 1, cv2.LINE_AA)


def bar(frame, x, y, w, h, value, color):
    value = max(0.0, min(1.0, float(value)))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (52, 52, 56), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * value), y + h), color, -1)


def load_window_model(path):
    if not os.path.exists(path):
        raise SystemExit(f"Missing model: {path}\nRun LIM_train_window.py first.")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], list(bundle["classes"]), bundle.get("meta", {})


def build_pose_model():
    try:
        from rtmlib import RTMO
        import rtmlib.tools.base as rtmlib_base
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install rtmlib onnxruntime") from exc

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            rtmlib_base.RTMLIB_SETTINGS["onnxruntime"]["dml"] = "DmlExecutionProvider"
            device = "dml"
        else:
            device = "cpu"
    except Exception:
        device = "cpu"

    url = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
        "rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip"
    )
    print(f"Loading RTMO pose model ({device})...")
    return RTMO(url, backend="onnxruntime", device=device)


class WristImputer:
    def __init__(self):
        self.prev = None
        self.prev_scores = None

    def apply(self, kp, scores):
        kp = np.asarray(kp, dtype=float).copy()
        scores = np.asarray(scores, dtype=float).copy()
        if self.prev is None:
            self.prev = kp.copy()
            self.prev_scores = scores.copy()
            return kp, scores

        for wrist, elbow, shoulder in (
            (F.L_WR, F.L_EL, F.L_SH),
            (F.R_WR, F.R_EL, F.R_SH),
        ):
            if scores[wrist] >= VIS_MIN:
                continue
            if self.prev_scores[wrist] >= VIS_MIN:
                # Keep fast rear-hand punches alive for a few blurred/occluded frames.
                prev_vec = self.prev[wrist, :2] - self.prev[elbow, :2]
                cur_vec = kp[elbow, :2] - kp[shoulder, :2]
                if np.linalg.norm(cur_vec) > 1e-6:
                    prev_len = np.linalg.norm(prev_vec)
                    cur_dir = cur_vec / np.linalg.norm(cur_vec)
                    kp[wrist, :2] = kp[elbow, :2] + cur_dir * prev_len
                else:
                    kp[wrist, :2] = self.prev[wrist, :2]
                scores[wrist] = max(scores[wrist], 0.26)

        self.prev = kp.copy()
        self.prev_scores = scores.copy()
        return kp, scores


class HandIdentityLock:
    """Keep user left/right hand identity stable across temporary pose label swaps."""

    def __init__(self, max_switch_cost=0.55):
        self.prev_wrists = None
        self.max_switch_cost = max_switch_cost

    def reset(self):
        self.prev_wrists = None

    def apply(self, kp, scores):
        kp = np.asarray(kp, dtype=float).copy()
        scores = np.asarray(scores, dtype=float).copy()
        if self.prev_wrists is None:
            self.prev_wrists = {
                "left": kp[F.L_WR, :2].copy(),
                "right": kp[F.R_WR, :2].copy(),
            }
            return kp, scores

        scale = F.body_scale(kp)
        keep = (
            np.linalg.norm(kp[F.L_WR, :2] - self.prev_wrists["left"]) +
            np.linalg.norm(kp[F.R_WR, :2] - self.prev_wrists["right"])
        ) / scale
        swap = (
            np.linalg.norm(kp[F.R_WR, :2] - self.prev_wrists["left"]) +
            np.linalg.norm(kp[F.L_WR, :2] - self.prev_wrists["right"])
        ) / scale

        if swap + 0.08 < keep and swap < self.max_switch_cost:
            kp = kp[LR_SWAP_IDX].copy()
            scores = scores[LR_SWAP_IDX].copy()

        self.prev_wrists["left"] = 0.75 * self.prev_wrists["left"] + 0.25 * kp[F.L_WR, :2]
        self.prev_wrists["right"] = 0.75 * self.prev_wrists["right"] + 0.25 * kp[F.R_WR, :2]
        return kp, scores


class PunchRecognizer:
    def __init__(self, model, classes, enter_thr=0.50, min_event_gap=0.72):
        self.model = model
        self.idx_map = [classes.index(c) for c in F.CLASSES]
        self.buf = deque(maxlen=F.WINDOW)
        self.tracker = F.PunchEventTracker(enter_thr=enter_thr)
        self.min_event_gap = min_event_gap
        self.frame_no = 0
        self.counts = Counter()
        self.live_probs = np.zeros(len(F.CLASSES), dtype=float)
        self.last_event = None
        self.last_event_time = 0.0

    def reset(self):
        self.buf.clear()
        self.tracker = F.PunchEventTracker(enter_thr=self.tracker.enter_thr)
        self.frame_no = 0
        self.counts.clear()
        self.live_probs[:] = 0
        self.last_event = None
        self.last_event_time = 0.0

    def update(self, kp, now, emit=True):
        self.frame_no += 1
        self.buf.append(np.asarray(kp[:, :2], dtype=float))
        if len(self.buf) < F.WINDOW:
            return None

        feat = F.window_feat(list(self.buf), F.HALF)
        probs = self.model.predict_proba([feat])[0][self.idx_map]
        if F.wrist_gate_speed(list(self.buf)) < F.MOTION_GATE_THR:
            probs = np.zeros(len(F.CLASSES), dtype=float)
            probs[0] = 1.0
        self.live_probs = probs
        if not emit:
            return None

        ev = self.tracker.push(self.frame_no - F.HALF, probs)
        if ev:
            return ev if self.register_event(ev, now) else None
        return None

    def register_event(self, ev, now):
        if now - self.last_event_time < self.min_event_gap:
            return False
        self.counts[ev["punch_type"]] += 1
        self.last_event = ev
        self.last_event_time = now
        return True


class FastPunchAssist:
    """Low-latency punch trigger that emits once when the punch reaches a stop."""

    CALIB_SEQUENCE = (
        "jab",
        "cross",
        "lead_hook",
        "rear_hook",
        "lead_uppercut",
        "rear_uppercut",
    )
    SAMPLES_PER_TEMPLATE = 5

    def __init__(self, cooldown=1.05):
        self.hist = deque(maxlen=12)
        self.last_emit = 0.0
        self.cooldown = cooldown
        self.active = {"lead": None, "rear": None}
        self.guard = None
        self.calib_samples = []
        self.calibrating = True
        self.calib_countdown_until = 0.0
        self.calib_stage = "guard"
        self.calib_index = 0
        self.templates = {name: [] for name in self.CALIB_SEQUENCE}

    def reset(self):
        self.hist.clear()
        self.last_emit = 0.0
        self.active = {"lead": None, "rear": None}

    def start_calibration(self, now=None):
        self.guard = None
        self.calib_samples = []
        self.calibrating = True
        self.calib_countdown_until = (now + 3.0) if now is not None else 0.0
        self.calib_stage = "guard"
        self.calib_index = 0
        self.templates = {name: [] for name in self.CALIB_SEQUENCE}
        self.reset()

    def update_calibration(self, kp, scores, now, duration_frames=6):
        if any(scores[i] < VIS_MIN for i in (F.L_SH, F.R_SH, F.L_EL, F.R_EL, F.L_WR, F.R_WR, F.L_HI, F.R_HI)):
            return False
        if self.calib_stage == "guard":
            if now < self.calib_countdown_until:
                return False
            self.calib_samples.append(np.asarray(kp[:, :2], dtype=float).copy())
            if len(self.calib_samples) < duration_frames:
                return False

            arr = np.asarray(self.calib_samples[-duration_frames:], dtype=float)
            guard_kp = np.median(arr, axis=0)
            scale = F.body_scale(guard_kp)
            self.guard = {
                "kp": guard_kp,
                "scale": scale,
                "lead_wr": guard_kp[F.L_WR].copy(),
                "rear_wr": guard_kp[F.R_WR].copy(),
                "lead_sh": guard_kp[F.L_SH].copy(),
                "rear_sh": guard_kp[F.R_SH].copy(),
            }
            self.calib_countdown_until = 0.0
            self.calib_stage = self.CALIB_SEQUENCE[0]
            self.hist.clear()
            self.active = {"lead": None, "rear": None}
            return True

        ev = self._track_motion(kp, scores, now, forced_label=self.calib_stage)
        if ev:
            self.templates[self.calib_stage].append(np.asarray(ev["feature"], dtype=float))
            if len(self.templates[self.calib_stage]) >= self.SAMPLES_PER_TEMPLATE:
                self.calib_index += 1
                self.hist.clear()
                self.active = {"lead": None, "rear": None}
                if self.calib_index >= len(self.CALIB_SEQUENCE):
                    self.calibrating = False
                    self.calib_stage = "ready"
                else:
                    self.calib_stage = self.CALIB_SEQUENCE[self.calib_index]
            return True
        return False

    def calibration_status(self, now):
        if not self.calibrating:
            return "ready", "C recalibrate full"
        if self.calib_stage == "guard":
            remain = max(0.0, self.calib_countdown_until - now)
            msg = f"guard in {remain:0.1f}s" if remain > 0 else "hold guard now"
            return msg, f"samples {len(self.calib_samples):02d}/06"
        count = len(self.templates[self.calib_stage])
        return (
            f"{CALIB_LABELS[self.calib_stage]} x{count + 1}/{self.SAMPLES_PER_TEMPLATE}",
            "punch and hold stop",
        )

    def update(self, kp, scores, now):
        if self.calibrating or self.guard is None:
            return None
        return self._track_motion(kp, scores, now)

    def _track_motion(self, kp, scores, now, forced_label=None):
        if any(scores[i] < VIS_MIN for i in (F.L_SH, F.R_SH, F.L_EL, F.R_EL, F.L_WR, F.R_WR)):
            return None
        self.hist.append(np.asarray(kp[:, :2], dtype=float).copy())
        if len(self.hist) < 3 or now - self.last_emit < self.cooldown:
            return None

        s = F.body_scale(kp)
        emitted = []
        for side, wrist, elbow, shoulder in (
            ("lead", F.L_WR, F.L_EL, F.L_SH),
            ("rear", F.R_WR, F.R_EL, F.R_SH),
        ):
            p0 = self.hist[-3][wrist]
            p1 = self.hist[-1][wrist]
            sh0 = self.hist[-3][shoulder]
            sh1 = self.hist[-1][shoulder]
            rel0 = p0 - sh0
            rel1 = p1 - sh1
            d = (rel1 - rel0) / s
            speed = float(np.linalg.norm(d))

            state = self.active[side]
            if state is None:
                if speed >= 0.025:
                    wrist_path = [p[wrist].copy() for p in self.hist]
                    self.active[side] = {
                        "side": side,
                        "start_kp": self.hist[-3].copy(),
                        "last_kp": np.asarray(kp[:, :2], dtype=float).copy(),
                        "started": now,
                        "path_gain": speed,
                        "max_speed": speed,
                        "min_x": min(float(p[0]) for p in wrist_path),
                        "max_x": max(float(p[0]) for p in wrist_path),
                        "min_y": min(float(p[1]) for p in wrist_path),
                        "max_y": max(float(p[1]) for p in wrist_path),
                        "stop_frames": 0,
                    }
                continue

            state["last_kp"] = np.asarray(kp[:, :2], dtype=float).copy()
            state["path_gain"] += speed
            state["max_speed"] = max(state["max_speed"], speed)
            state["min_x"] = min(state["min_x"], float(p1[0]))
            state["max_x"] = max(state["max_x"], float(p1[0]))
            state["min_y"] = min(state["min_y"], float(p1[1]))
            state["max_y"] = max(state["max_y"], float(p1[1]))
            stopped = speed < 0.025
            state["stop_frames"] = state["stop_frames"] + 1 if stopped else 0
            timed_out = now - state["started"] > 1.25
            live_path_horiz = (state["max_x"] - state["min_x"]) / s
            live_path_vert = (state["max_y"] - state["min_y"]) / s
            meaningful_motion = (
                (state["path_gain"] > 0.16 and state["max_speed"] > 0.050) or
                live_path_horiz > 0.10 or
                live_path_vert > 0.12
            )
            if meaningful_motion and (state["stop_frames"] >= 3 or timed_out):
                ev = self.classify_stopped_pose(state, kp, s, forced_label=forced_label)
                if ev:
                    emitted.append(ev)
                self.active[side] = None
            elif not meaningful_motion and now - state["started"] > 0.55:
                self.active[side] = None

        if emitted:
            best = max(emitted, key=lambda ev: ev["confidence"])
            self.last_emit = now
            return best
        return None

    @staticmethod
    def guard_pose(kp, wrist, elbow, shoulder, scale):
        wrist_pt = kp[wrist]
        elbow_pt = kp[elbow]
        shoulder_pt = kp[shoulder]
        rel = (wrist_pt - shoulder_pt) / scale
        arm_len = float(np.linalg.norm(rel))
        elbow_angle = angle_deg(shoulder_pt, elbow_pt, wrist_pt)
        near_head_box = abs(float(rel[0])) < 0.62 and -0.78 <= float(rel[1]) <= 0.52
        return near_head_box and arm_len < 0.82 and elbow_angle < 132

    def classify_stopped_pose(self, state, kp, scale, forced_label=None):
        side = state["side"]
        if forced_label and DETAIL_EXPECTED_SIDE.get(forced_label) != side:
            return None
        wrist, elbow, shoulder = (
            (F.L_WR, F.L_EL, F.L_SH) if side == "lead"
            else (F.R_WR, F.R_EL, F.R_SH)
        )
        stop = np.asarray(kp[:, :2], dtype=float)
        stop_wr = stop[wrist]
        stop_el = stop[elbow]
        stop_sh = stop[shoulder]
        nose = stop[F.NOSE]
        sh_center = (stop[F.L_SH] + stop[F.R_SH]) * 0.5
        guard_wr = self.guard["lead_wr"] if side == "lead" else self.guard["rear_wr"]
        guard_sh = self.guard["lead_sh"] if side == "lead" else self.guard["rear_sh"]

        path_horiz = (state["max_x"] - state["min_x"]) / scale
        path_vert = (state["max_y"] - state["min_y"]) / scale

        if self.guard_pose(stop, wrist, elbow, shoulder, scale) and state["path_gain"] < 0.48:
            return None

        move = (stop_wr - guard_wr) / scale
        shoulder_move = (stop_sh - guard_sh) / scale
        rel_move = move - shoulder_move
        rel = (stop_wr - stop_sh) / scale
        ext = float(np.linalg.norm(rel))
        elbow_angle = angle_deg(stop_sh, stop_el, stop_wr)
        up = -float(rel_move[1])
        down_then_up = self.path_dipped_then_rose(state, wrist, guard_wr, scale)
        dipped = (state["max_y"] - float(guard_wr[1])) / scale > 0.06
        rose_from_dip = (state["max_y"] - stop_wr[1]) / scale > 0.07
        horiz = abs(float(rel_move[0]))
        vert = abs(float(rel_move[1]))
        wrist_height = float((stop_sh[1] - stop_wr[1]) / scale)
        wrist_to_center = abs(float((stop_wr[0] - sh_center[0]) / scale))
        hand_near_head = abs(float((stop_wr[1] - nose[1]) / scale)) < 0.80
        feature = self.punch_feature(
            side=side,
            rel_move=rel_move,
            ext=ext,
            elbow_angle=elbow_angle,
            up=up,
            horiz=horiz,
            vert=vert,
            path_horiz=path_horiz,
            path_vert=path_vert,
            path_gain=state["path_gain"],
            max_speed=state["max_speed"],
            dipped=dipped,
            rose_from_dip=rose_from_dip,
        )
        confidence = min(0.98, 0.54 + state["path_gain"] * 0.35 + state["max_speed"] * 1.2)

        if forced_label:
            detail = forced_label
            punch = DETAIL_TO_PUNCH[detail]
            source = "calibration_sample"
        elif self.templates_ready():
            detail, confidence = self.classify_by_template(feature)
            punch = DETAIL_TO_PUNCH[detail]
            source = "personal_template"
        else:
            detail = None
            punch = None
            source = "stopped_pose"

        if punch is None:
            if (up > 0.13 or down_then_up or (dipped and rose_from_dip)) and elbow_angle < 165 and path_vert > 0.16:
                punch = "uppercut"
                detail = "lead_uppercut" if side == "lead" else "rear_uppercut"
                source = "stopped_pose_upper"
            elif elbow_angle < 170 and (horiz > 0.07 or path_horiz > 0.11) and path_horiz >= path_vert * 0.45:
                punch = "hook"
                detail = "lead_hook" if side == "lead" else "rear_hook"
                source = "stopped_pose_hook"
            elif (elbow_angle >= 118 or ext > 0.62) and hand_near_head:
                punch = "jab" if side == "lead" else "cross"
                detail = punch
                source = "stopped_pose_straight"
            elif elbow_angle < 135 and wrist_to_center < 0.75 and hand_near_head:
                punch = "hook"
                detail = "lead_hook" if side == "lead" else "rear_hook"
                source = "stopped_pose_compact_hook"

        if not punch:
            return None
        return {
            "frame": 0,
            "class_idx": F.CLASSES.index(punch),
            "punch_type": punch,
            "detail_type": detail,
            "confidence": confidence,
            "run_len": 1,
            "source": source,
            "feature": feature,
        }

    @staticmethod
    def punch_feature(**values):
        side_code = 0.0 if values["side"] == "lead" else 1.0
        return np.asarray([
            side_code,
            float(values["rel_move"][0]),
            float(values["rel_move"][1]),
            float(values["ext"]),
            float(values["elbow_angle"]) / 180.0,
            float(values["up"]),
            float(values["horiz"]),
            float(values["vert"]),
            float(values["path_horiz"]),
            float(values["path_vert"]),
            float(values["path_gain"]),
            float(values["max_speed"]),
            1.0 if values["dipped"] else 0.0,
            1.0 if values["rose_from_dip"] else 0.0,
        ], dtype=float)

    def templates_ready(self):
        return all(len(self.templates[name]) >= self.SAMPLES_PER_TEMPLATE for name in self.CALIB_SEQUENCE)

    def classify_by_template(self, feature):
        weights = np.asarray([0.75, 1.2, 1.2, 0.8, 1.0, 1.3, 1.4, 1.2,
                              1.6, 1.6, 0.9, 1.0, 1.4, 1.4], dtype=float)
        best_name, best_dist = None, float("inf")
        for name in self.CALIB_SEQUENCE:
            center = np.mean(np.asarray(self.templates[name], dtype=float), axis=0)
            dist = float(np.linalg.norm((feature - center) * weights))
            if dist < best_dist:
                best_name, best_dist = name, dist
        confidence = max(0.45, min(0.98, 1.0 - best_dist / 4.0))
        return best_name, confidence

    def path_dipped_then_rose(self, state, wrist, guard_wr, scale):
        if len(self.hist) < 4:
            return False
        ys = np.asarray([p[wrist, 1] for p in self.hist], dtype=float)
        guard_y = float(guard_wr[1])
        dipped = (ys.max() - guard_y) / scale > 0.08
        rose = (ys.max() - ys[-1]) / scale > 0.10
        return bool(dipped and rose)


def draw_skeleton(frame, kp, scores):
    for a, b in SKELETON:
        if a < len(kp) and b < len(kp) and scores[a] > VIS_MIN and scores[b] > VIS_MIN:
            cv2.line(frame, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)),
                     (170, 174, 184), 2, cv2.LINE_AA)
    for idx, color in ((F.L_WR, (0, 220, 255)), (F.R_WR, (255, 145, 40))):
        if scores[idx] > VIS_MIN:
            cv2.circle(frame, tuple(kp[idx, :2].astype(int)), 8, color, -1, cv2.LINE_AA)
    for i in range(min(17, len(kp))):
        if scores[i] > VIS_MIN:
            cv2.circle(frame, tuple(kp[i, :2].astype(int)), 3, (235, 235, 235), -1, cv2.LINE_AA)


def angle_deg(a, b, c):
    ba = a - b
    bc = c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / den, -1.0, 1.0)))


def stance_quality(kp, scores):
    if any(scores[i] < VIS_MIN for i in [F.L_SH, F.R_SH, F.L_HI, F.R_HI, F.L_WR, F.R_WR]):
        return 0, "몸 전체를 화면에 넣어주세요"
    scale = F.body_scale(kp)
    shoulder_y = (kp[F.L_SH, 1] + kp[F.R_SH, 1]) * 0.5
    lead_guard = (shoulder_y - kp[F.L_WR, 1]) / scale
    rear_guard = (shoulder_y - kp[F.R_WR, 1]) / scale
    shoulder_width = np.linalg.norm(kp[F.R_SH, :2] - kp[F.L_SH, :2])
    torso = np.linalg.norm(((kp[F.L_SH, :2] + kp[F.R_SH, :2]) * 0.5) -
                           ((kp[F.L_HI, :2] + kp[F.R_HI, :2]) * 0.5)) + 1e-6
    view = shoulder_width / torso

    score = 0
    score += 35 if 0.05 <= lead_guard <= 0.65 else 18 if -0.05 <= lead_guard <= 0.85 else 0
    score += 35 if 0.08 <= rear_guard <= 0.75 else 18 if -0.05 <= rear_guard <= 0.95 else 0
    score += 30 if 0.45 <= view <= 0.95 else 15 if 0.30 <= view <= 1.15 else 0

    if view < 0.30:
        msg = "너무 측면입니다. 카메라를 조금 정면으로"
    elif view > 1.15:
        msg = "너무 정면입니다. 앞발 쪽 30-45도 권장"
    elif rear_guard < 0.05:
        msg = "오른손 뒷손 가드가 낮습니다"
    elif lead_guard < 0.05:
        msg = "왼손 앞손 가드가 낮습니다"
    else:
        msg = "45도 각도와 가드가 좋습니다"
    return score, msg


def draw_hand_labels(frame, kp, scores, texts):
    if scores[F.L_WR] > VIS_MIN:
        x, y = kp[F.L_WR, :2].astype(int)
        texts.append(("LEFT LEAD", (x + 12, y - 26), 16, PUNCH_COLORS["jab"]))
    if scores[F.R_WR] > VIS_MIN:
        x, y = kp[F.R_WR, :2].astype(int)
        texts.append(("RIGHT REAR", (x + 12, y - 26), 16, PUNCH_COLORS["cross"]))


def draw_ui(frame, recognizer, fast_assist, fps, guide_score, guide_msg, flash_until, now, mirror):
    h, w = frame.shape[:2]
    texts = []

    panel(frame, 16, 16, 230, 170)
    texts.append(("REAL-TIME PUNCH", (30, 26), 16, (185, 185, 190)))
    for i, name in enumerate(("jab", "cross", "hook", "uppercut")):
        y = 58 + i * 28
        texts.append((PUNCH_LABELS[name], (30, y), 18, PUNCH_COLORS[name]))
        texts.append((f"{recognizer.counts[name]:3d}", (176, y), 22, (245, 245, 245)))

    panel(frame, w - 310, 16, 294, 136)
    title = "CALIBRATING GUARD" if fast_assist.calibrating else "CAMERA GUIDE"
    texts.append((title, (w - 294, 28), 16, (185, 185, 190)))
    bar(frame, w - 294, 62, 262, 8, guide_score / 100.0,
        (90, 230, 120) if guide_score >= 75 else (70, 180, 255))
    texts.append((f"{guide_score:3d}/100", (w - 294, 78), 22, (245, 245, 245)))
    if fast_assist.calibrating:
        msg, sub = fast_assist.calibration_status(now)
        msg_color = (70, 190, 255)
    else:
        msg = guide_msg
        sub = "C full calibration"
        msg_color = (90, 230, 120) if guide_score >= 75 else (70, 180, 255)
    texts.append((msg, (w - 294, 108), 16, msg_color))
    texts.append((sub, (w - 294, 130), 14, (160, 160, 166)))

    probs = recognizer.live_probs
    panel(frame, 16, h - 78, 510, 62)
    texts.append((f"{fps:4.1f} FPS   mirror:{'on' if mirror else 'off'}", (30, h - 66), 14, (170, 170, 176)))
    for i, name in enumerate(("jab", "cross", "hook", "uppercut")):
        x = 30 + i * 118
        ci = F.CLASSES.index(name)
        texts.append((PUNCH_LABELS[name], (x, h - 42), 14, (190, 190, 196)))
        bar(frame, x, h - 20, 96, 7, probs[ci], PUNCH_COLORS[name])

    ev = recognizer.last_event
    if ev and now < flash_until:
        name = ev["punch_type"]
        detail = ev.get("detail_type")
        label = CALIB_LABELS.get(detail, PUNCH_LABELS[name])
        color = PUNCH_COLORS.get(name, (255, 255, 255))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 10, cv2.LINE_AA)
        texts.append((label, (38, h - 148), 48, color))
        texts.append((f"{ev['confidence'] * 100:.0f}%", (250, h - 124), 30, (245, 245, 245)))

    texts.append(("Q quit  R reset  C guard  M mirror", (w // 2 - 160, h - 28), 14, (170, 170, 176)))
    return texts


def choose_person(kps, scores):
    if len(kps) == 0:
        return None, None
    best = max(range(len(kps)), key=lambda i: float(np.mean(scores[i][:17])))
    return kps[best].astype(float), scores[best].astype(float)


def main():
    parser = argparse.ArgumentParser(description="Real-time boxing punch recognizer")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--enter-thr", type=float, default=0.35)
    parser.add_argument("--no-mirror", action="store_true", help="disable webcam mirror view")
    parser.add_argument("--no-swap-hands", action="store_true",
                        help="do not swap left/right pose labels in mirror mode")
    args = parser.parse_args()

    model, classes, meta = load_window_model(os.path.join(BASE_DIR, "lim_window_model.pkl"))
    print(f"Punch model loaded. classes={classes}")
    if meta:
        print(f"LOVO macro-F1={meta.get('lovo_macro_f1')} samples={meta.get('samples')}")

    pose_model = build_pose_model()
    recognizer = PunchRecognizer(model, classes, enter_thr=args.enter_thr)
    fast_assist = FastPunchAssist()
    fast_assist.start_calibration(time.time())
    smoother = F.KeypointFilter(n_kp=17)
    imputer = WristImputer()
    hand_lock = HandIdentityLock()

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    mirror = isinstance(src, int) and not args.no_mirror
    win = "Boxing Punch Recognizer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    prev_t = time.time()
    fps_t = prev_t
    fps_n = 0
    fps = 0.0
    flash_until = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        now = time.time()
        dt = max(1.0 / 120.0, min(0.15, now - prev_t))
        prev_t = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kps, scores = pose_model(rgb)
        kp, sc = choose_person(kps, scores)

        status = []
        guide_score, guide_msg = 0, "사람을 찾는 중입니다"
        if kp is not None:
            if mirror and not args.no_swap_hands:
                kp, sc = swap_left_right(kp[:17], sc[:17])
            kp, sc = imputer.apply(kp[:17], sc[:17])
            kp, sc = hand_lock.apply(kp, sc)
            smooth_xy = smoother.apply(kp, dt)
            kp[:, :2] = smooth_xy
            guide_score, guide_msg = stance_quality(kp, sc)

            if fast_assist.calibrating:
                fast_assist.update_calibration(kp, sc, now)
            elif all(sc[i] > VIS_MIN for i in NEEDED):
                ev = recognizer.update(kp, now, emit=False)
                fast_ev = fast_assist.update(kp, sc, now)
                if fast_ev and (not ev):
                    recognizer.register_event(fast_ev, now)
                    ev = fast_ev
                if ev:
                    flash_until = now + 0.50
            else:
                status.append(("손목/팔꿈치가 가려졌습니다. 오른손 뒷손을 얼굴 옆에 보여주세요.",
                               (frame.shape[1] // 2 - 260, 46), 18, (70, 190, 255)))
            draw_skeleton(frame, kp, sc)
        else:
            status.append(("전신이 보이도록 한 걸음 물러서세요.",
                           (frame.shape[1] // 2 - 150, 46), 18, (70, 190, 255)))

        fps_n += 1
        if now - fps_t >= 0.5:
            fps = fps_n / (now - fps_t)
            fps_t = now
            fps_n = 0

        texts = draw_ui(frame, recognizer, fast_assist, fps, guide_score, guide_msg, flash_until, now, mirror)
        if kp is not None:
            draw_hand_labels(frame, kp, sc, texts)
        draw_texts(frame, texts + status)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            recognizer.reset()
            fast_assist.reset()
        if key == ord("c"):
            recognizer.reset()
            fast_assist.start_calibration(now)
            hand_lock.reset()
        if key == ord("m"):
            mirror = not mirror
            recognizer.reset()
            fast_assist.start_calibration(now)
            smoother = F.KeypointFilter(n_kp=17)
            imputer = WristImputer()
            hand_lock.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
