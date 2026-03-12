"""
Canelo AI Coach 2
Real-time boxing coach using YOLOv8-pose (webcam).
Coaching metrics: guard, stance, shoulder, head, bounce.
Punch detection: jab, cross, hook via dist/shoulder_width rolling extend.
Punch direction lane guide + punch trail visualization.
"""

import cv2, numpy as np, collections, time, math
from ultralytics import YOLO

model = YOLO("yolov8x-pose.pt")

# --- per-boxer DNA ---
DNA = {
    "guard_perfect": 0.68, "guard_ok": 1.00, "guard_max_ratio": 1.30,
    "stance_ratio_min": 1.10, "stance_ideal_min": 1.40, "stance_ideal_max": 1.90, "stance_ratio_max": 2.20,
    "shoulder_tilt_ok": 0.18, "shoulder_tilt_max": 0.35,
    "head_height_good": 0.22, "head_height_min": 0.12,
    "bounce_target": 0.12, "bounce_min": 0.05,
}

DYN_EXTEND_THRESH = 0.28
DYN_HOOK_THRESH   = 0.42
STRAIGHT_ANGLE    = 114
WINDOW_TITLE      = "Canelo Coach 2"
HEADER_LABEL      = "CANELO AI COACH 2"

# --- COCO 17 keypoint indices ---
NOSE = 0
L_SHOULDER = 5;  R_SHOULDER = 6
L_ELBOW    = 7;  R_ELBOW    = 8
L_WRIST    = 9;  R_WRIST    = 10
L_HIP      = 11; R_HIP      = 12
L_KNEE     = 13; R_KNEE     = 14
L_ANKLE    = 15; R_ANKLE    = 16

_JAB_IDX   = (L_WRIST, L_ELBOW, L_SHOULDER)
_CROSS_IDX = (R_WRIST, R_ELBOW, R_SHOULDER)

COCO_SKEL = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
    (0,5),(0,6)
]

HOOK_ANGLE_MIN = 50
HOOK_ANGLE_MAX = 110
PUNCH_COOLDOWN = 0.75

# view mode EMA (0=front, 1=side)
_view_mode = 0.0
FRONT_SW = 0.11
SIDE_SW  = 0.05

# history buffers
_jab_dist_hist   = collections.deque(maxlen=12)
_cross_dist_hist = collections.deque(maxlen=12)
_ankle_y_hist    = collections.deque(maxlen=30)

# punch counts / timing
_jab_count   = 0
_cross_count = 0
_hook_count  = 0
_last_jab    = 0.0
_last_cross  = 0.0
_dbg = dict(
    j_ext=0.0, j_el_ang=0.0, j_hook_n=0.0, j_arm=0.0,
    c_ext=0.0, c_el_ang=0.0, c_hook_n=0.0, c_arm=0.0
)

# toggles
_show_debug = True
_show_trail = True
_show_guide = True


# --- _TrailBuf class ---
class _TrailBuf:
    def __init__(self, maxlen=40):
        self._buf = collections.deque(maxlen=maxlen)

    def append(self, pt, tag="idle"):
        self._buf.append((pt, tag))

    def mark_last(self, tag):
        if self._buf:
            pt, _ = self._buf[-1]
            self._buf[-1] = (pt, tag)

    def __iter__(self):
        return iter(self._buf)

    def __len__(self):
        return len(self._buf)


_jab_trail   = _TrailBuf()
_cross_trail = _TrailBuf()

_TRAIL_COLORS = {
    "idle": (120, 120, 120),
    "fwd":  (100, 230, 100),
    "side": (100, 220, 180),
    "jab":  (0, 255, 80),
    "cross":(80, 120, 255),
    "hook": (0, 160, 255),
}


# --- helper functions ---

def kp(kps_xyn, idx):
    """normalized keypoint -> np.array([x, y])"""
    return np.array([float(kps_xyn[idx, 0]), float(kps_xyn[idx, 1])])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def angle3(a, b, c):
    """angle at b formed by a-b-c"""
    v1 = a - b
    v2 = c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def draw_skeleton(frame, kps_px, kps_conf):
    for a, b in COCO_SKEL:
        if kps_conf[a] > 0.3 and kps_conf[b] > 0.3:
            cv2.line(frame, tuple(kps_px[a].astype(int)), tuple(kps_px[b].astype(int)), (180, 180, 180), 2)
    for i, (pt, c) in enumerate(zip(kps_px, kps_conf)):
        if c > 0.3:
            cv2.circle(frame, tuple(pt.astype(int)), 4, (255, 255, 255), -1)


def score_bar(frame, x, y, score, label, color):
    bar_w = 160
    cv2.rectangle(frame, (x, y), (x + bar_w, y + 14), (60, 60, 60), -1)
    fill = int(bar_w * score / 100)
    cv2.rectangle(frame, (x, y), (x + fill, y + 14), color, -1)
    cv2.putText(frame, f"{label}: {score}pt", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


# --- detect_punch ---
def detect_punch(kps, now):
    global _view_mode, _jab_count, _cross_count, _hook_count, _last_jab, _last_cross

    jw, je, js = _JAB_IDX
    cw, ce, cs = _CROSS_IDX

    j_wr = kp(kps, jw); j_el = kp(kps, je); j_sh = kp(kps, js)
    c_wr = kp(kps, cw); c_el = kp(kps, ce); c_sh = kp(kps, cs)
    l_sh = kp(kps, L_SHOULDER); r_sh = kp(kps, R_SHOULDER)
    nose  = kp(kps, NOSE)

    sw = dist(l_sh, r_sh)
    if sw < 1e-4:
        sw = 0.15

    # view mode update
    raw_sw = sw
    _view_mode_new = np.clip((FRONT_SW - raw_sw) / (FRONT_SW - SIDE_SW), 0, 1)
    _view_mode = 0.85 * _view_mode + 0.15 * float(_view_mode_new)
    side = _view_mode

    # arm lengths
    j_arm_len = dist(j_sh, j_wr) / sw
    c_arm_len = dist(c_sh, c_wr) / sw
    _jab_dist_hist.append(j_arm_len)
    _cross_dist_hist.append(c_arm_len)

    def rolling_extend(hist):
        n = len(hist)
        if n < 2:
            return 0.0
        h = list(hist)
        v  = h[-1] - h[-7] if n >= 7 else h[-1] - h[0]
        v2 = (h[-2] - h[-8]) if n >= 8 else 0
        v3 = (h[-3] - h[-9]) if n >= 9 else 0
        return max(v, v2, v3)

    j_extend = rolling_extend(_jab_dist_hist)
    c_extend = rolling_extend(_cross_dist_hist)

    j_el_ang = angle3(j_wr, j_el, j_sh)
    c_el_ang = angle3(c_wr, c_el, c_sh)

    # hook swing normalized
    nose_x    = float(nose[0])
    sh_mid_x  = (float(l_sh[0]) + float(r_sh[0])) / 2
    facing_right = nose_x > sh_mid_x

    j_xn = abs(float(j_wr[0]) - float(j_sh[0])) / sw
    j_yn = abs(float(j_wr[1]) - float(j_sh[1])) / sw
    c_xn = abs(float(c_wr[0]) - float(c_sh[0])) / sw
    c_yn = abs(float(c_wr[1]) - float(c_sh[1])) / sw

    j_hook_n = (1 - side) * j_xn + side * j_yn
    c_hook_n = (1 - side) * c_xn + side * c_yn

    _dbg.update(
        j_ext=j_extend, j_el_ang=j_el_ang, j_hook_n=j_hook_n, j_arm=j_arm_len,
        c_ext=c_extend, c_el_ang=c_el_ang, c_hook_n=c_hook_n, c_arm=c_arm_len
    )

    def guard_ok(wr_norm):
        wr_y   = float(wr_norm[1])
        nose_y = float(nose[1])
        return (wr_y - nose_y) / sw < DNA["guard_max_ratio"] * 1.5

    def fire(hand, punch_type, guard_good):
        global _jab_count, _cross_count, _hook_count, _last_jab, _last_cross
        if hand == "jab":
            if now - _last_jab < PUNCH_COOLDOWN:
                return
            _last_jab = now
            if punch_type == "훅":
                _hook_count += 1
                _jab_trail.mark_last("hook")
            else:
                _jab_count += 1
                _jab_trail.mark_last("jab")
        else:
            if now - _last_cross < PUNCH_COOLDOWN:
                return
            _last_cross = now
            if punch_type == "훅":
                _hook_count += 1
                _cross_trail.mark_last("hook")
            else:
                _cross_count += 1
                _cross_trail.mark_last("cross")

    # append trail
    _jab_trail.append(tuple(j_wr), "fwd" if j_extend > DYN_EXTEND_THRESH * 0.5 else "idle")
    _cross_trail.append(tuple(c_wr), "fwd" if c_extend > DYN_EXTEND_THRESH * 0.5 else "idle")

    # jab / cross
    if j_extend > DYN_EXTEND_THRESH:
        if j_el_ang >= STRAIGHT_ANGLE:
            fire("jab", "원(잽)", guard_ok(c_wr))
    elif j_hook_n > DYN_HOOK_THRESH:
        if HOOK_ANGLE_MIN <= j_el_ang <= HOOK_ANGLE_MAX:
            if abs(float(j_wr[1]) - float(j_sh[1])) < 0.22:
                fire("jab", "훅", guard_ok(c_wr))

    if c_extend > DYN_EXTEND_THRESH:
        if c_el_ang >= STRAIGHT_ANGLE:
            fire("cross", "크로스", guard_ok(j_wr))
    elif c_hook_n > DYN_HOOK_THRESH:
        if HOOK_ANGLE_MIN <= c_el_ang <= HOOK_ANGLE_MAX:
            if abs(float(c_wr[1]) - float(c_sh[1])) < 0.22:
                fire("cross", "훅", guard_ok(j_wr))


# --- draw_punch_trails ---
def draw_punch_trails(frame, kps, fw, fh):
    if not _show_trail:
        return

    def px(pt):
        return (int(float(pt[0]) * fw), int(float(pt[1]) * fh))

    for trail, col_key in [(_jab_trail, "jab"), (_cross_trail, "cross")]:
        pts = list(trail)
        for i in range(1, len(pts)):
            p1  = px(pts[i - 1][0])
            p2  = px(pts[i][0])
            tag = pts[i][1]
            col = _TRAIL_COLORS.get(tag, _TRAIL_COLORS["idle"])
            alpha = i / len(pts)
            cv2.line(frame, p1, p2, tuple(int(c * alpha) for c in col), 2)

    jw = _JAB_IDX[0];  cw = _CROSS_IDX[0]
    j_wr_px = (int(float(kps[jw, 0]) * fw), int(float(kps[jw, 1]) * fh))
    c_wr_px = (int(float(kps[cw, 0]) * fw), int(float(kps[cw, 1]) * fh))

    j_fwd  = max(0.0, _dbg["j_ext"])
    c_fwd  = max(0.0, _dbg["c_ext"])
    j_ring = max(5, int(j_fwd * fw * 0.15))
    c_ring = max(5, int(c_fwd * fw * 0.15))
    cv2.circle(frame, j_wr_px, j_ring, (0, 230, 80), 1)
    cv2.circle(frame, c_wr_px, c_ring, (80, 120, 255), 1)

    h_img = frame.shape[0]
    cv2.putText(
        frame,
        f"JAB:{_jab_count}  CROSS:{_cross_count}  HOOK:{_hook_count}",
        (10, h_img - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 180), 2
    )


# --- draw_punch_guide ---
def draw_punch_guide(frame, kps, fw, fh):
    if not _show_guide:
        return

    jw, js = _JAB_IDX[0],   _JAB_IDX[2]
    cw, cs = _CROSS_IDX[0], _CROSS_IDX[2]
    j_wr = (int(float(kps[jw, 0]) * fw), int(float(kps[jw, 1]) * fh))
    j_sh = (int(float(kps[js, 0]) * fw), int(float(kps[js, 1]) * fh))
    c_wr = (int(float(kps[cw, 0]) * fw), int(float(kps[cw, 1]) * fh))
    c_sh = (int(float(kps[cs, 0]) * fw), int(float(kps[cs, 1]) * fh))

    cx   = fw / 2
    side = _view_mode
    nose_x   = float(kps[NOSE, 0]) * fw
    sh_mid_x = (float(kps[L_SHOULDER, 0]) + float(kps[R_SHOULDER, 0])) / 2 * fw
    facing_rt = nose_x > sh_mid_x

    def get_dir(sh_px):
        front_dx = -1.0 if sh_px[0] > cx else 1.0
        side_dx  =  1.0 if facing_rt else -1.0
        dx = front_dx * (1 - side) + side_dx * side
        return dx, 0.0

    def draw_lane(sh_px, wr_px, direction, ratio, col_on, col_off, label):
        dx, dy = direction
        mag = max((dx**2 + dy**2)**0.5, 1e-6)
        dx, dy = dx / mag, dy / mag
        pxd, pyd = -dy, dx
        lane_len = int(fw * 0.22)
        half_w   = int(fw * 0.035)
        n_seg    = 8
        col = col_on if ratio > 0.5 else col_off
        for i in range(n_seg):
            t0   = i * lane_len // n_seg
            t1   = (i + 1) * lane_len // n_seg
            fade = (i + 1) / n_seg
            c    = tuple(int(v * fade) for v in col)
            w    = max(1, int(2 * fade))
            lp1  = (int(sh_px[0] + dx * t0 + pxd * half_w), int(sh_px[1] + dy * t0 + pyd * half_w))
            lp2  = (int(sh_px[0] + dx * t1 + pxd * half_w), int(sh_px[1] + dy * t1 + pyd * half_w))
            cv2.line(frame, lp1, lp2, c, w)
            rp1  = (int(sh_px[0] + dx * t0 - pxd * half_w), int(sh_px[1] + dy * t0 - pyd * half_w))
            rp2  = (int(sh_px[0] + dx * t1 - pxd * half_w), int(sh_px[1] + dy * t1 - pyd * half_w))
            cv2.line(frame, rp1, rp2, c, w)
        tip = (int(sh_px[0] + dx * lane_len), int(sh_px[1] + dy * lane_len))
        cv2.arrowedLine(frame, sh_px, tip, col, 2, tipLength=0.15)
        cv2.circle(frame, tip, 18, col, 2)
        cv2.circle(frame, tip, 5,  col, -1)
        cv2.putText(frame, label, (tip[0] - 18, tip[1] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        proj = (wr_px[0] - sh_px[0]) * dx + (wr_px[1] - sh_px[1]) * dy
        proj = max(0.0, min(proj, lane_len))
        on_ln = (int(sh_px[0] + dx * proj), int(sh_px[1] + dy * proj))
        fist_col = (0, 255, 80) if ratio >= 1.0 else (200, 200, 0) if ratio > 0.5 else (100, 100, 100)
        cv2.line(frame, wr_px, on_ln, fist_col, 1)
        cv2.circle(frame, wr_px, 9, fist_col, 2)

    j_ratio = min(_dbg['j_ext'] / max(DYN_EXTEND_THRESH, 1e-6), 1.5)
    c_ratio = min(_dbg['c_ext'] / max(DYN_EXTEND_THRESH, 1e-6), 1.5)
    draw_lane(j_sh, j_wr, get_dir(j_sh), j_ratio, (0, 230, 80),  (0, 55, 25),  "JAB")
    draw_lane(c_sh, c_wr, get_dir(c_sh), c_ratio, (80, 120, 255),(20, 30, 80), "CROSS")


# --- draw_debug_ui ---
def draw_debug_ui(frame):
    h   = frame.shape[0]
    px2 = 10
    py  = h - 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (px2 - 4, py - 18), (px2 + 380, py + 200), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "[ DEBUG - D key ]", (px2, py - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    def val_row(y, label, val, threshold, higher=True):
        ok  = (val >= threshold) if higher else (val <= threshold)
        col = (0, 230, 80) if ok else (60, 60, 255)
        bar_max = 280
        ratio   = min(val / max(threshold, 1e-6), 2.0)
        fill    = int(bar_max * ratio / 2.0)
        cv2.rectangle(frame, (px2 + 100, y), (px2 + 100 + bar_max, y + 12), (50, 50, 50), -1)
        cv2.rectangle(frame, (px2 + 100, y), (px2 + 100 + fill,    y + 12), col,          -1)
        mx = px2 + 100 + bar_max // 2
        cv2.line(frame, (mx, y - 2), (mx, y + 14), (255, 200, 0), 2)
        cv2.putText(frame, f"{label}: {val:.3f}/{threshold:.3f}", (px2, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    y = py + 10
    cv2.putText(frame, "-- JAB --", (px2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 0), 1); y += 18
    val_row(y, "EXTEND  ", _dbg["j_ext"],    DYN_EXTEND_THRESH); y += 20
    val_row(y, "EL_ANG  ", _dbg["j_el_ang"], STRAIGHT_ANGLE);    y += 20
    val_row(y, "HOOK_SW ", _dbg["j_hook_n"], DYN_HOOK_THRESH);   y += 24
    cv2.putText(frame, "-- CROSS --", (px2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 0), 1); y += 18
    val_row(y, "EXTEND  ", _dbg["c_ext"],    DYN_EXTEND_THRESH); y += 20
    val_row(y, "EL_ANG  ", _dbg["c_el_ang"], STRAIGHT_ANGLE);    y += 20
    val_row(y, "HOOK_SW ", _dbg["c_hook_n"], DYN_HOOK_THRESH)


# --- main loop ---
cap         = cv2.VideoCapture(0)
frame_count = 0
print(f"{HEADER_LABEL} 활성화! 종료: Q")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame = cv2.flip(frame, 1)
    now   = time.time()
    h_f, w_f = frame.shape[:2]

    results = model(frame, verbose=False)

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (310, 310), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, HEADER_LABEL, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

    if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
        kps_xyn  = results[0].keypoints.xyn[0].cpu().numpy()   # (17,2) normalized
        kps_xyc  = results[0].keypoints.data[0].cpu().numpy()  # (17,3) with conf
        kps_conf = kps_xyc[:, 2]

        kps_px = np.stack([kps_xyn[:, 0] * w_f, kps_xyn[:, 1] * h_f], axis=1)

        draw_skeleton(frame, kps_px, kps_conf)

        l_sh    = kps_xyn[L_SHOULDER]
        r_sh    = kps_xyn[R_SHOULDER]
        nose_pt = kps_xyn[NOSE]
        l_wr    = kps_xyn[L_WRIST]
        r_wr    = kps_xyn[R_WRIST]
        l_ank   = kps_xyn[L_ANKLE]
        r_ank   = kps_xyn[R_ANKLE]

        sw = float(np.linalg.norm(l_sh - r_sh))
        if sw < 1e-4:
            sw = 0.15

        # [1] Guard
        l_guard_ratio = (float(l_wr[1]) - float(nose_pt[1])) / sw
        r_guard_ratio = (float(r_wr[1]) - float(nose_pt[1])) / sw

        def guard_pts(ratio):
            if ratio <= DNA["guard_perfect"]:   return 15
            if ratio <= DNA["guard_ok"]:        return 10
            if ratio <= DNA["guard_max_ratio"]: return 5
            return 0

        def guard_status(ratio, side):
            if ratio <= DNA["guard_perfect"]:   return f"{side} Guard Perfect!", (0, 255, 100)
            if ratio <= DNA["guard_ok"]:        return f"{side} Guard OK",        (0, 210, 100)
            if ratio <= DNA["guard_max_ratio"]: return f"{side} Guard LOW!",      (0, 165, 255)
            return f"{side} Guard DOWN!", (0, 0, 255)

        l_pts = guard_pts(l_guard_ratio)
        r_pts = guard_pts(r_guard_ratio)
        guard_score = l_pts + r_pts
        if l_guard_ratio >= r_guard_ratio:
            guard_msg, guard_col = guard_status(l_guard_ratio, "Left")
        else:
            guard_msg, guard_col = guard_status(r_guard_ratio, "Right")
        if l_pts == 15 and r_pts == 15:
            guard_msg, guard_col = "Perfect Guard!", (0, 255, 100)

        # [2] Stance
        stance_ratio = float(np.linalg.norm(l_ank - r_ank)) / sw
        if DNA["stance_ideal_min"] <= stance_ratio <= DNA["stance_ideal_max"]:
            stance_score = 20
            stance_msg, stance_col = f"Perfect Stance! ({stance_ratio:.2f}x)", (0, 255, 100)
        elif DNA["stance_ratio_min"] <= stance_ratio < DNA["stance_ideal_min"]:
            stance_score = 12
            stance_msg, stance_col = f"Slightly Narrow ({stance_ratio:.2f}x)", (0, 210, 100)
        elif DNA["stance_ideal_max"] < stance_ratio <= DNA["stance_ratio_max"]:
            stance_score = 12
            stance_msg, stance_col = f"Slightly Wide ({stance_ratio:.2f}x)", (0, 210, 100)
        elif stance_ratio < DNA["stance_ratio_min"]:
            stance_score = max(0, int(12 - (DNA["stance_ratio_min"] - stance_ratio) * 30))
            stance_msg, stance_col = f"TOO NARROW ({stance_ratio:.2f}x)", (0, 165, 255)
        else:
            stance_score = max(0, int(12 - (stance_ratio - DNA["stance_ratio_max"]) * 30))
            stance_msg, stance_col = f"TOO WIDE ({stance_ratio:.2f}x)", (0, 165, 255)

        # [3] Shoulder
        tilt = abs(float(l_sh[1]) - float(r_sh[1])) / sw
        if tilt <= DNA["shoulder_tilt_ok"]:
            shoulder_score = 15
            shoulder_msg, shoulder_col = "Shoulders Level!", (0, 255, 100)
        elif tilt <= DNA["shoulder_tilt_max"]:
            shoulder_score = 8
            shoulder_msg, shoulder_col = f"Slightly Tilted ({tilt:.2f})", (0, 210, 100)
        else:
            shoulder_score = 0
            shoulder_msg, shoulder_col = f"Shoulders Tilted! ({tilt:.2f})", (0, 0, 255)

        # [4] Head
        head_height = ((float(l_sh[1]) + float(r_sh[1])) / 2 - float(nose_pt[1])) / sw
        if head_height >= DNA["head_height_good"]:
            head_score = 15
            head_msg, head_col = "Head Up!", (0, 255, 100)
        elif head_height >= DNA["head_height_min"]:
            head_score = 8
            head_msg, head_col = f"Head Slightly Low ({head_height:.2f})", (0, 210, 100)
        else:
            head_score = 0
            head_msg, head_col = "Chin DOWN! Head up!", (0, 0, 255)

        # [5] Bounce
        norm_ankle_y = float(l_ank[1]) / sw
        _ankle_y_hist.append(norm_ankle_y)
        bounce = float(np.std(list(_ankle_y_hist))) if len(_ankle_y_hist) == 30 else 0.0
        if bounce >= DNA["bounce_target"]:
            bounce_score = 20
            bounce_msg, bounce_col = f"Active Step! ({bounce:.3f})", (0, 255, 100)
        elif bounce >= DNA["bounce_min"]:
            ratio_b = (bounce - DNA["bounce_min"]) / (DNA["bounce_target"] - DNA["bounce_min"])
            bounce_score = int(ratio_b * 20)
            bounce_msg, bounce_col = f"Move your feet ({bounce:.3f})", (0, 210, 100)
        else:
            bounce_score = 0
            bounce_msg, bounce_col = f"STATIC! Step! ({bounce:.3f})", (0, 0, 255)

        # punch detection + overlays
        detect_punch(kps_xyn, now)
        draw_punch_guide(frame, kps_xyn, w_f, h_f)
        draw_punch_trails(frame, kps_xyn, w_f, h_f)

        # total score
        total = guard_score + stance_score + shoulder_score + head_score + bounce_score
        if total >= 85:   total_col, grade = (0, 255, 100), "S"
        elif total >= 70: total_col, grade = (0, 200, 255), "A"
        elif total >= 50: total_col, grade = (0, 165, 255), "B"
        else:             total_col, grade = (0, 0, 255),   "C"

        # score bars
        y = 50
        score_bar(frame, 10, y, guard_score,    "[1] Guard",    guard_col);    y += 38
        score_bar(frame, 10, y, stance_score,   "[2] Stance",   stance_col);   y += 38
        score_bar(frame, 10, y, shoulder_score, "[3] Shoulder", shoulder_col); y += 38
        score_bar(frame, 10, y, head_score,     "[4] Head",     head_col);     y += 38
        score_bar(frame, 10, y, bounce_score,   "[5] Bounce",   bounce_col);   y += 38
        cv2.putText(frame, f"TOTAL: {total}/100  [{grade}]", (10, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, total_col, 2)

        h_img = frame.shape[0]
        for i, (msg, col) in enumerate([
            (guard_msg, guard_col), (stance_msg, stance_col),
            (shoulder_msg, shoulder_col), (head_msg, head_col), (bounce_msg, bounce_col)
        ]):
            cv2.putText(frame, msg, (10, h_img - 130 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        if _show_debug:
            draw_debug_ui(frame)

    else:
        cv2.putText(frame, "자세를 감지할 수 없습니다", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

    cv2.imshow(WINDOW_TITLE, frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        _show_debug = not _show_debug
    elif key == ord('t'):
        _show_trail = not _show_trail
    elif key == ord('g'):
        _show_guide = not _show_guide

cap.release()
cv2.destroyAllWindows()
