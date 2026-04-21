import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
from collections import deque
from PIL import ImageFont, ImageDraw, Image as PILImage

MODEL_PATH = 'pose_landmarker_full.task'
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("포즈 모델 다운로드 중 (약 30MB)...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
        MODEL_PATH
    )
    print("다운로드 완료!")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (27, 31), (28, 30), (28, 32),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
]

# ── 카넬로 기준값 (DNA) ───────────────────────────────────────────────────────
CANELO = {
    "guard_perfect":      0.84,
    "guard_ok":           1.26,
    "guard_max_ratio":    1.71,
    "stance_ratio_min":   1.10,
    "stance_ideal_min":   1.35,
    "stance_ideal_max":   1.75,
    "stance_ratio_max":   2.10,
    "shoulder_tilt_ok":   0.15,
    "shoulder_tilt_max":  0.30,
    "head_height_good":   0.25,
    "head_height_min":    0.15,
    "bounce_target":      0.10,
    "bounce_min":         0.04,
}
DNA = CANELO

# ══════════════════════════════════════════════════════════════════════════════
# 펀치 감지 모듈 (비볼/카넬로/가르시아 공통)
# ══════════════════════════════════════════════════════════════════════════════
STANCE = "orthodox"

_JAB_IDX   = (16, 14, 12)
_CROSS_IDX = (15, 13, 11)
if STANCE == "southpaw":
    _JAB_IDX, _CROSS_IDX = _CROSS_IDX, _JAB_IDX

DYN_EXTEND_THRESH = 0.28   # 잽/크로스: shoulder_w 대비 6프레임 팔 펴짐 비율
DYN_HOOK_THRESH   = 0.42   # 훅: shoulder_w 대비 4프레임 손목 스윙 비율
STRAIGHT_ANGLE  = 114    # CSV p85: 114.1° → 카넬로는 팔을 완전히 안 뻗는 스타일
HOOK_ANGLE_MIN  = 50     # 훅: 팔꿈치 최소 각도
HOOK_ANGLE_MAX  = 110    # 훅: 팔꿈치 최대 각도 (65→110)
PUNCH_COOLDOWN  = 0.75   # 같은 손 재감지 방지 (초)

# ── 어퍼컷 감지 ───────────────────────────────────────────────────────────────
UPPERCUT_Y_THRESH  = 0.18
UPPERCUT_ANGLE_MIN = 50
UPPERCUT_ANGLE_MAX = 120

# ── 캘리브레이션 ─────────────────────────────────────────────────────────────
_CALIB_DURATION = 7.0
_calib_start    = None
_calib_j_vals   = []
_calib_c_vals   = []
_calib_done     = False

# ── 시작 준비 체크 ────────────────────────────────────────────────────────────
_READY_FRAMES   = 20
_pose_stable    = 0
_pose_ready     = False

# ── 카메라 앵글 자동감지 ──────────────────────────────────────────────────────
FRONT_SW  = 0.11
SIDE_SW   = 0.05
_view_mode = 0.0

_jab_hist      = deque(maxlen=8)
_cross_hist    = deque(maxlen=8)
_jab_dist_hist   = deque(maxlen=12)  # 잽손 팔 펴짐 거리 히스토리
_cross_dist_hist = deque(maxlen=12)  # 크로스손 팔 펴짐 거리 히스토리
_last_pt    = {"jab": 0.0, "cross": 0.0}
_pdisplay   = {"text": "", "col": (255, 255, 255), "until": 0.0}
_pstats     = {"원(잽)": 0, "투(크로스)": 0, "훅": 0, "어퍼컷": 0}
_show_debug = True
_show_guide = True   # G키로 토글
_dbg = {
    "j_ext":    0.0, "j_el_ang": 0.0, "j_hook_n": 0.0, "j_arm": 0.0,
    "c_ext":    0.0, "c_el_ang": 0.0, "c_hook_n": 0.0, "c_arm": 0.0,
}

_TRAIL_COLORS = {
    'idle':  (100, 100, 100),
    'fwd':   (0,  180,  60),
    'side':  (0,  200, 200),
    'jab':   (0,  255,  80),
    'cross': (255, 80,   0),
    'hook':     (0,  180, 255),
    'uppercut': (180,  0, 255),
}
class _TrailBuf:
    def __init__(self, n=25):
        self._d = []
        self._n = n
    def append(self, px, py, state='idle'):
        self._d.append([px, py, state])
        if len(self._d) > self._n:
            self._d.pop(0)
    def mark_last(self, count, state):
        for i in range(max(0, len(self._d) - count), len(self._d)):
            self._d[i][2] = state
    def __iter__(self): return iter(self._d)
    def __len__(self):  return len(self._d)

_jab_trail   = _TrailBuf(25)
_cross_trail = _TrailBuf(25)
_show_trail  = True

# ── 한글 렌더링 (PIL) ─────────────────────────────────────────────────────────
def _load_font(size):
    for path in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc",
                 "C:/Windows/Fonts/batang.ttc"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

_KR_SM = _load_font(17)
_KR_MD = _load_font(26)

def put_kr(img, text, pos, color, font):
    pil  = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def lm(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y])

def lm3(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y, p.z])

def dist(a, b):
    return np.linalg.norm(a - b)

def angle3pt(a, b, c):
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def detect_punch(lms, now):
    """펀치 감지 – 팔 펴짐 거리 기반 (Z축 폐기, shoulder_w 비례 동적 임계)"""
    global _view_mode, _calib_start, _calib_done, DYN_EXTEND_THRESH
    jw, je, js = _JAB_IDX
    cw, ce, cs = _CROSS_IDX

    _jab_hist.append(lm3(lms, jw))
    _cross_hist.append(lm3(lms, cw))

    if len(_jab_hist) < 7:
        return

    nose  = lm(lms, 0)
    l_sh_ = lm(lms, 11);  r_sh_ = lm(lms, 12)
    j_wr  = lm(lms, jw);  j_el  = lm(lms, je);  j_sh = lm(lms, js)
    c_wr  = lm(lms, cw);  c_el  = lm(lms, ce);  c_sh = lm(lms, cs)

    # ── 앵글 감지 (스무딩) ────────────────────────────────────────────────────
    sw_raw = float(np.linalg.norm(l_sh_[:2] - r_sh_[:2]))
    if sw_raw >= FRONT_SW:      target = 0.0
    elif sw_raw <= SIDE_SW:     target = 1.0
    else:                       target = (FRONT_SW - sw_raw) / (FRONT_SW - SIDE_SW)
    _view_mode = _view_mode * 0.70 + target * 0.30
    side = _view_mode

    sh_mid_x     = (l_sh_[0] + r_sh_[0]) / 2
    facing_right = nose[0] > sh_mid_x

    # ── 유효 어깨너비 ──────────────────────────────────────────────────────────
    head_to_sh = abs(((l_sh_[1] + r_sh_[1]) / 2) - nose[1])
    side_ref   = max(head_to_sh * 0.55, 0.05)
    sw = max(sw_raw * (1 - side) + side_ref * side, 0.05)

    # ── 팔 펴짐 거리 (shoulder↔wrist 2D 거리 / sw) ────────────────────────────
    j_arm_len = float(dist(j_sh, j_wr)) / sw
    c_arm_len = float(dist(c_sh, c_wr)) / sw
    _jab_dist_hist.append(j_arm_len)
    _cross_dist_hist.append(c_arm_len)

    # ── 캘리브레이션 데이터 수집 ──────────────────────────────────────────────
    if not _calib_done:
        if _calib_start is None:
            _calib_start = now
        _calib_j_vals.append(j_arm_len)
        _calib_c_vals.append(c_arm_len)
        if now - _calib_start >= _CALIB_DURATION and len(_calib_j_vals) > 10:
            max_ext = max(max(_calib_j_vals), max(_calib_c_vals))
            if max_ext > DYN_EXTEND_THRESH * 1.2:
                DYN_EXTEND_THRESH = max_ext * 0.60
            _calib_done = True

    if len(_jab_dist_hist) < 7:
        return

    # 롤링 맥스: 최근 3프레임 중 최대 확장값 (노이즈 감소)
    j_extend_2d = max(
        float(_jab_dist_hist[-1] - _jab_dist_hist[-7]),
        float(_jab_dist_hist[-2] - _jab_dist_hist[-8]) if len(_jab_dist_hist) >= 8 else 0,
        float(_jab_dist_hist[-3] - _jab_dist_hist[-9]) if len(_jab_dist_hist) >= 9 else 0,
    )
    c_extend_2d = max(
        float(_cross_dist_hist[-1] - _cross_dist_hist[-7]),
        float(_cross_dist_hist[-2] - _cross_dist_hist[-8]) if len(_cross_dist_hist) >= 8 else 0,
        float(_cross_dist_hist[-3] - _cross_dist_hist[-9]) if len(_cross_dist_hist) >= 9 else 0,
    )
    j_z_fwd = max(0.0, float(_jab_hist[-7][2]   - _jab_hist[-1][2]))   / sw
    c_z_fwd = max(0.0, float(_cross_hist[-7][2] - _cross_hist[-1][2])) / sw
    j_extend = j_extend_2d * (1 - side) + j_z_fwd * side
    c_extend = c_extend_2d * (1 - side) + c_z_fwd * side

    # ── 훅: 손목 스윙 (sw 정규화) ─────────────────────────────────────────────
    j_xn = abs(float(_jab_hist[-1][0]   - _jab_hist[-5][0])) / sw
    c_xn = abs(float(_cross_hist[-1][0] - _cross_hist[-5][0])) / sw
    j_yn = abs(float(_jab_hist[-1][1]   - _jab_hist[-5][1])) / sw
    c_yn = abs(float(_cross_hist[-1][1] - _cross_hist[-5][1])) / sw
    j_hook_n = (1 - side) * j_xn + side * j_yn
    c_hook_n = (1 - side) * c_xn + side * c_yn

    # ── 팔꿈치 각도 ────────────────────────────────────────────────────────────
    j_el_ang = angle3pt(j_sh, j_el, j_wr)
    c_el_ang = angle3pt(c_sh, c_el, c_wr)

    # ── _dbg 업데이트 ──────────────────────────────────────────────────────────
    _dbg.update({
        "j_ext":    j_extend,   "j_el_ang": j_el_ang,
        "j_hook_n": j_hook_n,   "j_arm":    j_arm_len,
        "c_ext":    c_extend,   "c_el_ang": c_el_ang,
        "c_hook_n": c_hook_n,   "c_arm":    c_arm_len,
    })

    def guard_ok(other_wr):
        return other_wr[1] < nose[1] + 0.10

    def fire(key, ptype, form_ok):
        _pstats[ptype] += 1
        color = (0, 255, 100) if form_ok else (0, 165, 255)
        suffix = "Good Form!" if form_ok else "Guard 손 내려감!"
        _pdisplay["text"]  = f"{ptype}!  {suffix}"
        _pdisplay["col"]   = color
        _pdisplay["until"] = now + 1.5
        _last_pt[key]      = now
        trail = _jab_trail if key == "jab" else _cross_trail
        state = "jab" if "잽" in ptype else "cross" if "크로스" in ptype else "uppercut" if "어퍼" in ptype else "hook"
        trail.mark_last(12, state)

    # ── 잽손 ──────────────────────────────────────────────────────────────────
    if now - _last_pt["jab"] > PUNCH_COOLDOWN:
        if j_extend > DYN_EXTEND_THRESH:
            if j_el_ang >= STRAIGHT_ANGLE:
                fire("jab", "원(잽)", guard_ok(c_wr))
        elif j_extend < DYN_EXTEND_THRESH * 0.35:
            if j_hook_n > DYN_HOOK_THRESH:
                if HOOK_ANGLE_MIN <= j_el_ang <= HOOK_ANGLE_MAX:
                    if abs(j_wr[1] - j_sh[1]) < 0.20:
                        if j_xn > j_yn * 1.5:
                            fire("jab", "훅", guard_ok(c_wr))
            else:
                j_yn_up = float(_jab_hist[-5][1] - _jab_hist[-1][1]) / sw
                if j_yn_up > UPPERCUT_Y_THRESH * 1.3:
                    if UPPERCUT_ANGLE_MIN <= j_el_ang <= UPPERCUT_ANGLE_MAX:
                        if float(_jab_hist[-5][1]) > (nose[1] + 0.05):
                            fire("jab", "어퍼컷", guard_ok(c_wr))

    # ── 크로스손 ──────────────────────────────────────────────────────────────
    if now - _last_pt["cross"] > PUNCH_COOLDOWN:
        if c_extend > DYN_EXTEND_THRESH:
            if c_el_ang >= STRAIGHT_ANGLE:
                fire("cross", "투(크로스)", guard_ok(j_wr))
        elif c_extend < DYN_EXTEND_THRESH * 0.35:
            if c_hook_n > DYN_HOOK_THRESH:
                if HOOK_ANGLE_MIN <= c_el_ang <= HOOK_ANGLE_MAX:
                    if abs(c_wr[1] - c_sh[1]) < 0.20:
                        if c_xn > c_yn * 1.5:
                            fire("cross", "훅", guard_ok(j_wr))
            else:
                c_yn_up = float(_cross_hist[-5][1] - _cross_hist[-1][1]) / sw
                if c_yn_up > UPPERCUT_Y_THRESH * 1.3:
                    if UPPERCUT_ANGLE_MIN <= c_el_ang <= UPPERCUT_ANGLE_MAX:
                        if float(_cross_hist[-5][1]) > (nose[1] + 0.05):
                            fire("cross", "어퍼컷", guard_ok(j_wr))


def draw_punch_ui(frame, now):
    h, w = frame.shape[:2]
    px = w - 200
    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 5, 5), (w - 5, 168), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "PUNCH COUNT", (px, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for i, (k, v) in enumerate(_pstats.items()):
        put_kr(frame, f"{k}: {v}", (px, 30 + i * 26), (0, 220, 255), _KR_SM)
    side_pct = int(_view_mode * 100)
    if side_pct < 20:
        angle_label = f"CAM: FRONT ({100 - side_pct}%)"
        angle_col   = (0, 255, 100)
    elif side_pct > 80:
        angle_label = f"CAM: SIDE ({side_pct}%)"
        angle_col   = (0, 200, 255)
    else:
        angle_label = f"CAM: ANGLE ({side_pct}%side)"
        angle_col   = (0, 165, 255)
    cv2.putText(frame, angle_label, (px, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, angle_col, 1)
    if now < _pdisplay["until"] and _pdisplay["text"]:
        txt  = _pdisplay["text"]
        bbox = _KR_MD.getbbox(txt)
        tw   = bbox[2] - bbox[0]
        put_kr(frame, txt, ((w - tw) // 2, h - 50), _pdisplay["col"], _KR_MD)


def draw_punch_guide(frame, lms, fw, fh):
    """잽/크로스 방향 차선 가이드 (G키 토글)"""
    if not _show_guide:
        return

    jw, js = _JAB_IDX[0], _JAB_IDX[2]
    cw, cs = _CROSS_IDX[0], _CROSS_IDX[2]
    j_wr = (int(lms[jw].x * fw), int(lms[jw].y * fh))
    j_sh = (int(lms[js].x * fw), int(lms[js].y * fh))
    c_wr = (int(lms[cw].x * fw), int(lms[cw].y * fh))
    c_sh = (int(lms[cs].x * fw), int(lms[cs].y * fh))

    cx        = fw / 2
    side      = _view_mode
    nose_x    = lms[0].x * fw
    sh_mid_x  = (lms[11].x + lms[12].x) / 2 * fw
    facing_rt = nose_x > sh_mid_x

    def get_dir(sh_px):
        front_dx = -1.0 if sh_px[0] > cx else 1.0
        side_dx  =  1.0 if facing_rt  else -1.0
        dx = front_dx * (1 - side) + side_dx * side
        return dx, 0.0

    def draw_lane(sh_px, wr_px, direction, ratio, col_on, col_off, label):
        dx, dy  = direction
        mag     = max((dx**2 + dy**2)**0.5, 1e-6)
        dx, dy  = dx / mag, dy / mag
        px_d, py_d = -dy, dx
        lane_len = int(fw * 0.22)
        half_w   = int(fw * 0.035)
        n_seg    = 8
        col      = col_on if ratio > 0.5 else col_off
        for i in range(n_seg):
            t0   = i       * lane_len // n_seg
            t1   = (i + 1) * lane_len // n_seg
            fade = (i + 1) / n_seg
            c    = tuple(int(v * fade) for v in col)
            w    = max(1, int(2 * fade))
            lp1 = (int(sh_px[0] + dx*t0 + px_d*half_w), int(sh_px[1] + dy*t0 + py_d*half_w))
            lp2 = (int(sh_px[0] + dx*t1 + px_d*half_w), int(sh_px[1] + dy*t1 + py_d*half_w))
            cv2.line(frame, lp1, lp2, c, w)
            rp1 = (int(sh_px[0] + dx*t0 - px_d*half_w), int(sh_px[1] + dy*t0 - py_d*half_w))
            rp2 = (int(sh_px[0] + dx*t1 - px_d*half_w), int(sh_px[1] + dy*t1 - py_d*half_w))
            cv2.line(frame, rp1, rp2, c, w)
        tip = (int(sh_px[0] + dx * lane_len), int(sh_px[1] + dy * lane_len))
        cv2.arrowedLine(frame, sh_px, tip, col, 2, tipLength=0.15)
        cv2.circle(frame, tip, 18, col, 2)
        cv2.circle(frame, tip,  5, col, -1)
        cv2.putText(frame, label, (tip[0] - 18, tip[1] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        proj  = (wr_px[0] - sh_px[0]) * dx + (wr_px[1] - sh_px[1]) * dy
        proj  = max(0.0, min(proj, lane_len))
        on_ln = (int(sh_px[0] + dx * proj), int(sh_px[1] + dy * proj))
        fist_col = (0, 255, 80) if ratio >= 1.0 else (200, 200, 0) if ratio > 0.5 else (100, 100, 100)
        cv2.line(frame, wr_px, on_ln, fist_col, 1)
        cv2.circle(frame, wr_px, 9, fist_col, 2)

    j_ratio = min(_dbg['j_ext'] / max(DYN_EXTEND_THRESH, 1e-6), 1.5)
    c_ratio = min(_dbg['c_ext'] / max(DYN_EXTEND_THRESH, 1e-6), 1.5)
    draw_lane(j_sh, j_wr, get_dir(j_sh), j_ratio, (0, 230, 80),  (0, 55, 25),  "JAB")
    draw_lane(c_sh, c_wr, get_dir(c_sh), c_ratio, (80, 120, 255), (20, 30, 80), "CROSS")


def draw_debug_ui(frame):
    if not _show_debug:
        return
    h = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, h - 200), (260, h - 5), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    rows = [
        (f"J_EXTEND : {_dbg['j_ext']:.3f}  >{DYN_EXTEND_THRESH}",
         (0,255,80) if _dbg['j_ext'] > DYN_EXTEND_THRESH else (180,180,180)),
        (f"J_EL_ANG : {_dbg['j_el_ang']:.1f}  >{STRAIGHT_ANGLE}",
         (0,255,80) if _dbg['j_el_ang'] >= STRAIGHT_ANGLE else (180,180,180)),
        (f"J_HOOK_SW: {_dbg['j_hook_n']:.3f}  >{DYN_HOOK_THRESH}",
         (0,200,255) if _dbg['j_hook_n'] > DYN_HOOK_THRESH else (180,180,180)),
        (f"J_ARM_LEN: {_dbg['j_arm']:.3f}", (180,180,180)),
        (f"C_EXTEND : {_dbg['c_ext']:.3f}  >{DYN_EXTEND_THRESH}",
         (0,255,80) if _dbg['c_ext'] > DYN_EXTEND_THRESH else (180,180,180)),
        (f"C_EL_ANG : {_dbg['c_el_ang']:.1f}  >{STRAIGHT_ANGLE}",
         (0,255,80) if _dbg['c_el_ang'] >= STRAIGHT_ANGLE else (180,180,180)),
        (f"C_HOOK_SW: {_dbg['c_hook_n']:.3f}  >{DYN_HOOK_THRESH}",
         (0,200,255) if _dbg['c_hook_n'] > DYN_HOOK_THRESH else (180,180,180)),
    ]
    for i, (txt, col) in enumerate(rows):
        cv2.putText(frame, txt, (10, h - 185 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)


def draw_punch_trails(frame, lms, fw, fh):
    if not _show_trail:
        return
    jw, je = _JAB_IDX[0], _JAB_IDX[1]
    cw, ce = _CROSS_IDX[0], _CROSS_IDX[1]
    j_px    = (int(lms[jw].x * fw), int(lms[jw].y * fh))
    c_px    = (int(lms[cw].x * fw), int(lms[cw].y * fh))
    j_el_px = (int(lms[je].x * fw), int(lms[je].y * fh))
    c_el_px = (int(lms[ce].x * fw), int(lms[ce].y * fh))

    def cur_state(ext, hook_n):
        if ext    > DYN_EXTEND_THRESH * 0.6: return 'fwd'
        if hook_n > DYN_HOOK_THRESH   * 0.6: return 'side'
        return 'idle'

    _jab_trail.append(j_px[0],  j_px[1],  cur_state(_dbg["j_ext"], _dbg["j_hook_n"]))
    _cross_trail.append(c_px[0], c_px[1], cur_state(_dbg["c_ext"], _dbg["c_hook_n"]))

    for trail in (_jab_trail, _cross_trail):
        pts = list(trail)
        n   = len(pts)
        for i in range(1, n):
            alpha = (i + 1) / n
            col   = _TRAIL_COLORS.get(pts[i][2], (100, 100, 100))
            col   = tuple(int(c * alpha) for c in col)
            cv2.line(frame, (pts[i-1][0], pts[i-1][1]), (pts[i][0], pts[i][1]), col, max(1, int(3 * alpha)))
            cv2.circle(frame, (pts[i][0], pts[i][1]), max(2, int(5 * alpha)), col, -1)

    for el_pt, ang in [(j_el_px, _dbg["j_el_ang"]), (c_el_px, _dbg["c_el_ang"])]:
        ratio  = min(ang / STRAIGHT_ANGLE, 1.0)
        el_col = (0, int(255 * ratio), int(255 * (1 - ratio)))
        cv2.circle(frame, el_pt, 10, el_col, -1)
        cv2.circle(frame, el_pt, 10, (255, 255, 255), 1)

    ext_r    = max(0.0, _dbg["j_ext"]) / max(DYN_EXTEND_THRESH, 1e-6)
    ring_sz  = max(5, int(14 + 22 * min(ext_r, 2.0)))
    ring_col = (0, 255, 80) if ext_r >= 1.0 else (0, int(200 * ext_r), 80)
    cv2.circle(frame, j_px, ring_sz, ring_col, 2)

    def cond_text(pt, ext, ang, hook_n):
        ty = max(pt[1] - 55, 15);  tx = max(pt[0] - 55, 0)
        if ext > DYN_EXTEND_THRESH:
            msg, col = ("JAB/CROSS OK!", (0, 255, 80)) if ang >= STRAIGHT_ANGLE \
                       else (f"EXTEND OK  ANG:{ang:.0f}<{STRAIGHT_ANGLE}", (0, 165, 255))
        elif hook_n > DYN_HOOK_THRESH:
            msg, col = ("HOOK OK!", (0, 180, 255)) if HOOK_ANGLE_MIN <= ang <= HOOK_ANGLE_MAX \
                       else (f"SWING OK  ANG:{ang:.0f}?", (0, 165, 255))
        else:
            msg, col = f"EXT:{ext:.2f}  HK:{hook_n:.2f}", (140, 140, 140)
        cv2.putText(frame, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    cond_text(j_px, _dbg["j_ext"], _dbg["j_el_ang"], _dbg["j_hook_n"])
    cond_text(c_px, _dbg["c_ext"], _dbg["c_el_ang"], _dbg["c_hook_n"])

    legend = [("원(잽)", 'jab'), ("투(크로스)", 'cross'), ("훅", 'hook'),
              ("어퍼컷", 'uppercut'), ("전진방향", 'fwd'), ("측면방향", 'side')]
    for i, (txt, key) in enumerate(legend):
        col = _TRAIL_COLORS[key]
        lx  = fw - 130
        ly  = fh - 147 + i * 22
        cv2.circle(frame, (lx, ly), 5, col, -1)
        cv2.putText(frame, txt, (lx + 12, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)


def draw_skeleton(image, landmarks):
    h, w = image.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(image, pts[a], pts[b], (180, 180, 180), 2)
    for pt in pts:
        cv2.circle(image, pt, 4, (255, 255, 255), -1)

def score_bar(image, x, y, score, label, color):
    bar_w = 160
    cv2.rectangle(image, (x, y), (x + bar_w, y + 14), (60, 60, 60), -1)
    fill = int(bar_w * score / 100)
    cv2.rectangle(image, (x, y), (x + fill, y + 14), color, -1)
    cv2.putText(image, f"{label}: {score}pt", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


cap             = cv2.VideoCapture(0)
ankle_y_history = []
frame_count     = 0
print("카넬로 코치 1 (원투훅어퍼컷 감지 포함) 활성화! 종료: Q")
print("정면/사선/측면 모두 자동 감지됩니다.")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now   = time.time()
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms == 0:
            timestamp_ms = frame_count * 33

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # ── 포즈 안정 감지 체크 ───────────────────────────────────────────────
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            _pose_stable = min(_pose_stable + 1, _READY_FRAMES)
        else:
            _pose_stable = max(_pose_stable - 2, 0)
        if _pose_stable >= _READY_FRAMES:
            _pose_ready = True

        # ── 준비 안 됐으면 준비 화면 표시 후 계속 ─────────────────────────────
        if not _pose_ready:
            h_rdy, w_rdy = frame.shape[:2]

            # 포즈 감지되면 뼈대 표시 + 앵글 미리 계산
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                _lms_rdy = results.pose_landmarks[0]
                draw_skeleton(frame, _lms_rdy)
                _l_sh_rdy = lm(_lms_rdy, 11)
                _r_sh_rdy = lm(_lms_rdy, 12)
                _sw_rdy = float(np.linalg.norm(_l_sh_rdy[:2] - _r_sh_rdy[:2]))
                if _sw_rdy >= FRONT_SW:    _tgt = 0.0
                elif _sw_rdy <= SIDE_SW:   _tgt = 1.0
                else:                      _tgt = (FRONT_SW - _sw_rdy) / (FRONT_SW - SIDE_SW)
                _view_mode = _view_mode * 0.70 + _tgt * 0.30

            ov_rdy = frame.copy()
            cv2.rectangle(ov_rdy, (0, 0), (w_rdy, h_rdy), (0, 0, 0), -1)
            cv2.addWeighted(ov_rdy, 0.5, frame, 0.5, 0, frame)

            # 카메라 앵글 실시간 표시
            side_pct = int(_view_mode * 100)
            if side_pct < 25:
                angle_txt, angle_col = f"정면 ({100-side_pct}%)", (0, 255, 100)
            elif side_pct < 60:
                angle_txt, angle_col = f"사선 ({side_pct}%)", (0, 200, 255)
            else:
                angle_txt, angle_col = f"측면 ({side_pct}%)", (0, 165, 255)

            put_kr(frame, "CANELO AI COACH", (w_rdy//2 - 130, 40), (0, 60, 220), _KR_MD)
            put_kr(frame, f"카메라 각도: {angle_txt}", (w_rdy//2 - 120, h_rdy//2 - 60),
                   angle_col, _KR_MD)

            # 감지 진행 바
            bar_x, bar_y = w_rdy//4, h_rdy//2
            bar_w_px = w_rdy//2
            progress = _pose_stable / _READY_FRAMES
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_px, bar_y + 16), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + int(bar_w_px * progress), bar_y + 16),
                          (0, 200, 255), -1)

            if _pose_stable == 0:
                status_txt = "카메라 앞에 서주세요..."
                status_col = (0, 100, 255)
            else:
                status_txt = f"감지 중... ({int(progress*100)}%)"
                status_col = (0, 200, 255)
            put_kr(frame, status_txt, (w_rdy//2 - 90, h_rdy//2 + 25), status_col, _KR_SM)
            put_kr(frame, "정면 / 사선 / 측면 모두 가능합니다",
                   (w_rdy//2 - 130, h_rdy//2 + 60), (140, 140, 140), _KR_SM)

            cv2.imshow('Canelo AI Coach 1 (+Punch)', frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            continue

        # 배경 패널 (준비 완료 후에만)
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (310, 310), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "CANELO AI COACH  +PUNCH", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 60, 220), 2)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            lms = results.pose_landmarks[0]
            draw_skeleton(frame, lms)

            l_sh = lm(lms, 11);  r_sh = lm(lms, 12)
            shoulder_w = dist(l_sh, r_sh)
            if shoulder_w < 1e-4:
                shoulder_w = 0.2

            nose  = lm(lms, 0)
            l_wr  = lm(lms, 15);  r_wr  = lm(lms, 16)
            l_ank = lm(lms, 27);  r_ank = lm(lms, 28)

            head_to_sh = abs(((l_sh[1] + r_sh[1]) / 2) - nose[1])
            side_ref   = max(head_to_sh * 0.55, 0.05)
            eff_sw     = shoulder_w * (1 - _view_mode) + side_ref * _view_mode
            if eff_sw < 1e-4:
                eff_sw = 0.2
            shoulder_w = eff_sw

            l_gr = (l_wr[1] - nose[1]) / shoulder_w
            r_gr = (r_wr[1] - nose[1]) / shoulder_w

            def guard_pts(r):
                if r <= DNA["guard_perfect"]:   return 15
                if r <= DNA["guard_ok"]:        return 10
                if r <= DNA["guard_max_ratio"]: return 5
                return 0
            def guard_status(r, side):
                if r <= DNA["guard_perfect"]:   return f"{side} Guard Perfect!", (0, 255, 100)
                if r <= DNA["guard_ok"]:        return f"{side} Guard OK",        (0, 210, 100)
                if r <= DNA["guard_max_ratio"]: return f"{side} Guard LOW!",      (0, 165, 255)
                return f"{side} Guard DOWN!",                                      (0, 0, 255)

            l_pts = guard_pts(l_gr);  r_pts = guard_pts(r_gr)
            guard_score = l_pts + r_pts
            if l_gr >= r_gr:
                guard_msg, guard_col = guard_status(l_gr, "Left")
            else:
                guard_msg, guard_col = guard_status(r_gr, "Right")
            if l_pts == 15 and r_pts == 15:
                guard_msg, guard_col = "Perfect Guard!", (0, 255, 100)

            stance_ratio = dist(l_ank, r_ank) / shoulder_w
            if DNA["stance_ideal_min"] <= stance_ratio <= DNA["stance_ideal_max"]:
                stance_score = 20;  stance_msg, stance_col = f"Perfect Stance! ({stance_ratio:.2f}x)", (0, 255, 100)
            elif DNA["stance_ratio_min"] <= stance_ratio < DNA["stance_ideal_min"]:
                stance_score = 12;  stance_msg, stance_col = f"Slightly Narrow ({stance_ratio:.2f}x)", (0, 210, 100)
            elif DNA["stance_ideal_max"] < stance_ratio <= DNA["stance_ratio_max"]:
                stance_score = 12;  stance_msg, stance_col = f"Slightly Wide ({stance_ratio:.2f}x)", (0, 210, 100)
            elif stance_ratio < DNA["stance_ratio_min"]:
                stance_score = max(0, int(12 - (DNA["stance_ratio_min"] - stance_ratio) * 30))
                stance_msg, stance_col = f"TOO NARROW ({stance_ratio:.2f}x)", (0, 165, 255)
            else:
                stance_score = max(0, int(12 - (stance_ratio - DNA["stance_ratio_max"]) * 30))
                stance_msg, stance_col = f"TOO WIDE ({stance_ratio:.2f}x)", (0, 165, 255)

            tilt = abs(l_sh[1] - r_sh[1]) / shoulder_w
            if tilt <= DNA["shoulder_tilt_ok"]:
                shoulder_score = 15;  shoulder_msg, shoulder_col = "Shoulders Level!", (0, 255, 100)
            elif tilt <= DNA["shoulder_tilt_max"]:
                shoulder_score = 8;   shoulder_msg, shoulder_col = f"Slightly Tilted ({tilt:.2f})", (0, 210, 100)
            else:
                shoulder_score = 0;   shoulder_msg, shoulder_col = f"Shoulders Tilted! ({tilt:.2f})", (0, 0, 255)

            head_height = (((l_sh[1] + r_sh[1]) / 2) - nose[1]) / shoulder_w
            if head_height >= DNA["head_height_good"]:
                head_score = 15;  head_msg, head_col = "Head Up!", (0, 255, 100)
            elif head_height >= DNA["head_height_min"]:
                head_score = 8;   head_msg, head_col = f"Head Slightly Low ({head_height:.2f})", (0, 210, 100)
            else:
                head_score = 0;   head_msg, head_col = "Chin DOWN! Head up!", (0, 0, 255)

            ankle_y_history.append(l_ank[1] / shoulder_w)
            if len(ankle_y_history) > 30:
                ankle_y_history.pop(0)
            bounce = np.std(ankle_y_history) if len(ankle_y_history) == 30 else 0
            if bounce >= DNA["bounce_target"]:
                bounce_score = 20;  bounce_msg, bounce_col = f"Active Step! ({bounce:.3f})", (0, 255, 100)
            elif bounce >= DNA["bounce_min"]:
                ratio_b = (bounce - DNA["bounce_min"]) / (DNA["bounce_target"] - DNA["bounce_min"])
                bounce_score = int(ratio_b * 20)
                bounce_msg, bounce_col = f"Move your feet ({bounce:.3f})", (0, 210, 100)
            else:
                bounce_score = 0;  bounce_msg, bounce_col = f"STATIC! Step! ({bounce:.3f})", (0, 0, 255)

            detect_punch(lms, now)
            h_f, w_f = frame.shape[:2]
            draw_punch_guide(frame, lms, w_f, h_f)
            draw_punch_trails(frame, lms, w_f, h_f)

            total = guard_score + stance_score + shoulder_score + head_score + bounce_score
            if total >= 85:   total_col, grade = (0, 255, 100), "S"
            elif total >= 70: total_col, grade = (0, 200, 255), "A"
            elif total >= 50: total_col, grade = (0, 165, 255), "B"
            else:             total_col, grade = (0, 0, 255),   "C"

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
                cv2.putText(frame, msg, (10, h_img - 130 + i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        else:
            cv2.putText(frame, "자세를 감지할 수 없습니다", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

        draw_punch_ui(frame, now)
        draw_debug_ui(frame)

        # ── 캘리브레이션 UI ────────────────────────────────────────────────────
        if not _calib_done:
            h_f, w_f = frame.shape[:2]
            elapsed  = (now - _calib_start) if _calib_start else 0.0
            remain   = max(0.0, _CALIB_DURATION - elapsed)
            ov2 = frame.copy()
            cv2.rectangle(ov2, (w_f//4, h_f//2 - 45), (w_f*3//4, h_f//2 + 45), (0, 0, 0), -1)
            cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)
            put_kr(frame, "캘리브레이션: 잽·크로스를 몇 번 던져주세요!",
                   (w_f//4 + 10, h_f//2 - 32), (0, 255, 200), _KR_SM)
            cv2.putText(frame, f"남은 시간: {remain:.1f}s  (건너뛰려면 C키)",
                        (w_f//4 + 10, h_f//2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow('Canelo AI Coach 1 (+Punch)', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            _show_trail = not _show_trail
        elif key == ord('d'):
            _show_debug = not _show_debug
        elif key == ord('g'):
            _show_guide = not _show_guide
        elif key == ord('c'):
            _calib_done = True

cap.release()
cv2.destroyAllWindows()
