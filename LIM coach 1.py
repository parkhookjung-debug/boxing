"""
LIM coach 1.py  ─ 측면 기준 자세 코치
──────────────────────────────────────
LIM 선수의 자세 DNA를 기준으로 사용자의 복싱 자세를 실시간 교정합니다.
카메라는 측면(옆)에서 촬영, 어깨폭은 3D(X²+Z²) 거리로 계산합니다.

교정 항목
  ① 가드    : 손목 높이 + 손목 전방 위치(Z)
  ② 스텝    : 3D 발 간격, 앞뒤 스텝 길이, 무릎 굽힘 각도
  ③ 상체    : 전방 기울기 (측면 핵심)
  ④ 머리    : 높이, 전방 돌출

단축키
  Q / ESC   종료
  D         디버그 패널 토글
  S         스냅샷 저장 (PNG)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os
import csv
import time
import urllib.request
from collections import deque
from PIL import ImageFont, ImageDraw, Image as PILImage

# ══════════════════════════════════════════════════════════════════
# 모델 & DNA 로드
# ══════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = 'pose_landmarker_full.task'   # 상대 경로 (한글 경로 회피)

if not os.path.exists(MODEL_PATH):
    print("포즈 모델 다운로드 중 (약 30MB)...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_full/float16/latest/pose_landmarker_full.task',
        MODEL_PATH)
    print("완료!")

DNA_PATH = os.path.join(BASE_DIR, 'LIM_DNA.csv')

# ── 전체 fallback 기본값 (CSV에 없는 키도 항상 존재) ─────────────
_DEFAULTS = {
    'guard_l_ydiff'   : -0.12,
    'guard_r_ydiff'   : -0.08,
    'guard_l_zdiff'   : -0.30,
    'guard_r_zdiff'   : -0.20,
    'guard_l_xfwd'    : -0.40,
    'guard_r_xfwd'    :  0.10,
    'guard_lr_xdiff'  :  0.50,
    'guard_l_elbow_y' :  0.08,
    'guard_r_elbow_y' :  0.10,
    'stance_3d_ratio' :  1.50,
    'stance_step_x'   :  0.80,
    'knee_bend_l'     : 155.0,
    'knee_bend_r'     : 155.0,
    'lean_forward'    : -0.05,
    'head_y_ratio'    : -0.80,
    'head_fwd_z'      : -0.10,
    'shoulder_tilt'   :  0.04,
}

# CSV 컬럼명 → 코드 내부 키 매핑
_CSV_KEY_MAP = {
    'guard_l_xdiff'  : 'guard_l_xfwd',
    'guard_r_xdiff'  : 'guard_r_xfwd',
    'guard_l_elbow'  : 'guard_l_elbow_y',
    'guard_r_elbow'  : 'guard_r_elbow_y',
    'stance_w_ratio' : 'stance_3d_ratio',
    'lead_foot_fwd'  : 'stance_step_x',
    'chin_tuck'      : 'head_fwd_z',
}

def load_dna(path):
    if not os.path.exists(path):
        return None
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = {k: float(v) for k, v in row.items()
                   if not k.endswith('_std')}
            # 기본값으로 시작 후 CSV 값으로 덮어씀 (키 이름 매핑 포함)
            merged = dict(_DEFAULTS)
            for csv_k, val in raw.items():
                internal_k = _CSV_KEY_MAP.get(csv_k, csv_k)
                if internal_k in merged:
                    merged[internal_k] = val
            return merged
    return None

LIM = load_dna(DNA_PATH)
DNA_LOADED = LIM is not None

if not DNA_LOADED:
    print("[경고] LIM_DNA.csv 없음 — 기본값으로 실행합니다.")
    print("       먼저 'LIM data extraction.py' → 'LIM master average.py' 실행 필요")
    LIM = dict(_DEFAULTS)

# ── Punch DNA 로드 ────────────────────────────────────────────────
PUNCH_DNA_PATH = os.path.join(BASE_DIR, 'LIM_punch_DNA.csv')

def load_punch_dna(path):
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ptype = row['punch_type']
            result[ptype] = {k: float(v) for k, v in row.items()
                             if k not in ('punch_type', 'count')}
    return result

PUNCH_DNA       = load_punch_dna(PUNCH_DNA_PATH)
PUNCH_DNA_LOADED = bool(PUNCH_DNA)
if PUNCH_DNA_LOADED:
    print(f"[펀치 DNA] 로드 완료: {list(PUNCH_DNA.keys())}")
else:
    print("[경고] LIM_punch_DNA.csv 없음 — 'LIM punch extraction.py' 실행 필요")

PUNCH_TOL = {
    'arm_extension': 0.10,
    'elbow_angle'  : 25.0,
    'lean_forward' : 0.10,
}

# ── 허용 오차 ─────────────────────────────────────────────────────
TOL = {
    'guard_ydiff'   : 0.18,   # 손목 높이 허용 범위
    'guard_zdiff'   : 0.25,   # 손목 전방 위치 허용 범위
    'guard_xfwd'    : 0.25,   # 손목 X 위치 허용 범위 (앞뒤)
    'guard_lr_xdiff': 0.25,   # 앞손-뒷손 간격 허용 범위
    'stance_3d'     : 0.35,   # 3D 발 간격 허용 범위
    'stance_step'   : 0.30,   # 앞뒤 스텝 허용 범위
    'knee_bend'     : 15.0,   # 무릎 각도 허용 범위 (도)
    'lean'          : 0.12,   # 상체 기울기 허용 범위
    'head_y'        : 0.25,   # 머리 높이 허용 범위
    'head_fwd'      : 0.20,   # 머리 전방 돌출 허용 범위
    'shoulder_tilt' : 0.10,
}

# ══════════════════════════════════════════════════════════════════
# MediaPipe 초기화
# ══════════════════════════════════════════════════════════════════
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
    (0, 11), (0, 12),
]

# 수평 flip 후 MediaPipe가 좌/우를 바꿔 labeling하는 문제 교정
_LR_SWAP = [(11,12),(13,14),(15,16),(17,18),(19,20),(21,22),
            (23,24),(25,26),(27,28),(29,30),(31,32)]

def fix_flip_lr(lms):
    lms = list(lms)
    for a, b in _LR_SWAP:
        lms[a], lms[b] = lms[b], lms[a]
    return lms

# ══════════════════════════════════════════════════════════════════
# 한글 폰트
# ══════════════════════════════════════════════════════════════════
def _load_font(size):
    for path in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc",
                 "C:/Windows/Fonts/batang.ttc"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

_KR_SM = _load_font(16)
_KR_MD = _load_font(24)
_KR_LG = _load_font(34)

def put_kr(img, text, pos, color, font=None):
    if font is None:
        font = _KR_MD
    pil  = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ══════════════════════════════════════════════════════════════════
# 헬퍼
# ══════════════════════════════════════════════════════════════════
def lm2(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y])

def lm3(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y, p.z])

def px(pt, w, h):
    return (int(pt[0] * w), int(pt[1] * h))

def angle3pt(ax, ay, bx, by, cx, cy):
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag = math.sqrt(bax**2 + bay**2) * math.sqrt(bcx**2 + bcy**2) + 1e-9
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))

def sw_3d(lms):
    """3D 어깨폭: X²+Z² — 측면도 정확"""
    l = lms[11]; r = lms[12]
    return math.sqrt((r.x - l.x)**2 + (r.z - l.z)**2) + 1e-6

def draw_skeleton(frame, lms, w, h):
    pts = [lm2(lms, i) for i in range(33)]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, px(pts[a], w, h), px(pts[b], w, h), (160, 160, 160), 2)
    for pt in pts:
        cv2.circle(frame, px(pt, w, h), 4, (0, 220, 255), -1)

# ══════════════════════════════════════════════════════════════════
# 자세 분석 상태
# ══════════════════════════════════════════════════════════════════
_feedback_queue  = deque(maxlen=5)
_last_feedback   = {}
FEEDBACK_COOLDOWN = 3.0
_score_hist      = deque(maxlen=60)
_ankle_y_hist    = deque(maxlen=30)

_READY_FRAMES = 25
_pose_stable  = 0
_pose_ready   = False
_show_debug   = False
_snap_count   = 0

# ── Motion Trail ──────────────────────────────────────────────────
_trail_l  = deque(maxlen=20)
_trail_r  = deque(maxlen=20)
_prev_lwr = None
_prev_rwr = None

# ── Score Pop-ups ─────────────────────────────────────────────────
_score_pops  = []
_pop_cd      = {}
POP_DURATION = 2.2

# ── Ghost Form ────────────────────────────────────────────────────
_show_ghost = True

# ── Punch Form 분석 ───────────────────────────────────────────────
_lm_buffer     = deque(maxlen=25)   # 최근 랜드마크 버퍼
_pending_punch = []                  # [(countdown, punch_type), ...]
EXTENSION_DELAY = 8                  # 속도 피크 후 분석까지 대기 프레임

def give_feedback(key, text, color=(0, 200, 255)):
    now = time.time()
    if now - _last_feedback.get(key, 0) < FEEDBACK_COOLDOWN:
        return
    _last_feedback[key] = now
    _feedback_queue.append((text, color, now + 4.5))

def add_pop(key, text, x, y, color, cooldown=5.0):
    now = time.time()
    if now - _pop_cd.get(key, 0) < cooldown:
        return
    _pop_cd[key] = now
    _score_pops.append({'text': text, 'x': x, 'y': y,
                        'start': now, 'color': color})

# ══════════════════════════════════════════════════════════════════
# 핵심 자세 분석
# ══════════════════════════════════════════════════════════════════
def analyse_pose(lms, w, h, frame, now):
    score = 100

    # ── 기준 좌표 ─────────────────────────────────────────────────
    l_sh = lms[11]; r_sh = lms[12]
    l_el = lms[13]
    l_wr = lms[15]; r_wr = lms[16]
    nose = lms[0]
    l_hi = lms[23]; r_hi = lms[24]
    l_kn = lms[25]; r_kn = lms[26]
    l_an = lms[27]; r_an = lms[28]

    sw   = sw_3d(lms)
    sh_cx = (l_sh.x + r_sh.x) / 2
    sh_cy = (l_sh.y + r_sh.y) / 2
    sh_cz = (l_sh.z + r_sh.z) / 2
    hi_cx = (l_hi.x + r_hi.x) / 2

    # ══════════════════════════════════════════════════════════════
    # ① 가드 — 높이 + 전방 위치(Z)
    # ══════════════════════════════════════════════════════════════
    l_ydiff = (l_wr.y - l_sh.y) / sw
    r_ydiff = (r_wr.y - r_sh.y) / sw
    l_zdiff = (l_wr.z - l_sh.z) / sw
    r_zdiff = (r_wr.z - r_sh.z) / sw

    # 왼손 높이
    err_ly = l_ydiff - LIM['guard_l_ydiff']
    if abs(err_ly) > TOL['guard_ydiff']:
        score -= 15
        if err_ly > 0:
            give_feedback('g_l_low',  '왼손 올려 — 가드가 내려가 있어', (0, 80, 255))
            add_pop('p_g_l_low', '가드 내려감  −15', l_wr.x, l_wr.y, (0, 80, 255))
        else:
            give_feedback('g_l_high', '왼손 조금 내려 — 가드가 너무 높아', (0, 180, 255))
            add_pop('p_g_l_high', '가드 너무 높아  −15', l_wr.x, l_wr.y, (0, 180, 255))

    # 오른손 높이
    err_ry = r_ydiff - LIM['guard_r_ydiff']
    if abs(err_ry) > TOL['guard_ydiff']:
        score -= 15
        if err_ry > 0:
            give_feedback('g_r_low',  '오른손 올려 — 가드가 내려가 있어', (0, 80, 255))
            add_pop('p_g_r_low', '오른 가드 내려감  −15', r_wr.x, r_wr.y, (0, 80, 255))
        else:
            give_feedback('g_r_high', '오른손 조금 내려 — 가드가 너무 높아', (0, 180, 255))

    # 왼손 전방 위치 (Z, 측면 핵심)
    err_lz = l_zdiff - LIM['guard_l_zdiff']
    if abs(err_lz) > TOL['guard_zdiff']:
        score -= 10
        if err_lz > 0:
            give_feedback('g_l_back', '왼손을 앞으로 더 내밀어 — 가드가 뒤로 빠져 있어',
                          (0, 140, 255))
        else:
            give_feedback('g_l_over', '왼손이 너무 앞으로 나와 있어', (0, 200, 200))

    # 오른손 전방 위치
    err_rz = r_zdiff - LIM['guard_r_zdiff']
    if abs(err_rz) > TOL['guard_zdiff']:
        score -= 10
        if err_rz > 0:
            give_feedback('g_r_back', '오른손을 앞으로 더 내밀어 — 가드가 뒤로 빠져 있어',
                          (0, 140, 255))
        else:
            give_feedback('g_r_over', '오른손이 너무 앞으로 나와 있어', (0, 200, 200))

    # ── 가드 위치: 측면 앞뒤 X 위치 체크 ────────────────────────────
    l_xfwd = (l_wr.x - sh_cx) / sw
    r_xfwd = (r_wr.x - sh_cx) / sw
    lr_xdiff = abs(l_wr.x - r_wr.x) / sw

    err_lxf = l_xfwd - LIM.get('guard_l_xfwd', -0.40)
    err_rxf = r_xfwd - LIM.get('guard_r_xfwd',  0.10)
    err_lrd = lr_xdiff - LIM.get('guard_lr_xdiff', 0.50)

    if abs(err_lxf) > TOL['guard_xfwd']:
        score -= 8
        if err_lxf > 0:
            give_feedback('g_l_xback', '앞손(잽)을 더 앞으로 뻗어 — 가드 위치가 뒤로 빠짐',
                          (0, 160, 255))
        else:
            give_feedback('g_l_xover', '앞손이 너무 앞으로 나와 있어 — 가드를 당겨',
                          (0, 200, 200))

    if abs(err_rxf) > TOL['guard_xfwd']:
        score -= 8
        if err_rxf < 0:
            give_feedback('g_r_xfwd', '뒷손(크로스)을 턱 쪽으로 — 가드 위치 교정',
                          (0, 160, 255))

    if err_lrd < -TOL['guard_lr_xdiff']:
        score -= 6
        give_feedback('g_spread', '두 손 간격이 너무 좁아 — 앞손을 더 앞으로',
                      (0, 160, 200))

    # 가드 시각화: 손목에 OK/NG 원 (높이+Z+X 모두 OK여야 초록)
    guard_l_ok = (abs(err_ly)  <= TOL['guard_ydiff'] and
                  abs(err_lz)  <= TOL['guard_zdiff'] and
                  abs(err_lxf) <= TOL['guard_xfwd'])
    guard_r_ok = (abs(err_ry)  <= TOL['guard_ydiff'] and
                  abs(err_rz)  <= TOL['guard_zdiff'] and
                  abs(err_rxf) <= TOL['guard_xfwd'])
    for wr, ey, ez, exf in [(l_wr, err_ly, err_lz, err_lxf),
                             (r_wr, err_ry, err_rz, err_rxf)]:
        cx_ = int(wr.x * w); cy_ = int(wr.y * h)
        ok  = (abs(ey)  <= TOL['guard_ydiff'] and
               abs(ez)  <= TOL['guard_zdiff'] and
               abs(exf) <= TOL['guard_xfwd'])
        cv2.circle(frame, (cx_, cy_), 18, (0, 220, 80) if ok else (0, 60, 255), 3)

    if guard_l_ok and guard_r_ok:
        mx = (l_wr.x + r_wr.x) / 2
        my = min(l_wr.y, r_wr.y) - 0.06
        add_pop('p_guard_ok', '가드 완벽 ✓  +8', mx, my, (0, 220, 80), 8.0)

    # ══════════════════════════════════════════════════════════════
    # ② 스텝 / 스탠스
    # ══════════════════════════════════════════════════════════════
    # 3D 발목 간격
    ankle_3d  = math.sqrt((r_an.x - l_an.x)**2 + (r_an.z - l_an.z)**2)
    stance_3d = ankle_3d / sw
    err_3d    = stance_3d - LIM['stance_3d_ratio']
    if abs(err_3d) > TOL['stance_3d']:
        score -= 12
        if err_3d < 0:
            give_feedback('st_narrow', '발 간격 더 벌려 — 스탠스가 좁아', (80, 220, 80))
            add_pop('p_st_narrow', '스탠스 좁아  −12', sh_cx, 0.85, (80, 220, 80))
        else:
            give_feedback('st_wide',   '발 간격 좁혀 — 스탠스가 너무 넓어', (80, 220, 80))
            add_pop('p_st_wide', '스탠스 너무 넓어  −12', sh_cx, 0.85, (80, 220, 80))

    # 앞뒤 스텝 길이 (측면 X 분리)
    step_x   = abs(r_an.x - l_an.x) / sw
    err_step = step_x - LIM['stance_step_x']
    if abs(err_step) > TOL['stance_step']:
        score -= 8
        if err_step < 0:
            give_feedback('st_step_short', '앞발을 더 앞으로 내밀어 — 스텝이 짧아',
                          (80, 220, 80))
        else:
            give_feedback('st_step_long', '스텝이 너무 길어 — 발 간격을 줄여',
                          (80, 220, 80))

    # 무릎 굽힘 각도
    ang_l = angle3pt(l_hi.x, l_hi.y, l_kn.x, l_kn.y, l_an.x, l_an.y)
    ang_r = angle3pt(r_hi.x, r_hi.y, r_kn.x, r_kn.y, r_an.x, r_an.y)
    ref_kl = LIM['knee_bend_l']; ref_kr = LIM['knee_bend_r']

    if ang_l > ref_kl + TOL['knee_bend']:
        score -= 8
        give_feedback('knee_l_str', '왼쪽 무릎 더 굽혀 — 중심을 낮춰', (100, 220, 100))
    if ang_r > ref_kr + TOL['knee_bend']:
        score -= 8
        give_feedback('knee_r_str', '오른쪽 무릎 더 굽혀 — 중심을 낮춰', (100, 220, 100))

    # 무릎 각도 시각화 (발목 위에 표시)
    for kn, ang, ref in [(l_kn, ang_l, ref_kl), (r_kn, ang_r, ref_kr)]:
        col = (0, 220, 80) if abs(ang - ref) <= TOL['knee_bend'] else (0, 80, 255)
        kx  = int(kn.x * w); ky = int(kn.y * h)
        cv2.circle(frame, (kx, ky), 10, col, 3)
        cv2.putText(frame, f"{int(ang)}", (kx + 12, ky - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # 발 간격 선 시각화
    lax = int(l_an.x * w); lay = int(l_an.y * h)
    rax = int(r_an.x * w); ray = int(r_an.y * h)
    st_col = (0, 220, 80) if abs(err_3d) <= TOL['stance_3d'] else (0, 60, 255)
    cv2.line(frame, (lax, lay + 8), (rax, ray + 8), st_col, 4)
    cv2.circle(frame, (lax, lay), 9, st_col, -1)
    cv2.circle(frame, (rax, ray), 9, st_col, -1)

    # 바운스 안정성
    avg_an_y = (l_an.y + r_an.y) / 2
    _ankle_y_hist.append(avg_an_y)
    if len(_ankle_y_hist) >= 15:
        bounce = (max(_ankle_y_hist) - min(_ankle_y_hist)) / sw
        if bounce > 0.20:
            score -= 5
            give_feedback('bounce', '너무 많이 튀고 있어 — 안정적으로', (150, 200, 80))

    # ══════════════════════════════════════════════════════════════
    # ③ 상체 전방 기울기 (측면 핵심)
    # ══════════════════════════════════════════════════════════════
    lean = (sh_cx - hi_cx) / sw
    err_lean = lean - LIM['lean_forward']
    if abs(err_lean) > TOL['lean']:
        score -= 10
        if err_lean > 0:
            give_feedback('lean_fwd',  '상체를 너무 앞으로 기울이고 있어', (200, 160, 0))
            add_pop('p_lean_fwd', '상체 과기울기  −10', sh_cx, 0.45, (200, 160, 0))
        else:
            give_feedback('lean_back', '상체를 앞으로 조금 더 숙여', (200, 160, 0))
            add_pop('p_lean_back', '상체 세워  −10', sh_cx, 0.45, (200, 160, 0))

    # 어깨 기울기
    sh_tilt = (r_sh.y - l_sh.y) / sw
    if abs(sh_tilt - LIM['shoulder_tilt']) > TOL['shoulder_tilt']:
        score -= 8
        give_feedback('sh_tilt', '어깨 수평을 맞춰', (200, 150, 50))

    # ══════════════════════════════════════════════════════════════
    # ④ 머리 자세
    # ══════════════════════════════════════════════════════════════
    head_y  = (nose.y - sh_cy) / sw
    err_hy  = head_y - LIM['head_y_ratio']
    if abs(err_hy) > TOL['head_y']:
        score -= 8
        if err_hy > 0:
            give_feedback('head_low',  '머리 들어 — 고개가 너무 숙여져 있어', (200, 100, 200))
        else:
            give_feedback('head_high', '턱을 당겨 — 고개가 너무 들려 있어', (200, 100, 200))

    head_fz  = (nose.z - sh_cz) / sw
    err_hfz  = head_fz - LIM['head_fwd_z']
    if abs(err_hfz) > TOL['head_fwd']:
        score -= 5
        if err_hfz < 0:
            give_feedback('chin_out', '턱을 당겨 — 얼굴이 너무 앞으로 나와 있어',
                          (200, 100, 200))

    score = max(0, min(100, score))
    _score_hist.append(score)
    return score


# ══════════════════════════════════════════════════════════════════
# 준비 화면
# ══════════════════════════════════════════════════════════════════
def draw_ready_screen(frame, pose_detected, stable_cnt, w, h):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    put_kr(frame, 'LIM 코치  (측면 모드)', (w // 2 - 130, 60), (80, 200, 255), _KR_LG)
    put_kr(frame, 'LIM 선수의 측면 자세를 기준으로 교정합니다', (w // 2 - 210, 110),
           (200, 200, 200), _KR_SM)

    status = '✓ LIM DNA 로드 완료' if DNA_LOADED else '⚠ DNA 없음 — 기본값 사용 중'
    scol   = (0, 220, 80) if DNA_LOADED else (0, 120, 255)
    put_kr(frame, status, (w // 2 - 130, 148), scol, _KR_SM)

    if pose_detected:
        prog = min(stable_cnt / _READY_FRAMES, 1.0)
        bx, by, bw, bh = w // 2 - 150, h // 2, 300, 22
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * prog), by + bh), (0, 200, 80), -1)
        put_kr(frame, f'자세 인식 중… {int(prog * 100)}%', (bx, by - 28),
               (200, 255, 200), _KR_SM)
    else:
        put_kr(frame, '카메라 옆에 서주세요 (전신이 보여야 합니다)',
               (w // 2 - 210, h // 2), (150, 150, 255), _KR_SM)

    tips = [
        '• 카메라 측면에서 전신이 보이도록 서주세요',
        '• 가드 자세로 시작하세요',
        '• 단축키: D=디버그  S=스냅샷  G=기준자세  Q=종료',
    ]
    for i, tip in enumerate(tips):
        put_kr(frame, tip, (40, h - 110 + i * 28), (160, 160, 160), _KR_SM)


# ══════════════════════════════════════════════════════════════════
# 피드백 패널
# ══════════════════════════════════════════════════════════════════
def draw_feedback_panel(frame, w, h, score, now):
    panel_w, panel_h = 330, 210
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    sc_col = (0, 220, 80) if score >= 80 else (0, 180, 255) if score >= 55 else (0, 60, 255)
    put_kr(frame, f'자세 점수: {score}', (10, 8), sc_col, _KR_MD)

    active = [(t, c, e) for t, c, e in _feedback_queue if e > now]
    for i, (text, color, _) in enumerate(reversed(list(active)[-4:])):
        put_kr(frame, f'▶ {text}', (10, 48 + i * 36), color, _KR_SM)
    if not active:
        put_kr(frame, '✓ 자세가 좋습니다!', (10, 48), (0, 220, 80), _KR_SM)

    # 점수 그래프
    if len(_score_hist) > 5:
        gh = 50; gx = 10; gy = 155; gw = panel_w - 20
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (30, 30, 40), -1)
        vals = list(_score_hist)
        for i in range(1, len(vals)):
            x1 = gx + int((i - 1) / 59 * gw)
            x2 = gx + int(i / 59 * gw)
            y1 = gy + gh - int(vals[i-1] / 100 * gh)
            y2 = gy + gh - int(vals[i] / 100 * gh)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 180, 120), 1)


def draw_debug_panel(frame, lms, w, score, sw_val):
    if lms is None:
        return
    l_sh = lms[11]; r_sh = lms[12]
    l_wr = lms[15]; r_wr = lms[16]
    l_hi = lms[23]; r_hi = lms[24]
    l_kn = lms[25]; r_kn = lms[26]
    l_an = lms[27]; r_an = lms[28]

    sh_cx = (l_sh.x + r_sh.x) / 2
    hi_cx = (l_hi.x + r_hi.x) / 2
    ang_l = angle3pt(l_sh.x, l_sh.y, l_kn.x, l_kn.y, l_an.x, l_an.y)
    ang_r = angle3pt(r_sh.x, r_sh.y, r_kn.x, r_kn.y, r_an.x, r_an.y)

    lines = [
        f"sw_3d={sw_val:.3f}",
        f"L손목Y={(l_wr.y-l_sh.y)/sw_val:+.3f} REF:{LIM['guard_l_ydiff']:+.3f}",
        f"R손목Y={(r_wr.y-r_sh.y)/sw_val:+.3f} REF:{LIM['guard_r_ydiff']:+.3f}",
        f"L손목Z={(l_wr.z-l_sh.z)/sw_val:+.3f} REF:{LIM['guard_l_zdiff']:+.3f}",
        f"R손목Z={(r_wr.z-r_sh.z)/sw_val:+.3f} REF:{LIM['guard_r_zdiff']:+.3f}",
        f"스탠스3D={math.sqrt((r_an.x-l_an.x)**2+(r_an.z-l_an.z)**2)/sw_val:.2f} REF:{LIM['stance_3d_ratio']:.2f}",
        f"스텝X={abs(r_an.x-l_an.x)/sw_val:.2f} REF:{LIM['stance_step_x']:.2f}",
        f"무릎L={int(ang_l)}° REF:{int(LIM['knee_bend_l'])}°",
        f"무릎R={int(ang_r)}° REF:{int(LIM['knee_bend_r'])}°",
        f"기울기={(sh_cx-hi_cx)/sw_val:+.3f} REF:{LIM['lean_forward']:+.3f}",
        f"앞손X={(l_wr.x-sh_cx)/sw_val:+.3f} REF:{LIM.get('guard_l_xfwd',-0.4):+.3f}",
        f"뒷손X={(r_wr.x-sh_cx)/sw_val:+.3f} REF:{LIM.get('guard_r_xfwd', 0.1):+.3f}",
        f"손간격X={abs(l_wr.x-r_wr.x)/sw_val:.3f} REF:{LIM.get('guard_lr_xdiff',0.5):.3f}",
        f"점수={score}",
    ]
    ox, oy = w - 300, 10
    cv2.rectangle(frame, (ox - 5, oy - 5), (w - 5, oy + len(lines) * 20 + 5),
                  (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (ox, oy + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


# ══════════════════════════════════════════════════════════════════
# Motion Trail — 손목 궤적 잔상
# ══════════════════════════════════════════════════════════════════
def update_trails(lms_raw):
    global _prev_lwr, _prev_rwr
    l_wr = lms_raw[15]
    r_wr = lms_raw[16]
    lx, ly = l_wr.x, l_wr.y
    rx, ry = r_wr.x, r_wr.y
    _trail_l.append((lx, ly))
    _trail_r.append((rx, ry))

    # 측면 뷰: X축 = 앞뒤 방향 → X가 더 작은 손이 앞손(잽), 큰 손이 뒷손(크로스)
    if lx < rx:
        lead_pos, lead_prev = (lx, ly), _prev_lwr
        rear_pos,  rear_prev  = (rx, ry), _prev_rwr
    else:
        lead_pos, lead_prev = (rx, ry), _prev_rwr
        rear_pos,  rear_prev  = (lx, ly), _prev_lwr

    if lead_prev is not None:
        d = math.sqrt((lead_pos[0]-lead_prev[0])**2 + (lead_pos[1]-lead_prev[1])**2)
        if d > 0.07:
            add_pop('p_jab', '잽 감지 ⚡', lead_pos[0], lead_pos[1] - 0.06, (255, 220, 50), 2.5)
            _pending_punch.append([EXTENSION_DELAY, 'jab'])
    if rear_prev is not None:
        d = math.sqrt((rear_pos[0]-rear_prev[0])**2 + (rear_pos[1]-rear_prev[1])**2)
        if d > 0.07:
            add_pop('p_cross', '크로스 감지 ⚡', rear_pos[0], rear_pos[1] - 0.06, (255, 180, 50), 2.5)
            _pending_punch.append([EXTENSION_DELAY, 'cross'])

    _prev_lwr = (lx, ly)
    _prev_rwr = (rx, ry)


def draw_motion_trail(frame, w, h):
    """손목 궤적을 잔상 라인으로 표시"""
    for trail, base in [(_trail_l, (0, 220, 255)), (_trail_r, (50, 160, 255))]:
        pts = list(trail)
        n   = len(pts)
        if n < 2:
            continue
        for i in range(1, n):
            a   = i / n
            col = tuple(int(c * a) for c in base)
            p1  = (int(pts[i-1][0] * w), int(pts[i-1][1] * h))
            p2  = (int(pts[i][0]   * w), int(pts[i][1]   * h))
            cv2.line(frame, p1, p2, col, max(1, int(4 * a)))
        tip = (int(pts[-1][0] * w), int(pts[-1][1] * h))
        cv2.circle(frame, tip, 6, base, -1)


# ══════════════════════════════════════════════════════════════════
# Ghost Form Overlay — LIM 기준 자세 가이드
# ══════════════════════════════════════════════════════════════════
def draw_ghost_form(frame, lms_raw, w, h, sw_val):
    """LIM DNA 기준 이상적 손목 위치를 반투명으로 표시"""
    l_sh  = lms_raw[11]; r_sh = lms_raw[12]
    sh_cx = (l_sh.x + r_sh.x) / 2

    targets = [
        (sh_cx + LIM.get('guard_l_xfwd', -0.40) * sw_val,
         l_sh.y + LIM['guard_l_ydiff']           * sw_val,
         (180, 100, 255)),
        (sh_cx + LIM.get('guard_r_xfwd',  0.10) * sw_val,
         r_sh.y + LIM['guard_r_ydiff']           * sw_val,
         (180, 100, 255)),
    ]

    ov = frame.copy()
    for (ix, iy, col) in targets:
        cx, cy = int(ix * w), int(iy * h)
        cv2.circle(ov, (cx, cy), 26, col, 2)
        cv2.circle(ov, (cx, cy),  6, col, -1)
        cv2.line(ov, (cx - 16, cy), (cx + 16, cy), col, 1)
        cv2.line(ov, (cx, cy - 16), (cx, cy + 16), col, 1)
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
    put_kr(frame, 'G — 기준 자세', (w - 170, 10), (180, 100, 255), _KR_SM)


# ══════════════════════════════════════════════════════════════════
# Score Pop-ups — 코칭 점수 팝업
# ══════════════════════════════════════════════════════════════════
def draw_score_pops(frame, w, h, now):
    _score_pops[:] = [p for p in _score_pops
                      if now - p['start'] <= POP_DURATION]
    if not _score_pops:
        return

    pil  = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    tmp  = PILImage.new('RGBA', pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    for pop in _score_pops:
        age   = now - pop['start']
        alpha = max(0.0, 1.0 - age / POP_DURATION)
        rise  = int(age * 70)
        sx = max(5, min(w - 240, int(pop['x'] * w)))
        sy = max(30, int(pop['y'] * h) - rise)
        b, g, r = pop['color']               # BGR → RGB swap
        draw.text((sx, sy), pop['text'], font=_KR_MD,
                  fill=(r, g, b, int(alpha * 230)))
    merged    = PILImage.alpha_composite(pil, tmp).convert('RGB')
    frame[:]  = cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════════
# Punch Form 분석 — DNA 비교 피드백
# ══════════════════════════════════════════════════════════════════
def analyse_punch_form(punch_type):
    """버퍼에서 팔 최대 뻗음 시점을 찾아 PUNCH_DNA와 비교"""
    if not PUNCH_DNA_LOADED or punch_type not in PUNCH_DNA:
        return
    buf = list(_lm_buffer)
    if len(buf) < 3:
        return

    label = '잽' if punch_type == 'jab' else '크로스'
    dna   = PUNCH_DNA[punch_type]

    # 앞손/뒷손 판별 (버퍼 마지막 프레임 기준)
    last = buf[-1]
    lx, rx = last[15].x, last[16].x
    if punch_type == 'jab':
        wr_idx = 15 if lx < rx else 16
    else:
        wr_idx = 16 if lx < rx else 15
    sh_idx = 11 if wr_idx == 15 else 12
    el_idx = 13 if wr_idx == 15 else 14

    # 버퍼에서 팔 뻗음 최대 시점 탐색
    best_lms, best_dist = buf[-1], 0.0
    for lms in buf:
        wr = lms[wr_idx]; sh = lms[sh_idx]
        d  = math.sqrt((wr.x - sh.x)**2 + (wr.y - sh.y)**2)
        if d > best_dist:
            best_dist = d
            best_lms  = lms

    # 메트릭 추출
    lms    = best_lms
    sw_val = sw_3d(lms)
    wr = lms[wr_idx]; sh = lms[sh_idx]; el = lms[el_idx]
    l_sh = lms[11];   r_sh = lms[12]
    sh_cx = (l_sh.x + r_sh.x) / 2
    hi_cx = (lms[23].x + lms[24].x) / 2

    arm_ext  = math.sqrt((wr.x - sh.x)**2 + (wr.y - sh.y)**2) / sw_val
    el_ang   = angle3pt(sh.x, sh.y, el.x, el.y, wr.x, wr.y)
    lean     = (sh_cx - hi_cx) / sw_val

    ref_ext  = dna['arm_extension_avg']
    ref_ang  = dna['elbow_angle_avg']
    ref_lean = dna['lean_forward_avg']

    # ── 피드백 ────────────────────────────────────────────────────
    if arm_ext < ref_ext - PUNCH_TOL['arm_extension']:
        give_feedback(f'pf_{punch_type}_ext',
                      f'{label} — 팔을 더 뻗어! ({arm_ext:.2f} / LIM {ref_ext:.2f})',
                      (0, 140, 255))
        add_pop(f'pp_{punch_type}_ext', f'{label} 팔 뻗음 부족  −8',
                wr.x, wr.y - 0.07, (0, 140, 255), 3.0)
    else:
        add_pop(f'pp_{punch_type}_ext_ok', f'{label} 뻗음 ✓  +8',
                wr.x, wr.y - 0.07, (0, 220, 80), 3.0)

    if el_ang < ref_ang - PUNCH_TOL['elbow_angle']:
        give_feedback(f'pf_{punch_type}_ang',
                      f'{label} — 팔꿈치 더 펴! ({int(el_ang)}° / LIM {int(ref_ang)}°)',
                      (0, 160, 255))
        add_pop(f'pp_{punch_type}_ang', f'{label} 팔꿈치 각도  −6',
                wr.x, wr.y - 0.12, (0, 160, 255), 3.0)

    if abs(lean - ref_lean) > PUNCH_TOL['lean_forward']:
        msg = f'{label} — 상체 더 숙여' if lean > ref_lean else f'{label} — 상체 너무 앞'
        give_feedback(f'pf_{punch_type}_lean', msg, (200, 160, 0))


# ══════════════════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()
        ts_ms = int(now * 1000)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res    = landmarker.detect_for_video(mp_img, ts_ms)

        pose_detected = (res.pose_landmarks and len(res.pose_landmarks) > 0)
        lms_raw = fix_flip_lr(res.pose_landmarks[0]) if pose_detected else None

        # ── 준비 화면 ──────────────────────────────────────────────
        if not _pose_ready:
            if pose_detected:
                _pose_stable = min(_pose_stable + 1, _READY_FRAMES)
            else:
                _pose_stable = max(_pose_stable - 2, 0)

            if lms_raw:
                draw_skeleton(frame, lms_raw, w, h)
            draw_ready_screen(frame, pose_detected, _pose_stable, w, h)

            if _pose_stable >= _READY_FRAMES:
                _pose_ready = True

            cv2.imshow('LIM Coach', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            continue

        # ── 분석 & 피드백 ──────────────────────────────────────────
        score  = 50
        sw_val = 0.0

        if pose_detected and lms_raw:
            _lm_buffer.append(lms_raw)
            # pending punch 카운트다운 → 완료 시 form 분석
            for entry in list(_pending_punch):
                entry[0] -= 1
                if entry[0] <= 0:
                    _pending_punch.remove(entry)
                    analyse_punch_form(entry[1])
            draw_skeleton(frame, lms_raw, w, h)
            sw_val = sw_3d(lms_raw)
            score  = analyse_pose(lms_raw, w, h, frame, now)
            update_trails(lms_raw)
            draw_motion_trail(frame, w, h)
            if _show_ghost:
                draw_ghost_form(frame, lms_raw, w, h, sw_val)
            if _show_debug:
                draw_debug_panel(frame, lms_raw, w, score, sw_val)
            if score >= 85:
                add_pop('p_perfect', '완벽한 자세! ✓', 0.35, 0.15,
                        (0, 255, 200), 10.0)
        else:
            give_feedback('lost', '자세를 잃었습니다 — 카메라 옆에 서주세요', (0, 80, 255))
            _pose_stable = max(_pose_stable - 2, 0)
            if _pose_stable < _READY_FRAMES // 2:
                _pose_ready = False

        draw_feedback_panel(frame, w, h, score, now)
        draw_score_pops(frame, w, h, now)

        dna_label = "LIM DNA ✓" if DNA_LOADED else "기본값 (DNA 없음)"
        put_kr(frame, dna_label, (w - 190, h - 30),
               (0, 200, 80) if DNA_LOADED else (0, 80, 255), _KR_SM)

        cv2.imshow('LIM Coach', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            _show_debug = not _show_debug
        elif key == ord('g'):
            _show_ghost = not _show_ghost
        elif key == ord('s'):
            _snap_count += 1
            fn = os.path.join(BASE_DIR, f'LIM_snapshot_{_snap_count:03d}.png')
            cv2.imwrite(fn, frame)
            give_feedback('snap', f'스냅샷 저장: {os.path.basename(fn)}', (200, 200, 80))

cap.release()
cv2.destroyAllWindows()
