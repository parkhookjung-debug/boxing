"""
LIM coach.py — 간단한 복싱 코치 (RTMO + ML + 룰)

설계 원칙
  ● 단일 OpenCV 창, 디자인 최대한 간단
  ● 잽 / 크로스 / 훅 / 어퍼 4종 인식 (학습 모델 우선, 룰 fallback)
  ● 자세 점수 (가드, 머리, 린포워드, 팔꿈치)
  ● 웹캠 실시간, FPS 표시

단축키:  Q 종료  R 카운트 초기화  D 스탠스 전환 (오르토독스/사우스포)
"""

import os, time, math, pickle
from collections import deque
import numpy as np
import cv2

try:
    from rtmlib import RTMO
except ImportError:
    raise SystemExit("pip install rtmlib onnxruntime")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── COCO 17 인덱스 ─────────────────────────────────────────
KP_NOSE = 0
KP_L_SH, KP_R_SH = 5, 6
KP_L_EL, KP_R_EL = 7, 8
KP_L_WR, KP_R_WR = 9, 10
KP_L_HI, KP_R_HI = 11, 12

COCO_LINES = [
    (KP_L_SH, KP_R_SH), (KP_L_SH, KP_L_EL), (KP_L_EL, KP_L_WR),
    (KP_R_SH, KP_R_EL), (KP_R_EL, KP_R_WR),
    (KP_L_SH, KP_L_HI), (KP_R_SH, KP_R_HI), (KP_L_HI, KP_R_HI),
    (KP_NOSE, KP_L_SH), (KP_NOSE, KP_R_SH),
]
NEEDED = [KP_L_SH, KP_R_SH, KP_L_EL, KP_R_EL, KP_L_WR, KP_R_WR]
VIS_MIN = 0.30

# ── 색상 (BGR) ──────────────────────────────────────────────
BG       = (22, 22, 24)
PANEL    = (34, 34, 38)
BORDER   = (60, 60, 66)
INK      = (235, 235, 235)
DIM      = (130, 130, 140)
ACCENT   = (87, 61, 255)
OK       = (90, 220, 110)
WARN     = (60, 170, 255)
SKELETON = (140, 140, 160)
JOINT    = (87, 61, 255)

PUNCH_COLOR = {
    'jab':      (0, 220, 255),
    'cross':    (255, 160, 40),
    'hook':     (200, 50, 255),
    'uppercut': (100, 255, 50),
}

# ── 모델 로드 ──────────────────────────────────────────────
print('RTMO 로드 중...')
try:
    import onnxruntime as ort
    device = 'dml' if 'DmlExecutionProvider' in ort.get_available_providers() else 'cpu'
except Exception:
    device = 'cpu'
print(f'  device: {device}')

RTMO_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
    'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip'
)
pose_model = RTMO(RTMO_URL, backend='onnxruntime', device=device)
print('RTMO 준비 완료\n')

# 펀치 분류 모델
PUNCH_MODEL = None
PUNCH_FEATS = None
PUNCH_WIN = 8
model_path = os.path.join(BASE_DIR, 'lim_punch_model.pkl')
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    PUNCH_MODEL = bundle['model']
    PUNCH_FEATS = bundle['features']
    PUNCH_WIN = bundle.get('window', 8)
    print(f'펀치 모델 로드: {bundle.get("classes")}')
else:
    print('펀치 모델 없음 — 룰베이스만 사용. (LIM_train.py 실행 권장)')

# ── 기하 헬퍼 ──────────────────────────────────────────────
def sw_px(kp):
    dx = kp[KP_R_SH][0] - kp[KP_L_SH][0]
    dy = kp[KP_R_SH][1] - kp[KP_L_SH][1]
    return math.hypot(dx, dy) + 1e-6


def angle3(ax, ay, bx, by, cx, cy):
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy) + 1e-9
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))


# ── 상태 ──────────────────────────────────────────────────
orthodox = True  # True: 왼손 잽
counts = {'jab': 0, 'cross': 0, 'hook': 0, 'uppercut': 0}
last_punch = None  # (type, side, time)

# 손목 히스토리 (정규 좌표): 펀치 감지 + 피처 계산용
HIST = 16
hist_L = deque(maxlen=HIST)  # (wx, wy, ex, ey, sx, sy, sw)
hist_R = deque(maxlen=HIST)

# 펀치 감지 상태
VEL_START = 0.06    # sw 기준 손목 속도 시작 임계
VEL_PEAK_MIN = 0.10 # 최소 피크 (잔진동 차단)
COOLDOWN_SAME = 0.30
COOLDOWN_OPP = 0.10
det = {
    'L': {'active': False, 'peak': 0.0, 'start': 0.0, 'settled': True},
    'R': {'active': False, 'peak': 0.0, 'start': 0.0, 'settled': True},
}
last_fire_t = {'L': 0.0, 'R': 0.0}
flash = None  # (punch_type, side, expire_t)


def punching_side_to_LR(side):
    """side: 'lead' or 'rear' → 'L' or 'R'."""
    if orthodox:
        return 'L' if side == 'lead' else 'R'
    return 'R' if side == 'lead' else 'L'


def LR_to_label(LR):
    """L/R → 'lead'/'rear' (현재 스탠스 기준)."""
    if orthodox:
        return 'lead' if LR == 'L' else 'rear'
    return 'rear' if LR == 'L' else 'lead'


# ── 피처 추출 ─────────────────────────────────────────────
def features_from_hist(hist):
    """hist: deque of (wx,wy,ex,ey,sx,sy,sw). 최근 프레임을 임팩트로 가정."""
    if len(hist) < 2:
        return None
    cur = hist[-1]
    wx, wy, ex, ey, sx, sy, s = cur

    arm_ext = math.hypot(wx - sx, wy - sy) / s
    ea = angle3(sx, sy, ex, ey, wx, wy)
    el_y = (ey - sy) / s
    el_lat = abs(ex - sx) / s
    wr_y = (wy - sy) / s
    ew = math.hypot(ex - wx, ey - wy) / s

    sub = list(hist)[-(PUNCH_WIN + 1):]
    wxs = np.array([h[0] for h in sub])
    wys = np.array([h[1] for h in sub])
    dwx = np.diff(wxs)
    dwy = np.diff(wys)
    wr_dx_abs = float((np.abs(dwx) / s).sum())
    wr_dy_min = float(((wys - wy) / s).min())
    wr_dy_rise = float((wys[0] - wys.min()) / s)
    wr_peak_v = float((np.hypot(dwx, dwy) / s).max())
    arm_dy_rise = float((sy - wys.min()) / s)

    return arm_ext, ea, el_y, el_lat, wr_y, ew, arm_dy_rise, \
        wr_dx_abs, wr_dy_min, wr_dy_rise, wr_peak_v


def opp_motion(opp_hist):
    if len(opp_hist) < 2:
        return 0.0
    sub = list(opp_hist)[-(PUNCH_WIN + 1):]
    s = sub[-1][6]
    wxs = np.array([h[0] for h in sub])
    wys = np.array([h[1] for h in sub])
    dwx = np.diff(wxs); dwy = np.diff(wys)
    return float((np.hypot(dwx, dwy) / s).sum())


def classify_punch(LR, hist_self, hist_opp):
    """학습 모델 우선 + 명확 신호 override + 룰 fallback."""
    feats = features_from_hist(hist_self)
    if feats is None:
        return None
    arm_ext, ea, el_y, el_lat, wr_y, ew, arm_dy_rise, \
        wr_dx_abs, wr_dy_min, wr_dy_rise, wr_peak_v = feats
    opp_v = opp_motion(hist_opp)
    full = list(feats) + [opp_v]

    # ── 1) 강한 신호 override ────────────────────────────
    # 어퍼: 손목이 크게 위로 + 팔꿈치 강하게 굽음
    if wr_dy_rise > 0.22 and ea < 125 and arm_dy_rise > 0.30:
        return 'uppercut'
    # 잽/크로스: 팔이 명확히 펴짐 + 팔꿈치-손목 거리 큼 + 위로 안 올라감
    if arm_ext > 1.10 and ea > 150 and ew > 0.45 and wr_dy_rise < 0.15:
        side = LR_to_label(LR)
        return 'jab' if side == 'lead' else 'cross'
    # 훅 명확: 팔꿈치 강하게 굽음 + 횡이동 크고 + 위로 안 올라감
    if ea < 110 and wr_dx_abs > 0.35 and ew < 0.35 and wr_dy_rise < 0.18:
        return 'hook'

    # ── 2) 학습 모델 ────────────────────────────────────
    if PUNCH_MODEL is not None:
        try:
            pred = str(PUNCH_MODEL.predict([full])[0])
            # 훅으로 예측됐는데 실제로는 팔이 펴졌으면 → 잽/크로스 보정
            if pred == 'hook' and arm_ext > 1.00 and ea > 140:
                side = LR_to_label(LR)
                return 'jab' if side == 'lead' else 'cross'
            # 어퍼로 예측됐는데 위로 안 올라갔으면 → 모호, 보정
            if pred == 'uppercut' and wr_dy_rise < 0.10:
                if arm_ext > 1.00 and ea > 140:
                    side = LR_to_label(LR)
                    return 'jab' if side == 'lead' else 'cross'
                if wr_dx_abs > 0.25 and ea < 130:
                    return 'hook'
            return pred
        except Exception:
            pass

    # ── 3) 룰 fallback ──────────────────────────────────
    if wr_dy_rise > 0.18 and ea < 130:
        return 'uppercut'
    if wr_dx_abs > 0.30 and ea < 130 and ew < 0.40:
        return 'hook'
    side = LR_to_label(LR)
    return 'jab' if side == 'lead' else 'cross'


# ── 펀치 감지 ─────────────────────────────────────────────
def update_history(kp, sc, side):
    if side == 'L':
        wi, ei, si = KP_L_WR, KP_L_EL, KP_L_SH
        hist = hist_L
    else:
        wi, ei, si = KP_R_WR, KP_R_EL, KP_R_SH
        hist = hist_R
    if sc[wi] < VIS_MIN or sc[si] < VIS_MIN or sc[ei] < VIS_MIN:
        return
    s = sw_px(kp)
    hist.append((kp[wi][0], kp[wi][1], kp[ei][0], kp[ei][1],
                 kp[si][0], kp[si][1], s))


def latest_speed(hist):
    if len(hist) < 2:
        return 0.0
    a, b = hist[-2], hist[-1]
    s = b[6]
    return math.hypot(b[0] - a[0], b[1] - a[1]) / s


def detect_and_classify(now):
    """양쪽 손 펀치 감지 → 분류 → counts/flash 갱신."""
    global flash
    fired = []

    for LR, hist, opp_hist in [
        ('L', hist_L, hist_R),
        ('R', hist_R, hist_L),
    ]:
        v = latest_speed(hist)
        st = det[LR]

        # 가라앉으면 재무장
        if v < VEL_START * 0.6:
            st['settled'] = True

        cd = COOLDOWN_SAME
        if now - last_fire_t[LR] < cd:
            continue

        if v > VEL_START and st['settled']:
            if not st['active']:
                st['active'] = True
                st['start'] = now
                st['peak'] = v
            else:
                st['peak'] = max(st['peak'], v)

        # 트리거: 속도 떨어짐 + 충분한 피크 + 시간 경과
        if st['active']:
            falling = v < st['peak'] * 0.55
            timeout = (now - st['start']) > 0.18
            if (falling or timeout) and st['peak'] > VEL_PEAK_MIN:
                pt = classify_punch(LR, hist, opp_hist)
                if pt:
                    fired.append((LR, pt, st['peak']))
                st['active'] = False
                st['peak'] = 0.0
                st['settled'] = False
                last_fire_t[LR] = now

    # 한 프레임에 양쪽 동시 발동하면 더 강한 쪽만
    if len(fired) >= 2:
        fired.sort(key=lambda x: -x[2])
        fired = fired[:1]

    for LR, pt, _ in fired:
        counts[pt] = counts.get(pt, 0) + 1
        flash = (pt, LR, now + 0.4)
        global last_punch
        last_punch = (pt, LR, now)


# ── 자세 점수 ──────────────────────────────────────────────
def calc_posture(kp, sc):
    """0~100. 4개 항목."""
    if any(sc[i] < VIS_MIN for i in [KP_L_SH, KP_R_SH, KP_L_WR, KP_R_WR, KP_NOSE]):
        return None
    s = sw_px(kp)
    if sc[KP_L_HI] > VIS_MIN and sc[KP_R_HI] > VIS_MIN:
        sh_y = (kp[KP_L_SH][1] + kp[KP_R_SH][1]) / 2
        hi_y = (kp[KP_L_HI][1] + kp[KP_R_HI][1]) / 2
        torso = abs(hi_y - sh_y)
        s = max(s, torso)

    # 가드: 양손목이 어깨 부근(머리/턱 근처) 있는지 — wr_y - sh_y 가 -0.4 ~ +0.2 정도
    l_yd = (kp[KP_L_WR][1] - kp[KP_L_SH][1]) / s
    r_yd = (kp[KP_R_WR][1] - kp[KP_R_SH][1]) / s
    def guard_score(y):
        if -0.35 <= y <= 0.10:
            return 18
        if -0.55 <= y <= 0.30:
            return 10
        return 0
    guard = guard_score(l_yd) + guard_score(r_yd)  # 0~36

    # 머리: 코가 어깨 위쪽인지
    nose_y = (kp[KP_NOSE][1] - (kp[KP_L_SH][1] + kp[KP_R_SH][1]) / 2) / s
    if nose_y < -0.55:
        head = 22
    elif nose_y < -0.35:
        head = 16
    elif nose_y < -0.10:
        head = 8
    else:
        head = 0

    # 린포워드: 어깨 중심이 엉덩이 중심보다 약간 앞 (측면 카메라일 때)
    lean = 22
    lean_msg = '좋아'
    if sc[KP_L_HI] > VIS_MIN and sc[KP_R_HI] > VIS_MIN:
        sh_cx = (kp[KP_L_SH][0] + kp[KP_R_SH][0]) / 2
        hi_cx = (kp[KP_L_HI][0] + kp[KP_R_HI][0]) / 2
        d = (sh_cx - hi_cx) / s
        if abs(d) < 0.12:
            lean = 22; lean_msg = '좋아'
        elif abs(d) < 0.25:
            lean = 12; lean_msg = '약간 기울임'
        else:
            lean = 0
            lean_msg = '앞으로 너무 기울임' if d > 0 else '뒤로 기울임'

    # 팔꿈치: 몸 안쪽에 붙어있는지 (lat 작음)
    elbow = 0
    if sc[KP_L_EL] > VIS_MIN and sc[KP_R_EL] > VIS_MIN:
        l_lat = abs(kp[KP_L_EL][0] - kp[KP_L_SH][0]) / s
        r_lat = abs(kp[KP_R_EL][0] - kp[KP_R_SH][0]) / s
        elbow = (10 if l_lat < 0.45 else 4 if l_lat < 0.65 else 0)
        elbow += (10 if r_lat < 0.45 else 4 if r_lat < 0.65 else 0)

    total = guard + head + lean + elbow
    grade = 'S' if total >= 85 else 'A' if total >= 70 else 'B' if total >= 50 else 'C'
    items = [
        ('가드',       guard,  36, '올려' if guard < 30 else '좋아'),
        ('머리',       head,   22, '턱 당겨' if head < 18 else '좋아'),
        ('린포워드',   lean,   22, lean_msg),
        ('팔꿈치',     elbow,  20, '안쪽으로' if elbow < 16 else '좋아'),
    ]
    return total, grade, items


# ── UI ────────────────────────────────────────────────────
WIN_W, WIN_H = 1280, 720
CAM_W = 880
SB_X = CAM_W + 12

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, pos, scale=0.6, color=INK, thick=1, font=FONT):
    cv2.putText(img, text, pos, font, scale, color, thick, cv2.LINE_AA)


def panel(img, x, y, w, h, color=PANEL, border=BORDER):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)


def bar(img, x, y, w, h, ratio, color=ACCENT, bg=(50, 50, 56)):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fw = int(w * max(0.0, min(1.0, ratio)))
    if fw > 0:
        cv2.rectangle(img, (x, y), (x + fw, y + h), color, -1)


def draw_skeleton(img, kp, sc):
    for a, b in COCO_LINES:
        if sc[a] > VIS_MIN and sc[b] > VIS_MIN:
            cv2.line(img, (int(kp[a][0]), int(kp[a][1])),
                     (int(kp[b][0]), int(kp[b][1])), SKELETON, 2, cv2.LINE_AA)
    for i in range(17):
        if sc[i] > VIS_MIN:
            cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), 4, JOINT, -1, cv2.LINE_AA)


def draw_flash(img, now):
    if not flash:
        return
    pt, LR, exp = flash
    if now > exp:
        return
    alpha = max(0.0, (exp - now) / 0.4)
    col = PUNCH_COLOR[pt]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), col, 12)
    cv2.addWeighted(overlay, alpha * 0.6, img, 1 - alpha * 0.6, 0, img)
    label = f'{pt.upper()} ({LR})'
    put(img, label, (28, 60), 1.4, col, 3)


def draw_sidebar(canvas, posture, fps, ml_ok):
    x = SB_X
    w = WIN_W - SB_X - 12

    # 헤더
    put(canvas, 'BOXING COACH', (x, 38), 1.0, INK, 2)
    put(canvas, f'{"ORTHODOX" if orthodox else "SOUTHPAW"}', (x, 62), 0.5, ACCENT, 1)
    put(canvas, f'ML {"ON" if ml_ok else "OFF"}  |  {fps:5.1f} fps',
        (x, 80), 0.45, DIM, 1)

    # 펀치 카운트
    cy = 100
    panel(canvas, x, cy, w, 130)
    put(canvas, 'PUNCHES', (x + 10, cy + 22), 0.55, DIM, 1)
    items = list(counts.items())
    for i, (pt, n) in enumerate(items):
        r = i // 2; c = i % 2
        bx = x + 10 + c * (w // 2 - 5)
        by = cy + 36 + r * 44
        col = PUNCH_COLOR[pt]
        put(canvas, pt.upper(), (bx, by + 14), 0.55, DIM, 1)
        put(canvas, f'{n:03d}', (bx, by + 38), 1.0, col, 2)

    # 자세 점수
    cy = 246
    if posture:
        total, grade, lines = posture
        panel(canvas, x, cy, w, 16 + len(lines) * 38 + 28)
        put(canvas, 'POSTURE', (x + 10, cy + 22), 0.55, DIM, 1)
        grade_col = OK if total >= 85 else ACCENT if total >= 70 else WARN
        put(canvas, f'{total:3d}/100', (x + w - 130, cy + 22), 0.7, INK, 2)
        put(canvas, f'[{grade}]', (x + w - 55, cy + 22), 0.7, grade_col, 2)
        ly = cy + 42
        for label, sc_, mx, msg in lines:
            put(canvas, label, (x + 10, ly + 12), 0.5, DIM, 1)
            put(canvas, f'{sc_}/{mx}', (x + w - 80, ly + 12), 0.5, INK, 1)
            ratio = sc_ / mx if mx else 0
            col = OK if ratio > 0.85 else ACCENT if ratio > 0.5 else WARN
            bar(canvas, x + 10, ly + 18, w - 20, 5, ratio, col)
            put(canvas, msg, (x + 10, ly + 32), 0.42,
                col if ratio < 0.85 else DIM, 1)
            ly += 38
    else:
        panel(canvas, x, cy, w, 80)
        put(canvas, 'POSTURE', (x + 10, cy + 22), 0.55, DIM, 1)
        put(canvas, '인식 대기...', (x + 10, cy + 50), 0.5, DIM, 1)

    # 단축키
    put(canvas, '[Q] quit   [R] reset   [D] stance',
        (x, WIN_H - 20), 0.45, DIM, 1)


def draw_status_overlay(cam, kp_disp, sc):
    """카메라 위 좌상단 작은 표시."""
    if kp_disp is None:
        put(cam, '전신이 카메라에 들어오게 서주세요', (24, 36), 0.6, WARN, 2)


# ── 메인 루프 ─────────────────────────────────────────────
def main():
    global orthodox

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise SystemExit('웹캠 열기 실패.')

    cv2.namedWindow('LIM coach', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('LIM coach', WIN_W, WIN_H)

    fps_t = time.time()
    fps_n = 0
    fps_v = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # 거울 모드 — 사용자에게 자연스러움
        now = time.time()

        fh, fw = frame.shape[:2]
        cam_view = cv2.resize(frame, (CAM_W, WIN_H))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kps, scs = pose_model(rgb)

        kp = None
        sc = None
        kp_disp = None
        if len(kps) > 0:
            kp = kps[0]
            sc = scs[0]
            if all(sc[i] > VIS_MIN for i in NEEDED):
                # 펀치 감지는 원본 좌표 기준
                update_history(kp, sc, 'L')
                update_history(kp, sc, 'R')
                detect_and_classify(now)

                kp_disp = kp.copy().astype(float)
                kp_disp[:, 0] *= CAM_W / fw
                kp_disp[:, 1] *= WIN_H / fh

        posture = calc_posture(kp, sc) if kp is not None and sc is not None else None

        canvas = np.full((WIN_H, WIN_W, 3), BG, dtype=np.uint8)
        canvas[:, :CAM_W] = cam_view
        if kp_disp is not None:
            draw_skeleton(canvas[:, :CAM_W], kp_disp, sc)
        else:
            draw_status_overlay(canvas[:, :CAM_W], None, None)
        draw_flash(canvas[:, :CAM_W], now)

        # 구분선
        cv2.line(canvas, (CAM_W, 0), (CAM_W, WIN_H), BORDER, 1)
        draw_sidebar(canvas, posture, fps_v, PUNCH_MODEL is not None)

        # FPS 계산
        fps_n += 1
        if now - fps_t >= 0.5:
            fps_v = fps_n / (now - fps_t)
            fps_t = now
            fps_n = 0

        cv2.imshow('LIM coach', canvas)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('r'):
            for kk in counts:
                counts[kk] = 0
        elif k == ord('d'):
            orthodox = not orthodox

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
