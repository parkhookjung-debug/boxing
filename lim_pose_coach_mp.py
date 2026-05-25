"""
lim_pose_coach_mp.py — LIM 복싱 코치 (MediaPipe 경량 / 측면 전용)

기존 lim_pose_coach_app.py(RTMO)를 대체하는 신규 앱. 모바일/실시간을
목표로 포즈 백엔드를 MediaPipe BlazePose(lite)로 교체했다.

※ 측면(3/4 각도) 전용 — 분류기는 2D 좌표만 쓰므로 정면에서는 잽/크로스가
  카메라축(Z) 모션이라 화면상 거의 안 움직여 인식되지 않는다. 단 완전
  옆모습(90°)도 안 된다 — 반대쪽 손이 머리·몸에 가려 NEEDED 게이트가
  꺼진다. 카메라에 몸을 45~60° 비스듬히(3/4) 틀어 양손이 다 보이게 한다.
  학습 영상(LIM)도 이 3/4 각도다. 모델은 측면 영상으로만 학습한다.

파이프라인:
  [카메라] → MediaPipe Pose(경량, 33랜드마크)
          → 33→COCO17 매핑
          → 1€ 필터 (지터링 제거, 펀치 속도는 보존)
          → 윈도우 분류기 (lim_window_model.pkl)
          → PunchEventTracker → 펀치 판정 + 자세 점수

학습 CSV·모델·런타임이 모두 같은 MediaPipe 좌표 + 같은 1€ 필터를 거치므로
(lim_punch_features.load_kp_csv 가 학습 측 필터 담당) 분포가 일치한다.

준비물:
  pose_landmarker_lite.task   ← MediaPipe 모델
  lim_window_model.pkl        ← python LIM_train_window.py 로 생성

실행:
  python lim_pose_coach_mp.py
  python lim_pose_coach_mp.py --source "LIM 7.mp4"
  python lim_pose_coach_mp.py --model full

단축키:  Q/ESC 종료   R 카운트 초기화   D 스탠스 전환   F 필터 ON/OFF
"""

from __future__ import annotations

import argparse
import csv
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

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import lim_punch_features as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIS_MIN = 0.30                       # MediaPipe visibility 임계
NEEDED = [F.NOSE, F.L_SH, F.R_SH, F.L_EL, F.R_EL, F.L_WR, F.R_WR]

# COCO 13~16 = 좌무릎,우무릎,좌발목,우발목
L_KN, R_KN, L_AN, R_AN = 13, 14, 15, 16
SKELETON = [
    (F.L_SH, F.R_SH), (F.L_SH, F.L_EL), (F.L_EL, F.L_WR),
    (F.R_SH, F.R_EL), (F.R_EL, F.R_WR),
    (F.L_SH, F.L_HI), (F.R_SH, F.R_HI), (F.L_HI, F.R_HI),
    (F.L_HI, L_KN), (L_KN, L_AN), (F.R_HI, R_KN), (R_KN, R_AN),
    (F.NOSE, F.L_SH), (F.NOSE, F.R_SH),
]

# COCO-17 인덱스 → MediaPipe BlazePose 33 랜드마크 인덱스
COCO_FROM_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

PUNCH_COLORS = {
    'jab': (0, 220, 255),
    'cross': (255, 150, 40),
    'hook': (210, 70, 255),
    'uppercut': (80, 240, 90),
}

POSE_DEFAULTS = {
    'guard_l_ydiff': -0.15, 'guard_r_ydiff': -0.10,
    'head_y_ratio': -0.85, 'lean_forward': 0.05,
}


# ══════════════════════════════════════════════════════════
# 폰트 (한글)
# ══════════════════════════════════════════════════════════
def load_font(size):
    if ImageFont is None:
        return None
    for path in ('C:/Windows/Fonts/malgun.ttf', 'C:/Windows/Fonts/gulim.ttc'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONTS = {s: load_font(s) for s in (15, 17, 20, 24, 30, 44)}


# ══════════════════════════════════════════════════════════
# 모델 / 설정 로드
# ══════════════════════════════════════════════════════════
def load_window_model(path):
    if not os.path.exists(path):
        raise SystemExit(
            f'{os.path.basename(path)} 없음. 먼저 "python LIM_train_window.py" 실행.')
    with open(path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], list(bundle['classes']), bundle.get('meta', {})


def load_pose_dna(path):
    dna = dict(POSE_DEFAULTS)
    if not os.path.exists(path):
        return dna
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            for key in dna:
                try:
                    dna[key] = float(row[key])
                except (KeyError, TypeError, ValueError):
                    pass
            break
    return dna


# ══════════════════════════════════════════════════════════
# MediaPipe 포즈
# ══════════════════════════════════════════════════════════
def build_landmarker(model_name):
    """MediaPipe PoseLandmarker (VIDEO 모드, 1인)."""
    path = os.path.join(BASE_DIR, f'pose_landmarker_{model_name}.task')
    if not os.path.exists(path):
        raise SystemExit(f'모델 파일 없음: {path}')
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_pose(landmarker, rgb, ts_ms):
    """rgb 프레임 → (kp (17,2) 정규화좌표, sc (17,) visibility) 또는 (None,None)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, ts_ms)
    if not result.pose_landmarks:
        return None, None
    lm = result.pose_landmarks[0]
    kp = np.empty((17, 2), dtype=float)
    sc = np.empty(17, dtype=float)
    for i, m in enumerate(COCO_FROM_MP):
        p = lm[m]
        kp[i, 0], kp[i, 1] = p.x, p.y
        sc[i] = p.visibility
    return kp, sc


# ══════════════════════════════════════════════════════════
# 윈도우 펀치 detector
# ══════════════════════════════════════════════════════════
class WindowPunchDetector:
    """매 프레임 윈도우 분류 → 이벤트 트래커. 학습/평가와 동일 로직.

    입력 kp 는 이미 1€ 필터를 거친 정규화 좌표 (메인 루프에서 평활).

    모션 게이트: 이벤트가 나와도 그 구간 손목 움직임(어깨 기준 상대속도)이
    gate_thr 미만이면 기각 — 단일 복서 학습 모델이 신규 사용자의 정적
    가드를 펀치(특히 어퍼)로 오인하는 것을 막는다."""

    def __init__(self, model, classes, gate_thr=F.MOTION_GATE_THR):
        self.model = model
        self.idx_map = [classes.index(c) for c in F.CLASSES]
        self.buf = deque(maxlen=F.WINDOW)
        self.tracker = F.PunchEventTracker()
        self.frame_no = 0
        self.counts = Counter()
        self.last_event = None
        self.last_event_time = 0.0
        self.live_probs = np.zeros(len(F.CLASSES))
        self.gate_thr = gate_thr
        self.gate_speed = 0.0          # 현재 윈도우 손목 모션 (UI 표시용)
        self.gated = False             # 현재 프레임이 게이트에 걸렸는가

    def update(self, kp, now):
        """kp: (17,>=2) 필터링된 좌표. 이벤트 발동 시 dict 반환."""
        self.frame_no += 1
        self.buf.append(np.asarray(kp[:, :2], dtype=float))
        if len(self.buf) < F.WINDOW:
            return None
        win = list(self.buf)
        feat = F.window_feat(win, F.HALF)
        probs = self.model.predict_proba([feat])[0][self.idx_map]
        self.live_probs = probs
        self.gate_speed = F.wrist_gate_speed(win)
        self.gated = self.gate_speed < self.gate_thr

        center_frame = self.frame_no - F.HALF
        ev = self.tracker.push(center_frame, probs)
        if ev:
            if self.gated:
                return None            # 모션 게이트 — 정적 자세 → 펀치 기각
            self.counts[ev['punch_type']] += 1
            self.last_event = ev
            self.last_event_time = now
        return ev


# ══════════════════════════════════════════════════════════
# 자세 점수 — 몸통높이 정규화 (뷰 불변)
# ══════════════════════════════════════════════════════════
def posture_score(kp, sc, dna, orthodox):
    if any(sc[i] < VIS_MIN for i in NEEDED):
        return None
    s = F.body_scale(kp)
    sh_y = (kp[F.L_SH, 1] + kp[F.R_SH, 1]) / 2

    lead_wr = F.L_WR if orthodox else F.R_WR
    lead_sh = F.L_SH if orthodox else F.R_SH
    rear_wr = F.R_WR if orthodox else F.L_WR
    rear_sh = F.R_SH if orthodox else F.L_SH

    def guard(y):
        if -0.55 <= y <= 0.05:
            return 18
        if -0.80 <= y <= 0.25:
            return 10
        return 0
    lead_y = (kp[lead_wr, 1] - kp[lead_sh, 1]) / s
    rear_y = (kp[rear_wr, 1] - kp[rear_sh, 1]) / s
    g = guard(lead_y) + guard(rear_y)

    head_y = (kp[F.NOSE, 1] - sh_y) / s
    if head_y < -0.45:
        head = 24
    elif head_y < -0.25:
        head = 16
    elif head_y < -0.05:
        head = 8
    else:
        head = 0

    lean, lean_msg = 20, '좋음'
    if sc[F.L_HI] > VIS_MIN and sc[F.R_HI] > VIS_MIN:
        sh_cx = (kp[F.L_SH, 0] + kp[F.R_SH, 0]) / 2
        hi_cx = (kp[F.L_HI, 0] + kp[F.R_HI, 0]) / 2
        d = (sh_cx - hi_cx) / s
        if abs(d) < 0.18:
            lean, lean_msg = 20, '좋음'
        elif abs(d) < 0.35:
            lean, lean_msg = 12, '약간 기울임'
        else:
            lean, lean_msg = 0, ('너무 숙임' if d > 0 else '뒤로 젖힘')

    elbow = 0
    if sc[F.L_EL] > VIS_MIN and sc[F.R_EL] > VIS_MIN:
        l_lat = abs(kp[F.L_EL, 0] - kp[F.L_SH, 0]) / s
        r_lat = abs(kp[F.R_EL, 0] - kp[F.R_SH, 0]) / s
        elbow = (10 if l_lat < 0.55 else 4 if l_lat < 0.80 else 0)
        elbow += (10 if r_lat < 0.55 else 4 if r_lat < 0.80 else 0)

    total = g + head + lean + elbow
    grade = 'S' if total >= 88 else 'A' if total >= 72 else 'B' if total >= 52 else 'C'
    items = [
        ('가드', g, 36, '가드 올려' if g < 30 else '좋음'),
        ('머리', head, 24, '턱 당겨' if head < 18 else '좋음'),
        ('상체', lean, 20, lean_msg),
        ('팔꿈치', elbow, 20, '안쪽으로' if elbow < 16 else '좋음'),
    ]
    return total, grade, items


# ══════════════════════════════════════════════════════════
# 그리기
# ══════════════════════════════════════════════════════════
def render_texts(img, items):
    """프레임마다 BGR→PIL→BGR 변환을 1회만 — 텍스트마다 하면 FPS 급락."""
    if not items:
        return
    if Image is None:
        for text, xy, size, color in items:
            cv2.putText(img, text, (int(xy[0]), int(xy[1] + size)),
                        cv2.FONT_HERSHEY_SIMPLEX, size / 28, color, 1,
                        cv2.LINE_AA)
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, xy, size, color in items:
        font = FONTS.get(size) or FONTS[17]
        rgb = (color[2], color[1], color[0])
        d.text((xy[0] + 2, xy[1] + 2), text, font=font, fill=(0, 0, 0))
        d.text(xy, text, font=font, fill=rgb)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_panel(img, x, y, w, h, alpha=0.55):
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), (24, 22, 20), -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 66, 62), 1)


def draw_skeleton(img, kp_px, sc):
    """kp_px: 픽셀 좌표 (17,2)."""
    for a, b in SKELETON:
        if a < 17 and b < 17 and sc[a] > VIS_MIN and sc[b] > VIS_MIN:
            cv2.line(img, (int(kp_px[a, 0]), int(kp_px[a, 1])),
                     (int(kp_px[b, 0]), int(kp_px[b, 1])), (170, 170, 180), 2,
                     cv2.LINE_AA)
    for i in range(17):
        if sc[i] > VIS_MIN:
            cv2.circle(img, (int(kp_px[i, 0]), int(kp_px[i, 1])), 4,
                       (90, 70, 255), -1, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, ratio, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 52, 50), -1)
    fw = int(w * max(0.0, min(1.0, ratio)))
    if fw > 0:
        cv2.rectangle(img, (x, y), (x + fw, y + h), color, -1)


def draw_overlay(frame, detector, posture, fps, orthodox, flash_until, now,
                 use_filter):
    """cv2 도형은 즉시, 텍스트는 리스트로 모아 반환 (1회 렌더용)."""
    h, w = frame.shape[:2]
    texts = []

    # 좌상단 — 펀치 카운트
    draw_panel(frame, 16, 16, 200, 150)
    texts.append(('PUNCH', (28, 24), 17, (170, 165, 160)))
    for i, pt in enumerate(['jab', 'cross', 'hook', 'uppercut']):
        y = 48 + i * 28
        texts.append((pt.upper(), (28, y), 17, (190, 185, 180)))
        texts.append((f'{detector.counts.get(pt, 0):3d}', (150, y), 20,
                      PUNCH_COLORS[pt]))

    # 우상단 — 자세
    px = w - 236
    if posture:
        total, grade, items = posture
        ph = 56 + len(items) * 34
        draw_panel(frame, px, 16, 220, ph)
        gcol = (90, 220, 110) if total >= 72 else (60, 170, 255)
        texts.append(('POSTURE', (px + 12, 24), 17, (170, 165, 160)))
        texts.append((f'{total}', (px + 120, 22), 30, (240, 240, 240)))
        texts.append((f'[{grade}]', (px + 170, 28), 20, gcol))
        for i, (label, sval, mx, msg) in enumerate(items):
            y = 56 + i * 34
            ratio = sval / mx if mx else 0
            bcol = ((90, 220, 110) if ratio > 0.85 else
                    (60, 170, 255) if ratio > 0.5 else (70, 90, 240))
            texts.append((label, (px + 12, y), 15, (190, 185, 180)))
            texts.append((msg, (px + 96, y), 15,
                          bcol if ratio < 0.85 else (140, 135, 130)))
            draw_bar(frame, px + 12, y + 20, 196, 4, ratio, bcol)

    # 테두리 플래시 + 마지막 펀치
    ev = detector.last_event
    if ev and now < flash_until:
        col = PUNCH_COLORS.get(ev['punch_type'], (255, 255, 255))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 10)
        texts.append((ev['punch_type'].upper(), (28, h - 96), 44, col))
        texts.append((f"{ev['confidence'] * 100:.0f}%", (250, h - 78), 24,
                      (235, 235, 235)))

    # 실시간 펀치 확률 막대
    probs = detector.live_probs
    bx, by = 16, h - 40
    texts.append(('LIVE', (bx, by - 22), 15, (150, 145, 140)))
    for i, pt in enumerate(['jab', 'cross', 'hook', 'uppercut']):
        ci = F.CLASSES.index(pt)
        x = bx + i * 116
        texts.append((pt[:4], (x, by - 4), 15, (170, 165, 160)))
        draw_bar(frame, x + 44, by + 2, 64, 8, float(probs[ci]),
                 PUNCH_COLORS[pt])

    # 상단중앙 — 상태
    fcol = (90, 220, 110) if use_filter else (90, 90, 230)
    texts.append((f'{"ORTHODOX" if orthodox else "SOUTHPAW"}   {fps:4.1f} fps',
                  (w // 2 - 110, 20), 15, (150, 145, 140)))
    texts.append((f'FILTER {"ON" if use_filter else "OFF"}',
                  (w // 2 + 70, 20), 15, fcol))

    # 모션 게이트 readout — 가드(정적)면 회색, 움직이면 초록
    gs, gt = detector.gate_speed, detector.gate_thr
    gcol = (90, 220, 110) if gs >= gt else (125, 125, 135)
    texts.append((f'MOTION {gs:.3f}   GATE {gt:.3f}',
                  (w // 2 - 110, 40), 15, gcol))
    if detector.gated:
        texts.append(('정적 — 펀치 억제', (w // 2 + 70, 40), 15, (135, 135, 210)))

    texts.append(('Q 종료  R 초기화  D 스탠스  F 필터  [ ] 게이트',
                  (w // 2 - 165, h - 26), 15, (150, 145, 140)))
    return texts


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='LIM 복싱 코치 (MediaPipe)')
    ap.add_argument('--source', default='0', help='카메라 인덱스 또는 영상 경로')
    ap.add_argument('--stance', choices=['orthodox', 'southpaw'],
                    default='orthodox')
    ap.add_argument('--model', choices=['lite', 'full'], default='lite',
                    help='MediaPipe 모델 (기본: lite — 모바일/실시간용)')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--gate', type=float, default=F.MOTION_GATE_THR,
                    help='모션 게이트 임계값 (이 미만 손목속도는 펀치 아님)')
    args = ap.parse_args()

    model, classes, meta = load_window_model(
        os.path.join(BASE_DIR, 'lim_window_model.pkl'))
    print(f'윈도우 모델 로드: classes={classes}')
    if meta:
        print(f'  LOVO Macro-F1={meta.get("lovo_macro_f1")}  '
              f'표본={meta.get("samples")}')
    dna = load_pose_dna(os.path.join(BASE_DIR, 'LIM_DNA.csv'))

    landmarker = build_landmarker(args.model)
    print(f'MediaPipe 로드: pose_landmarker_{args.model}.task')
    print(f'1€ 필터: min_cutoff={F.SMOOTH_MIN_CUTOFF} beta={F.SMOOTH_BETA}')
    print(f'모션 게이트: {args.gate}  ([ ] 키로 실시간 조절)')
    print('※ 측면 전용 — 카메라에 몸을 45~60° 비스듬히 (완전 옆모습 X: '
          '뒷손이 가려짐).')

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f'소스 열기 실패: {args.source}')

    is_camera = isinstance(src, int)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if is_camera:
        fps_src = 30.0                    # 카메라는 명목값

    orthodox = args.stance == 'orthodox'
    detector = WindowPunchDetector(model, classes, gate_thr=args.gate)
    kp_filter = F.KeypointFilter()        # 1€ 필터 (17키포인트 x/y)
    use_filter = True

    win = 'LIM Boxing Coach - 측면 전용 (MediaPipe)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    flash_until = 0.0
    fps, fps_t, fps_n = 0.0, time.time(), 0
    t_start = time.time()
    prev_now = t_start
    frame_idx = 0
    last_ts = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        frame_idx += 1

        # MediaPipe 타임스탬프 (ms, 단조 증가 필수)
        if is_camera:
            ts_ms = int((now - t_start) * 1000)
        else:
            ts_ms = int(frame_idx * 1000.0 / fps_src)
        if ts_ms <= last_ts:
            ts_ms = last_ts + 1
        last_ts = ts_ms

        # 1€ 필터 dt — 영상 타임라인 기준 (카메라=실시간, 파일=프레임간격)
        if is_camera:
            dt = min(0.2, max(0.005, now - prev_now))
        else:
            dt = 1.0 / fps_src
        prev_now = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kp_n, sc = detect_pose(landmarker, rgb, ts_ms)

        status_texts = []
        posture = None
        if kp_n is not None:
            # 검출되면 항상 필터 통과 → 필터 상태 연속 유지
            kp_use = kp_filter.apply(kp_n, dt) if use_filter else kp_n
            h, w = frame.shape[:2]
            draw_skeleton(frame, kp_use * np.array([w, h]), sc)   # 항상 표시
            if all(sc[i] > VIS_MIN for i in NEEDED):
                ev = detector.update(kp_use, now)
                if ev:
                    flash_until = now + 0.5
                posture = posture_score(kp_use, sc, dna, orthodox)
            else:
                # 어떤 부위가 안 보이는지 구체적으로 안내. 팔/손만 가려졌으면
                # 완전 옆모습이라 반대쪽 손이 몸에 가린 것 → 3/4 각도로 유도.
                missing = [i for i in NEEDED if sc[i] <= VIS_MIN]
                arm = {F.L_EL, F.R_EL, F.L_WR, F.R_WR}
                if missing and all(i in arm for i in missing):
                    msg, mx = '뒷손이 가려졌어요 — 살짝 비스듬히 서세요', 200
                else:
                    msg, mx = '전신이 보이도록 서주세요', 130
                status_texts.append((msg, (frame.shape[1] // 2 - mx, 44), 20,
                                     (70, 200, 255)))
        else:
            status_texts.append(('사람을 찾는 중...',
                                 (frame.shape[1] // 2 - 90, 44), 20,
                                 (70, 200, 255)))

        # 거울 표시 — 포즈 검출·분류·자세점수는 모두 원본(비반전) 좌표로
        # 끝냈다. 학습 CSV(extract_lim_mp.py)가 비반전 추출이라, 여기서
        # 미리 뒤집으면 손잡이가 반전돼 분류기가 OOD 입력을 받는다(어퍼
        # 오인의 한 원인). 스켈레톤까지 그린 뒤 영상을 좌우반전하면 골격이
        # 거울상과 함께 뒤집혀 정합하고, 오버레이 텍스트/패널은 반전 후에
        # 그리므로 글자가 뒤집히지 않는다.
        if is_camera:
            frame = cv2.flip(frame, 1)

        fps_n += 1
        if now - fps_t >= 0.5:
            fps = fps_n / (now - fps_t)
            fps_t, fps_n = now, 0

        texts = draw_overlay(frame, detector, posture, fps, orthodox,
                             flash_until, now, use_filter)
        render_texts(frame, texts + status_texts)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            detector.counts.clear()
            detector.last_event = None
        if key == ord('d'):
            orthodox = not orthodox
        if key == ord('f'):
            use_filter = not use_filter
            kp_filter = F.KeypointFilter()   # 토글 시 필터 상태 리셋
        if key == ord('['):                  # 게이트 ↓ (펀치 더 잘 잡힘)
            detector.gate_thr = round(max(0.0, detector.gate_thr - 0.005), 3)
        if key == ord(']'):                  # 게이트 ↑ (오탐 더 억제)
            detector.gate_thr = round(min(0.30, detector.gate_thr + 0.005), 3)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
