"""
lim_pose_coach_app.py — LIM 포즈 복싱 코치 (윈도우 분류기 버전)

펀치 감지/분류를 5-클래스 슬라이딩 윈도우 분류기 하나로 통합:
  · 매 프레임 ±7프레임 윈도우를 보고 {none,jab,cross,hook,uppercut} 분류
  · PunchEventTracker 가 연속 예측을 이벤트로 묶음
  · 학습/평가와 동일한 lim_punch_features 모듈 사용 (코드 단일화)

성능 (leave-one-video-out 정직 평가): 이벤트 Macro-F1 ≈ 0.77
  ※ 구버전 속도-피크 detector 는 0.25 수준이었음.

준비물:
  lim_window_model.pkl   ← python LIM_train_window.py 로 생성

실행:
  python lim_pose_coach_app.py
  python lim_pose_coach_app.py --source "LIM 7.mp4"

단축키:  Q/ESC 종료   R 카운트 초기화   D 스탠스 전환
"""

from __future__ import annotations

import argparse
import csv
import math
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

VIS_MIN = 0.30
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
# 모델 로드
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
# 윈도우 펀치 detector
# ══════════════════════════════════════════════════════════
class WindowPunchDetector:
    """매 프레임 윈도우 분류 → 이벤트 트래커. 학습/평가와 동일 로직."""

    def __init__(self, model, classes):
        self.model = model
        # 모델 클래스 순서 → F.CLASSES 순서 매핑
        self.idx_map = [classes.index(c) for c in F.CLASSES]
        self.buf = deque(maxlen=F.WINDOW)
        self.tracker = F.PunchEventTracker()
        self.frame_no = 0
        self.counts = Counter()
        self.last_event = None       # 마지막 발동 이벤트 dict
        self.last_event_time = 0.0
        self.live_probs = np.zeros(len(F.CLASSES))  # 현재 프레임 확률 (UI용)

    def update(self, kp, now):
        """kp: (17,>=2). 이벤트 발동 시 dict 반환."""
        self.frame_no += 1
        self.buf.append(np.asarray(kp[:, :2], dtype=float))
        if len(self.buf) < F.WINDOW:
            return None
        feat = F.window_feat(list(self.buf), F.HALF)
        probs = self.model.predict_proba([feat])[0][self.idx_map]
        self.live_probs = probs
        center_frame = self.frame_no - F.HALF
        ev = self.tracker.push(center_frame, probs)
        if ev:
            self.counts[ev['punch_type']] += 1
            self.last_event = ev
            self.last_event_time = now
        return ev


# ══════════════════════════════════════════════════════════
# 자세 점수
# ══════════════════════════════════════════════════════════
def posture_score(kp, sc, dna, orthodox):
    """0~100 점수 + 항목 리스트. 몸통높이 정규화 (뷰 불변)."""
    if any(sc[i] < VIS_MIN for i in NEEDED):
        return None
    s = F.body_scale(kp)
    sh_y = (kp[F.L_SH, 1] + kp[F.R_SH, 1]) / 2

    lead_wr = F.L_WR if orthodox else F.R_WR
    lead_sh = F.L_SH if orthodox else F.R_SH
    rear_wr = F.R_WR if orthodox else F.L_WR
    rear_sh = F.R_SH if orthodox else F.L_SH

    # 가드 — 손목이 어깨~턱 높이 (음수 = 위)
    def guard(y):
        if -0.55 <= y <= 0.05:
            return 18
        if -0.80 <= y <= 0.25:
            return 10
        return 0
    lead_y = (kp[lead_wr, 1] - kp[lead_sh, 1]) / s
    rear_y = (kp[rear_wr, 1] - kp[rear_sh, 1]) / s
    g = guard(lead_y) + guard(rear_y)          # 0~36

    # 머리 — 코가 어깨보다 위
    head_y = (kp[F.NOSE, 1] - sh_y) / s
    if head_y < -0.45:
        head = 24
    elif head_y < -0.25:
        head = 16
    elif head_y < -0.05:
        head = 8
    else:
        head = 0

    # 린포워드 — 어깨중심 vs 엉덩이중심
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

    # 팔꿈치 — 몸 안쪽
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
    """items: [(text, (x,y), size, color), ...] — 한 번의 PIL 변환으로 전부 렌더.

    프레임마다 BGR→PIL→BGR 변환을 1회만 수행 (텍스트마다 하면 FPS 급락).
    """
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


def draw_skeleton(img, kp, sc):
    for a, b in SKELETON:
        if a < 17 and b < 17 and sc[a] > VIS_MIN and sc[b] > VIS_MIN:
            cv2.line(img, (int(kp[a, 0]), int(kp[a, 1])),
                     (int(kp[b, 0]), int(kp[b, 1])), (170, 170, 180), 2,
                     cv2.LINE_AA)
    for i in range(17):
        if sc[i] > VIS_MIN:
            cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), 4,
                       (90, 70, 255), -1, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, ratio, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 52, 50), -1)
    fw = int(w * max(0.0, min(1.0, ratio)))
    if fw > 0:
        cv2.rectangle(img, (x, y), (x + fw, y + h), color, -1)


def draw_overlay(frame, detector, posture, fps, orthodox, flash_until, now):
    """cv2 도형은 즉시 그리고, 텍스트는 리스트로 모아 반환 (1회 렌더용)."""
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

    # 하단 — 마지막 펀치
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
    texts.append((f'{"ORTHODOX" if orthodox else "SOUTHPAW"}  {fps:4.1f} fps',
                  (w // 2 - 90, 20), 15, (150, 145, 140)))
    texts.append(('Q 종료   R 초기화   D 스탠스', (w // 2 - 110, h - 26), 15,
                  (150, 145, 140)))
    return texts


# ══════════════════════════════════════════════════════════
# RTMO
# ══════════════════════════════════════════════════════════
def build_pose_model():
    try:
        from rtmlib import RTMO
    except ImportError as exc:
        raise SystemExit('pip install rtmlib onnxruntime') from exc
    try:
        import onnxruntime as ort
        device = ('dml' if 'DmlExecutionProvider' in ort.get_available_providers()
                  else 'cpu')
    except Exception:
        device = 'cpu'
    url = ('https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
           'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip')
    print(f'RTMO 로드 중... (device={device})')
    return RTMO(url, backend='onnxruntime', device=device)


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='LIM 포즈 복싱 코치')
    ap.add_argument('--source', default='0', help='카메라 인덱스 또는 영상 경로')
    ap.add_argument('--stance', choices=['orthodox', 'southpaw'],
                    default='orthodox')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    args = ap.parse_args()

    model, classes, meta = load_window_model(
        os.path.join(BASE_DIR, 'lim_window_model.pkl'))
    print(f'윈도우 모델 로드: classes={classes}')
    if meta:
        print(f'  LOVO Macro-F1={meta.get("lovo_macro_f1")}  '
              f'표본={meta.get("samples")}')
    dna = load_pose_dna(os.path.join(BASE_DIR, 'LIM_DNA.csv'))

    pose_model = build_pose_model()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f'소스 열기 실패: {args.source}')

    orthodox = args.stance == 'orthodox'
    is_camera = isinstance(src, int)
    detector = WindowPunchDetector(model, classes)

    win = 'LIM Boxing Coach'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    flash_until = 0.0
    fps, fps_t, fps_n = 0.0, time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if is_camera:
            frame = cv2.flip(frame, 1)   # 거울 모드
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kps, scores = pose_model(rgb)

        status_texts = []
        if len(kps) > 0:
            kp = kps[0].astype(float)
            sc = scores[0].astype(float)
            if all(sc[i] > VIS_MIN for i in NEEDED):
                ev = detector.update(kp, now)
                if ev:
                    flash_until = now + 0.5
                draw_skeleton(frame, kp, sc)
                posture = posture_score(kp, sc, dna, orthodox)
            else:
                posture = None
                status_texts.append(('전신이 보이도록 서주세요',
                                     (frame.shape[1] // 2 - 130, 44), 20,
                                     (70, 200, 255)))
        else:
            posture = None
            status_texts.append(('사람을 찾는 중...',
                                 (frame.shape[1] // 2 - 90, 44), 20,
                                 (70, 200, 255)))

        fps_n += 1
        if now - fps_t >= 0.5:
            fps = fps_n / (now - fps_t)
            fps_t, fps_n = now, 0

        texts = draw_overlay(frame, detector, posture, fps, orthodox,
                             flash_until, now)
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
