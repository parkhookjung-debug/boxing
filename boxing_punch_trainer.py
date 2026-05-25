"""
boxing_punch_trainer.py — 정면 복싱 펀치 트레이너 (개인 보정 / MediaPipe / CPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
카메라를 정면으로 보고 서서, 시작 전에 펀치를 직접 시연해 "보정"한 뒤
실시간으로 6종 펀치(잽·스트레이트·좌우 훅·좌우 어퍼)를 인식한다.

설계 의도
  ● 포즈 백엔드 = MediaPipe BlazePose — CPU/모바일에서 30fps. 무거운
    학습 모델(RTMO/ONNX)을 쓰지 않는다.
  ● "모델" = 개인 보정 k-NN — 사용자가 직접 친 5회 샘플로 클래스별
    템플릿을 만들고 최근접 분류한다. 학습 단계가 없고(즉시 사용 가능)
    사용자/카메라/체형에 자동 적응한다. 모바일 이식도 그대로 가능.
  ● 정면 뷰 대응 — 정면에서 잽/스트레이트는 카메라축(Z) 모션이라
    2D 좌표만으로는 거의 안 움직인다. 그래서 MediaPipe 의 z(깊이)와
    팔꿈치 펴짐각을 피처로 함께 써서 "앞으로 뻗는" 펀치를 잡는다.
  ● 좌우손 고정 — 보정 때 잡은 좌/우 손 정체성을 세션 내내 유지하며,
    화면에 항상 '왼손/오른손' 라벨을 표시한다. 뒤바뀌면 S 로 교정.

펀치 판별 규칙 (보정으로 개인화되지만 직관은 다음과 같다)
  잽/스트레이트 : 팔이 앞으로(z↓) 뻗고 팔꿈치가 크게 펴진다.
  훅            : 손목이 가로로 크게 이동, 얼굴 옆을 스친다, 팔꿈치 덜 펴짐.
  어퍼컷        : 손목이 아래→위로 이동(세로 우세), 팔꿈치는 "조금만" 펴짐.
                  → 정적인 가드와 달리 위로 올라가는 '움직임'이 있어야 한다.

진행 단계
  1 STANCE  전신·정면 확인
  2 HANDS   좌/우 손 인식·고정 (틀리면 S 로 교정)
  3 GUARD   가드 자세 등록
  4 CALIB   잽5 · 스트레이트5 · 훅(좌5·우5) · 어퍼(좌5·우5)
  5 PLAY    실시간 인식

준비물:  pose_landmarker_lite.task  또는  pose_landmarker_full.task
실행:    python boxing_punch_trainer.py
         python boxing_punch_trainer.py --reset       (저장 보정 무시)
         python boxing_punch_trainer.py --source 1    (다른 카메라)

단축키:  Q 종료   C 처음부터 재보정   R 현재 단계 다시   S 좌우손 교정
         L (시작 화면) 저장된 보정 불러와 바로 PLAY
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    raise SystemExit("MediaPipe 가 필요합니다:  pip install mediapipe")

try:
    import lim_punch_features as F          # OneEuroFilter 재사용
    _ONE_EURO = F.OneEuroFilter
except Exception:                          # 모듈이 없어도 단독 동작
    _ONE_EURO = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(BASE_DIR, "punch_calib.json")

# ══════════════════════════════════════════════════════════════════
# COCO-17 인덱스 / MediaPipe 매핑
# ══════════════════════════════════════════════════════════════════
NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HI, R_HI = 11, 12
L_KN, R_KN, L_AN, R_AN = 13, 14, 15, 16

# COCO-17 인덱스 → MediaPipe BlazePose 33 랜드마크 인덱스
COCO_FROM_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# 좌우 라벨만 교환(좌표는 유지) — 손 정체성 교정용
LR_SWAP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

SKELETON = [
    (L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR),
    (R_SH, R_EL), (R_EL, R_WR),
    (L_SH, L_HI), (R_SH, R_HI), (L_HI, R_HI),
    (L_HI, L_KN), (L_KN, L_AN), (R_HI, R_KN), (R_KN, R_AN),
    (NOSE, L_SH), (NOSE, R_SH),
]

VIS_MIN = 0.30
BODY_KP = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HI, R_HI]
ARM_KP = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR]

Z_W = 1.5          # reach/속도에서 깊이(z) 가중치

# ══════════════════════════════════════════════════════════════════
# 색상 (BGR)
# ══════════════════════════════════════════════════════════════════
C_BG = (22, 18, 16)
C_PANEL = (36, 30, 27)
C_LINE = (74, 68, 62)
C_TX = (236, 236, 236)
C_DIM = (150, 148, 150)
C_OK = (90, 220, 110)
C_INFO = (255, 184, 72)
C_WARN = (78, 88, 240)
C_LEFT = (60, 210, 255)
C_RIGHT = (255, 150, 50)
C_SKEL = (165, 162, 170)

PUNCH_COL = {
    'jab': (60, 210, 255),
    'straight': (255, 150, 50),
    'hook': (215, 90, 255),
    'uppercut': (95, 235, 100),
}

# ══════════════════════════════════════════════════════════════════
# 보정 시퀀스
# ══════════════════════════════════════════════════════════════════
CALIB_SEQUENCE = [
    ('jab', 'left'),
    ('straight', 'right'),
    ('hook', 'left'),
    ('hook', 'right'),
    ('uppercut', 'left'),
    ('uppercut', 'right'),
]
SAMPLES_PER = 5

PUNCH_KR = {
    ('left', 'jab'): '잽',
    ('right', 'straight'): '스트레이트',
    ('left', 'hook'): '왼훅',
    ('right', 'hook'): '오른훅',
    ('left', 'uppercut'): '왼어퍼',
    ('right', 'uppercut'): '오른어퍼',
}
STAGE_KR = {
    ('jab', 'left'): '잽  —  왼손',
    ('straight', 'right'): '스트레이트  —  오른손',
    ('hook', 'left'): '훅  —  왼손',
    ('hook', 'right'): '훅  —  오른손',
    ('uppercut', 'left'): '어퍼컷  —  왼손',
    ('uppercut', 'right'): '어퍼컷  —  오른손',
}
STAGE_HINT = {
    'jab': '가드에서 왼손을 앞으로 곧게 뻗었다 가드로',
    'straight': '가드에서 오른손을 앞으로 곧게 뻗었다 가드로',
    'hook': '얼굴 옆을 스치듯 손목을 가로로 휘둘러 치기',
    'uppercut': '아래에서 위로 올려치기 — 팔꿈치는 살짝만 펴기',
}

# 피처: [dx, dy, dz, elbow, straighten, armext, path_h, path_v, up, facedx, speed]
FEATURE_DIM = 11
FEATURE_W = np.array([1.0, 1.1, 1.1, 0.7, 1.3, 0.8, 1.3, 1.3, 1.4, 1.1, 0.7])

# 펀치 이벤트 트래커 파라미터
START_P = 0.16          # 펀치 시작으로 보는 progress 임계
RETRACT_RATIO = 0.60    # 정점 대비 이만큼 줄면 회수 → 이벤트 마감
MAX_DUR = 1.4           # 펀치 최대 지속 (초)
STOP_SPEED = 0.018
STOP_FRAMES = 4
COOLDOWN = 0.30         # 펀치 간 최소 간격 (콤보 허용)


# ══════════════════════════════════════════════════════════════════
# 폰트 / 텍스트
# ══════════════════════════════════════════════════════════════════
def load_font(size):
    if ImageFont is None:
        return None
    for path in ('C:/Windows/Fonts/malgun.ttf', 'C:/Windows/Fonts/gulim.ttc',
                 'C:/Windows/Fonts/segoeui.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONTS = {s: load_font(s) for s in (15, 18, 22, 28, 40, 64)}


def draw_texts(img, items):
    """items: [(text, (x, y), size, color_bgr), ...] — 프레임당 1회 변환."""
    if not items:
        return
    if Image is None:
        for text, xy, size, color in items:
            cv2.putText(img, text, (int(xy[0]), int(xy[1] + size)),
                        cv2.FONT_HERSHEY_SIMPLEX, size / 30, color, 1,
                        cv2.LINE_AA)
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, xy, size, color in items:
        font = FONTS.get(size) or FONTS[18]
        rgb = (color[2], color[1], color[0])
        d.text((xy[0] + 2, xy[1] + 2), text, font=font, fill=(0, 0, 0))
        d.text(xy, text, font=font, fill=rgb)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def text_w(text, size):
    font = FONTS.get(size)
    if font is None or Image is None:
        return int(len(text) * size * 0.6)
    box = font.getbbox(text)
    return box[2] - box[0]


# ══════════════════════════════════════════════════════════════════
# 기하 헬퍼
# ══════════════════════════════════════════════════════════════════
def body_scale(kp):
    """뷰에 둔감한 정규화 스케일 — 어깨폭(정면에서 안정)을 우선."""
    sw = float(np.linalg.norm(kp[R_SH, :2] - kp[L_SH, :2]))
    sh_c = (kp[L_SH, :2] + kp[R_SH, :2]) * 0.5
    hi_c = (kp[L_HI, :2] + kp[R_HI, :2]) * 0.5
    torso = float(np.linalg.norm(sh_c - hi_c))
    return max(sw, 0.6 * torso, 1e-6)


def angle3(a, b, c):
    """b 를 꼭짓점으로 하는 3D 각도(도). a,b,c 는 길이 3 벡터."""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cos_v = max(-1.0, min(1.0, float(np.dot(ba, bc)) / denom))
    return math.degrees(math.acos(cos_v))


def wdist(a, b):
    """가중 유클리드 거리."""
    d = a - b
    return float(math.sqrt(float(np.sum(FEATURE_W * d * d))))


# ══════════════════════════════════════════════════════════════════
# MediaPipe 포즈
# ══════════════════════════════════════════════════════════════════
def build_landmarker(prefer):
    order, seen = [], set()
    for nm in (prefer, 'lite', 'full'):
        if nm and nm not in seen:
            seen.add(nm)
            order.append(nm)
    for nm in order:
        path = os.path.join(BASE_DIR, f'pose_landmarker_{nm}.task')
        if os.path.exists(path):
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return vision.PoseLandmarker.create_from_options(options), nm
    raise SystemExit(
        'pose_landmarker_lite.task 또는 pose_landmarker_full.task 가 없습니다.\n'
        'MediaPipe 포즈 모델 파일을 boxing 폴더에 두세요.')


def detect_pose(landmarker, rgb, ts_ms):
    """rgb → (kp (17,3) [x,y,z 정규화], sc (17,) visibility) 또는 (None, None)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, ts_ms)
    if not result.pose_landmarks:
        return None, None
    lm = result.pose_landmarks[0]
    kp = np.empty((17, 3), dtype=float)
    sc = np.empty(17, dtype=float)
    for i, m in enumerate(COCO_FROM_MP):
        p = lm[m]
        kp[i, 0], kp[i, 1], kp[i, 2] = p.x, p.y, p.z
        sc[i] = p.visibility
    return kp, sc


# ══════════════════════════════════════════════════════════════════
# 1€ 필터 — 키포인트 (17,3) 지터링 제거
# ══════════════════════════════════════════════════════════════════
class _FallbackEuro:
    """lim_punch_features 가 없을 때 쓰는 간이 1€ 필터."""

    def __init__(self, min_cutoff=1.0, beta=10.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        if self._x is None or dt <= 0:
            self._x = x
            return x
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x
        self._x, self._dx = x_hat, dx_hat
        return x_hat


class Kp3Filter:
    """키포인트 x/y/z 채널마다 1€ 필터. 손이 느릴 땐 강하게, 빠를 땐
    약하게 평활 → 가드의 떨림은 지우고 펀치 속도는 보존한다."""

    def __init__(self, n=17):
        euro = _ONE_EURO or _FallbackEuro
        self.f = [[euro() for _ in range(3)] for _ in range(n)]
        self.n = n

    def apply(self, kp, dt):
        out = np.array(kp, dtype=float)
        for i in range(self.n):
            for c in range(3):
                out[i, c] = self.f[i][c](float(kp[i, c]), dt)
        return out


# ══════════════════════════════════════════════════════════════════
# 손별 펀치 이벤트 트래커
# ══════════════════════════════════════════════════════════════════
class HandTracker:
    """한 손의 펀치를 가드 기준 탄도 운동으로 추적한다.

    progress P = reach(가드 대비 손목 변위 xyz) + 0.7·팔꿈치 펴짐.
    P 가 정점을 찍고 회수되면(또는 멈추면) 하나의 펀치 이벤트로 마감,
    정점 프레임의 피처 벡터를 만들어 반환한다.
    """

    def __init__(self, hand):
        self.hand = hand
        if hand == 'left':
            self.wr, self.el, self.sh = L_WR, L_EL, L_SH
        else:
            self.wr, self.el, self.sh = R_WR, R_EL, R_SH
        self.guard = None
        self.min_P = START_P          # PLAY 단계에서 보정값으로 갱신
        self.hard_reset()

    def hard_reset(self):
        self.state = 'idle'
        self.prev = None
        self.cool_until = 0.0
        self.acc = None
        self.live_P = 0.0
        self.live_reach = 0.0

    def set_guard(self, kp, scale):
        self.guard = {
            'rel': (kp[self.wr, :2] - kp[self.sh, :2]) / scale,
            'relz': float(kp[self.wr, 2] - kp[self.sh, 2]),
            'elbow': angle3(kp[self.sh], kp[self.el], kp[self.wr]),
        }

    def guard_from_dict(self, d):
        self.guard = {
            'rel': np.asarray(d['rel'], dtype=float),
            'relz': float(d['relz']),
            'elbow': float(d['elbow']),
        }

    def guard_dict(self):
        return {
            'rel': [float(self.guard['rel'][0]), float(self.guard['rel'][1])],
            'relz': float(self.guard['relz']),
            'elbow': float(self.guard['elbow']),
        }

    def _frame(self, kp, scale):
        rel = (kp[self.wr, :2] - kp[self.sh, :2]) / scale
        relz = float(kp[self.wr, 2] - kp[self.sh, 2])
        move = rel - self.guard['rel']                 # dx, dy (가드 기준)
        dz = self.guard['relz'] - relz                 # + : 카메라 쪽으로
        elbow = angle3(kp[self.sh], kp[self.el], kp[self.wr])
        straighten = max(0.0, elbow - self.guard['elbow'])
        reach = math.sqrt(move[0] ** 2 + move[1] ** 2 + (Z_W * dz) ** 2)
        P = reach + 0.7 * (straighten / 120.0)
        return {
            'rel': rel, 'relz': relz, 'move': move, 'dz': dz,
            'elbow': elbow, 'straighten': straighten,
            'armext': float(np.linalg.norm(rel)), 'reach': reach, 'P': P,
        }

    def update(self, kp, scale, now, calibrating=False):
        """프레임 1장 처리. 펀치 이벤트가 마감되면 dict 반환."""
        if self.guard is None:
            return None
        fr = self._frame(kp, scale)
        cur = np.array([fr['rel'][0], fr['rel'][1], Z_W * fr['relz']])
        speed = 0.0 if self.prev is None else float(np.linalg.norm(cur - self.prev))
        self.prev = cur
        self.live_P = fr['P']
        self.live_reach = fr['reach']
        facedx = abs(float(kp[self.wr, 0] - kp[NOSE, 0])) / scale

        emit = None
        if self.state == 'idle':
            if now >= self.cool_until and fr['P'] >= START_P and speed > 0.012:
                self.state = 'active'
                self.acc = {
                    't0': now, 'peakP': fr['P'], 'stopf': 0,
                    'mnx': fr['move'][0], 'mxx': fr['move'][0],
                    'mny': fr['move'][1], 'mxy': fr['move'][1],
                    'facedx': facedx, 'maxspeed': speed,
                    'peak': self._snap(fr),
                }
        else:
            a = self.acc
            m = fr['move']
            a['mnx'] = min(a['mnx'], m[0])
            a['mxx'] = max(a['mxx'], m[0])
            a['mny'] = min(a['mny'], m[1])
            a['mxy'] = max(a['mxy'], m[1])
            a['facedx'] = min(a['facedx'], facedx)
            a['maxspeed'] = max(a['maxspeed'], speed)
            if fr['P'] > a['peakP']:
                a['peakP'] = fr['P']
                a['peak'] = self._snap(fr)
            a['stopf'] = a['stopf'] + 1 if speed < STOP_SPEED else 0

            retract = fr['P'] < a['peakP'] * RETRACT_RATIO
            timeout = now - a['t0'] > MAX_DUR
            settled = a['stopf'] >= STOP_FRAMES
            done = (retract and a['peakP'] >= START_P) or timeout or \
                   (settled and a['peakP'] >= START_P)
            if done:
                gate = START_P if calibrating else self.min_P
                if a['peakP'] >= gate:
                    emit = self._make_event(a)
                self.state = 'idle'
                self.cool_until = now + COOLDOWN
                self.acc = None
            elif now - a['t0'] > 0.45 and a['peakP'] < START_P:
                self.state = 'idle'
                self.acc = None
        return emit

    @staticmethod
    def _snap(fr):
        return {
            'dx': float(fr['move'][0]), 'dy': float(fr['move'][1]),
            'dz': float(fr['dz']), 'elbow': float(fr['elbow']),
            'straighten': float(fr['straighten']),
            'armext': float(fr['armext']),
        }

    def _make_event(self, a):
        pk = a['peak']
        path_h = a['mxx'] - a['mnx']
        path_v = a['mxy'] - a['mny']
        up = max(0.0, -a['mny'])          # 가드(0)보다 위로 올라간 양
        feat = np.array([
            pk['dx'], pk['dy'], pk['dz'],
            pk['elbow'] / 180.0, pk['straighten'] / 120.0, pk['armext'],
            path_h, path_v, up, a['facedx'], a['maxspeed'],
        ], dtype=float)
        return {'hand': self.hand, 'peakP': float(a['peakP']), 'feature': feat}


# ══════════════════════════════════════════════════════════════════
# 개인 보정 분류기 (k-NN / 최근접 중심)
# ══════════════════════════════════════════════════════════════════
class PunchClassifier:
    """보정 샘플로 만든 z-정규화 + 손별 최근접 중심 분류기."""

    def __init__(self, mean, std, centroids, reject):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.centroids = centroids          # {(hand, punch): zscored vec}
        self.reject = float(reject)

    @staticmethod
    def build(samples):
        """samples: [(hand, punch, feature(np 11)), ...]  (총 30개)."""
        X = np.array([f for _, _, f in samples], dtype=float)
        mean = X.mean(axis=0)
        std = np.maximum(X.std(axis=0), 0.03)
        Z = (X - mean) / std
        groups = defaultdict(list)
        for (hand, punch, _), z in zip(samples, Z):
            groups[(hand, punch)].append(z)
        centroids = {k: np.mean(v, axis=0) for k, v in groups.items()}
        dists = [wdist(z, centroids[(h, p)])
                 for (h, p, _), z in zip(samples, Z)]
        dists = np.array(dists, dtype=float)
        reject = float(np.clip(dists.mean() + 3.0 * dists.std(), 2.5, 9.0))
        return PunchClassifier(mean, std, centroids, reject)

    def classify(self, hand, feat):
        z = (np.asarray(feat, dtype=float) - self.mean) / self.std
        cand = [(p, wdist(z, c)) for (h, p), c in self.centroids.items()
                if h == hand]
        cand.sort(key=lambda t: t[1])
        best, d1 = cand[0]
        d2 = cand[1][1] if len(cand) > 1 else d1 + 1.0
        rejected = d1 > self.reject
        conf = float(np.clip(0.5 + 0.5 * (d2 - d1) / (d2 + 1e-6), 0.5, 0.99))
        return best, conf, d1, rejected

    def to_dict(self):
        return {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'reject': self.reject,
            'centroids': {f'{h}|{p}': c.tolist()
                          for (h, p), c in self.centroids.items()},
        }

    @staticmethod
    def from_dict(d):
        centroids = {}
        for key, vec in d['centroids'].items():
            h, p = key.split('|')
            centroids[(h, p)] = np.asarray(vec, dtype=float)
        return PunchClassifier(d['mean'], d['std'], centroids, d['reject'])


# ══════════════════════════════════════════════════════════════════
# 보정 저장 / 불러오기
# ══════════════════════════════════════════════════════════════════
def save_calibration(path, classifier, trackers, swapped):
    data = {
        'version': 1,
        'swapped': bool(swapped),
        'classifier': classifier.to_dict(),
        'guard': {h: trackers[h].guard_dict() for h in ('left', 'right')},
        'min_P': {h: trackers[h].min_P for h in ('left', 'right')},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def load_calibration(path, trackers):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    classifier = PunchClassifier.from_dict(data['classifier'])
    for h in ('left', 'right'):
        trackers[h].guard_from_dict(data['guard'][h])
        trackers[h].min_P = float(data['min_P'][h])
    return classifier, bool(data.get('swapped', False))


# ══════════════════════════════════════════════════════════════════
# 자세 / 가드 검사
# ══════════════════════════════════════════════════════════════════
def stance_ready(kp, sc):
    """상체·양손 가시성만 확인한다.

    정면/측면을 게이트하지 않는다 — 어깨폭/몸통높이 비율은 복싱 스탠스의
    블레이딩(몸 틀기)과 펀치 동작에 따라 크게 출렁여(0.24~1.33) 뷰 판별에
    쓸 수 없다. 보정이 사용자가 실제 쓰는 뷰를 그대로 학습하므로, 여기서는
    검출이 안정적인지(상체+양손이 보이는지)만 본다."""
    need = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR]
    if any(sc[i] < VIS_MIN for i in need):
        return False, '상체와 양손이 모두 보이도록 서주세요'
    scale = body_scale(kp)
    if abs(kp[L_SH, 1] - kp[R_SH, 1]) / scale > 0.6:
        return False, '카메라가 기울었어요 — 수평으로 맞춰주세요'
    return True, '좋습니다 — 자세를 유지하세요'


def guard_ready(kp, sc):
    """양손이 얼굴 옆 가드 영역에 있는지."""
    if any(sc[i] < VIS_MIN for i in ARM_KP) or sc[NOSE] < VIS_MIN:
        return False
    scale = body_scale(kp)
    nose_y = kp[NOSE, 1]
    sh_y = (kp[L_SH, 1] + kp[R_SH, 1]) * 0.5
    for wr in (L_WR, R_WR):
        wy = kp[wr, 1]
        if not (nose_y - 0.7 * scale <= wy <= sh_y + 0.35 * scale):
            return False
        if abs(kp[wr, 0] - kp[NOSE, 0]) / scale > 1.15:
            return False
    return True


# ══════════════════════════════════════════════════════════════════
# 그리기
# ══════════════════════════════════════════════════════════════════
def draw_panel(img, x, y, w, h, alpha=0.62):
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), C_PANEL, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), C_LINE, 1, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, ratio, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (58, 54, 52), -1)
    fw = int(w * max(0.0, min(1.0, ratio)))
    if fw > 0:
        cv2.rectangle(img, (x, y), (x + fw, y + h), color, -1)


def draw_skeleton(img, kp_px, sc, hot_hand=None):
    for a, b in SKELETON:
        if sc[a] > VIS_MIN and sc[b] > VIS_MIN:
            cv2.line(img, (int(kp_px[a, 0]), int(kp_px[a, 1])),
                     (int(kp_px[b, 0]), int(kp_px[b, 1])), C_SKEL, 2,
                     cv2.LINE_AA)
    for i in range(17):
        if sc[i] > VIS_MIN:
            cv2.circle(img, (int(kp_px[i, 0]), int(kp_px[i, 1])), 3,
                       (210, 205, 215), -1, cv2.LINE_AA)
    for wr, hand, col in ((L_WR, 'left', C_LEFT), (R_WR, 'right', C_RIGHT)):
        if sc[wr] > VIS_MIN:
            r = 13 if hand == hot_hand else 8
            cv2.circle(img, (int(kp_px[wr, 0]), int(kp_px[wr, 1])), r, col,
                       -1, cv2.LINE_AA)


def center_overlay(img, w, h, title, lines, accent=C_INFO):
    """화면 중앙 안내 박스."""
    bw = 720
    bh = 90 + len(lines) * 40
    x = (w - bw) // 2
    y = (h - bh) // 2
    draw_panel(img, x, y, bw, bh, alpha=0.78)
    cv2.rectangle(img, (x, y), (x + bw, y + 5), accent, -1)
    texts = [(title, (x + (bw - text_w(title, 40)) // 2, y + 26), 40, accent)]
    for i, (ln, col) in enumerate(lines):
        tw = text_w(ln, 22)
        texts.append((ln, (x + (bw - tw) // 2, y + 86 + i * 40), 22, col))
    return texts


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='정면 복싱 펀치 트레이너')
    ap.add_argument('--source', default='0', help='카메라 인덱스 또는 영상 경로')
    ap.add_argument('--model', choices=['lite', 'full'], default='lite',
                    help='MediaPipe 포즈 모델 (기본 lite — 모바일/CPU)')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--reset', action='store_true',
                    help='저장된 보정을 무시하고 처음부터')
    args = ap.parse_args()

    landmarker, model_name = build_landmarker(args.model)
    print(f'MediaPipe 포즈: pose_landmarker_{model_name}.task')

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f'카메라/영상 열기 실패: {args.source}')
    is_camera = isinstance(src, int)

    win = '복싱 펀치 트레이너'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    trackers = {'left': HandTracker('left'), 'right': HandTracker('right')}
    kp_filter = Kp3Filter()
    classifier = None
    swapped = False

    have_saved = os.path.exists(CALIB_PATH) and not args.reset

    # ── 단계 상태 ─────────────────────────────────────────────────
    phase = 'STANCE'
    ok_since = None                 # STANCE/HANDS/GUARD 의 유지 타이머
    guard_buf = deque(maxlen=12)    # GUARD 안정 시 median 산출용
    calib_idx = 0
    calib_samples = []              # [(hand, punch, feature)]
    calib_peakP = {'left': [], 'right': []}
    stage_count = 0
    counts = defaultdict(int)
    last_event = None
    flash_until = 0.0
    flash_col = C_OK
    ready_until = 0.0

    def reset_all():
        nonlocal phase, ok_since, calib_idx, calib_samples, stage_count
        nonlocal classifier, last_event, kp_filter
        phase = 'STANCE'
        ok_since = None
        calib_idx = 0
        calib_samples = []
        calib_peakP['left'].clear()
        calib_peakP['right'].clear()
        stage_count = 0
        classifier = None
        last_event = None
        counts.clear()
        guard_buf.clear()
        kp_filter = Kp3Filter()
        for t in trackers.values():
            t.guard = None
            t.min_P = START_P
            t.hard_reset()

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
        h, w = frame.shape[:2]

        ts_ms = int((now - t_start) * 1000) if is_camera \
            else int(frame_idx * 1000.0 / 30.0)
        if ts_ms <= last_ts:
            ts_ms = last_ts + 1
        last_ts = ts_ms
        dt = min(0.2, max(0.005, now - prev_now)) if is_camera else 1.0 / 30.0
        prev_now = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kp_raw, sc = detect_pose(landmarker, rgb, ts_ms)

        kp = None
        scale = 1.0
        if kp_raw is not None:
            if swapped:
                kp_raw = kp_raw[LR_SWAP_IDX]
                sc = sc[LR_SWAP_IDX]
            kp = kp_filter.apply(kp_raw, dt)
            scale = body_scale(kp)

        # ── 단계별 로직 ───────────────────────────────────────────
        status = []          # 화면 상단 안내
        overlay = []         # 중앙 안내 박스 텍스트
        events = []          # 이번 프레임 펀치 이벤트

        if phase == 'STANCE':
            good, msg = (False, '사람을 찾는 중...') if kp is None \
                else stance_ready(kp, sc)
            if good:
                ok_since = ok_since or now
                remain = max(0.0, 1.3 - (now - ok_since))
                lines = [(msg, C_OK),
                         (f'{remain:0.1f}초 후 다음 단계', C_DIM)]
                if remain <= 0:
                    phase, ok_since = 'HANDS', now
            else:
                ok_since = None
                lines = [(msg, C_INFO)]
            tip = ('[L] 저장된 보정으로 바로 시작' if have_saved
                   else '[Q] 종료')
            lines.append((tip, C_DIM))
            overlay = center_overlay(frame, w, h, '1 / 5  자세 확인', lines)

        elif phase == 'HANDS':
            if kp is None or any(sc[i] < VIS_MIN for i in ARM_KP):
                ok_since = None
                lines = [('양손이 보이도록 가드를 올리세요', C_INFO)]
            else:
                ok_since = ok_since or now
                remain = max(0.0, 2.0 - (now - ok_since))
                lines = [('손목의 라벨이 실제와 맞는지 확인하세요', C_TX),
                         ('왼손/오른손이 바뀌었다면  [S]  로 교정', C_DIM),
                         (f'{remain:0.1f}초 후 다음 단계', C_DIM)]
                if remain <= 0:
                    phase, ok_since = 'GUARD', now
                    guard_buf.clear()
            overlay = center_overlay(frame, w, h, '2 / 5  좌우손 인식', lines)

        elif phase == 'GUARD':
            ready = kp is not None and guard_ready(kp, sc)
            if ready:
                guard_buf.append(kp.copy())
                ok_since = ok_since or now
                remain = max(0.0, 1.3 - (now - ok_since))
                lines = [('가드 자세를 등록하는 중...', C_OK),
                         (f'{remain:0.1f}초 — 그대로 유지', C_DIM)]
                if remain <= 0 and len(guard_buf) >= 6:
                    guard_kp = np.median(np.array(guard_buf), axis=0)
                    gscale = body_scale(guard_kp)
                    for t in trackers.values():
                        t.set_guard(guard_kp, gscale)
                        t.hard_reset()
                    phase, ok_since = 'CALIB', now
                    calib_idx = 0
                    stage_count = 0
            else:
                ok_since = None
                guard_buf.clear()
                lines = [('양손을 얼굴 옆으로 올려 가드를 잡으세요', C_INFO)]
            overlay = center_overlay(frame, w, h, '3 / 5  가드 등록', lines)

        elif phase == 'CALIB':
            punch, hand = CALIB_SEQUENCE[calib_idx]
            other = 'right' if hand == 'left' else 'left'
            if kp is not None and all(sc[i] > VIS_MIN for i in ARM_KP):
                ev = trackers[hand].update(kp, scale, now, calibrating=True)
                trackers[other].update(kp, scale, now, calibrating=True)
                if ev is not None:
                    calib_samples.append((hand, punch, ev['feature']))
                    calib_peakP[hand].append(ev['peakP'])
                    stage_count += 1
                    flash_until, flash_col = now + 0.35, C_OK
                    if stage_count >= SAMPLES_PER:
                        calib_idx += 1
                        stage_count = 0
                        trackers['left'].hard_reset()
                        trackers['right'].hard_reset()
                        if calib_idx >= len(CALIB_SEQUENCE):
                            # 보정 완료 → 분류기 + 손별 게이트 구축
                            classifier = PunchClassifier.build(calib_samples)
                            for hh in ('left', 'right'):
                                arr = calib_peakP[hh]
                                med = float(np.median(arr)) if arr else START_P
                                trackers[hh].min_P = float(
                                    np.clip(0.5 * med, 0.16, 0.5))
                            try:
                                save_calibration(CALIB_PATH, classifier,
                                                 trackers, swapped)
                            except OSError:
                                pass
                            phase = 'READY'
                            ready_until = now + 2.0
            else:
                status.append(('양손이 보이도록 서주세요', C_INFO))

            if phase == 'CALIB':
                stage = STAGE_KR[(punch, hand)]
                done_dots = '●' * stage_count + '○' * (SAMPLES_PER - stage_count)
                lines = [(f'{stage}    {done_dots}',
                          C_LEFT if hand == 'left' else C_RIGHT),
                         (STAGE_HINT[punch], C_TX),
                         (f'{SAMPLES_PER}회 반복   ·   [R] 이 단계 다시', C_DIM)]
                title = f'4 / 5  보정  ({calib_idx + 1}/{len(CALIB_SEQUENCE)})'
                overlay = center_overlay(
                    frame, w, h, title, lines,
                    accent=C_LEFT if hand == 'left' else C_RIGHT)

        elif phase == 'READY':
            lines = [('보정 완료 — 6종 펀치를 인식할 준비가 됐습니다', C_OK),
                     ('잽 · 스트레이트 · 좌우 훅 · 좌우 어퍼', C_TX)]
            overlay = center_overlay(frame, w, h, '준비 완료!', lines,
                                     accent=C_OK)
            if now >= ready_until:
                phase = 'PLAY'

        elif phase == 'PLAY':
            if kp is not None and all(sc[i] > VIS_MIN for i in ARM_KP):
                for hand in ('left', 'right'):
                    ev = trackers[hand].update(kp, scale, now)
                    if ev is not None:
                        events.append(ev)
            else:
                status.append(('손목이 가려졌습니다 — 양손을 보여주세요',
                               C_INFO))
            for ev in events:
                punch, conf, dist, rejected = classifier.classify(
                    ev['hand'], ev['feature'])
                if rejected:
                    flash_until, flash_col = now + 0.30, C_WARN
                    last_event = {'label': '인식 실패', 'col': C_WARN,
                                  'conf': conf, 'time': now}
                    continue
                key = (ev['hand'], punch)
                counts[key] += 1
                col = PUNCH_COL[punch]
                flash_until, flash_col = now + 0.45, col
                last_event = {'label': PUNCH_KR[key], 'col': col,
                              'conf': conf, 'time': now}

        # ── 렌더링 ────────────────────────────────────────────────
        if is_camera:
            # 검출/분류는 비반전 좌표로 끝냈다. 표시만 거울로 뒤집고,
            # 골격은 픽셀 좌표를 거울 변환해 다시 그린다.
            frame = cv2.flip(frame, 1)
        if kp is not None:
            kp_px = kp[:, :2].copy()
            if is_camera:
                kp_px[:, 0] = 1.0 - kp_px[:, 0]
            kp_px = kp_px * np.array([w, h])
            hot = None
            if phase in ('CALIB', 'PLAY'):
                hot = max(trackers, key=lambda hd: trackers[hd].live_P)
                if trackers[hot].live_P < START_P * 0.6:
                    hot = None
            draw_skeleton(frame, kp_px, sc, hot_hand=hot)
            # 손 라벨 (거울 좌표에 맞춰 반전 후 그림)
            for wr, txt, col in ((L_WR, '왼손', C_LEFT),
                                 (R_WR, '오른손', C_RIGHT)):
                if sc[wr] > VIS_MIN:
                    lx, ly = int(kp_px[wr, 0]), int(kp_px[wr, 1])
                    status.append((txt, col, (lx + 14, ly - 30)))

        # 단계별 HUD
        hud = []
        if phase in ('CALIB', 'PLAY') and kp is not None:
            # 좌우손 progress 바 (우하단)
            px = w - 266
            draw_panel(frame, px, h - 92, 250, 76)
            hud.append(('손 활동', (px + 12, h - 86), 15, C_DIM))
            for i, (hand, col) in enumerate((('left', C_LEFT),
                                             ('right', C_RIGHT))):
                y = h - 60 + i * 26
                name = '왼손' if hand == 'left' else '오른손'
                hud.append((name, (px + 12, y - 4), 15, col))
                draw_bar(frame, px + 68, y, 172, 10,
                         trackers[hand].live_P / 0.9, col)

        if phase == 'PLAY':
            # 펀치 카운트 패널
            draw_panel(frame, 16, 16, 240, 252)
            hud.append(('펀치 카운트', (28, 24), 18, C_TX))
            for i, (punch, hand) in enumerate(CALIB_SEQUENCE):
                key = (hand, punch)
                y = 58 + i * 30
                col = PUNCH_COL[punch]
                hud.append((PUNCH_KR[key], (28, y), 18, col))
                hud.append((f'{counts[key]:3d}', (200, y), 18, C_TX))
            total = sum(counts.values())
            hud.append((f'합계  {total}', (28, 58 + 6 * 30 + 6), 18, C_DIM))

            # 마지막 펀치 플래시
            if last_event and now < flash_until:
                col = flash_col
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 10)
                lbl = last_event['label']
                lw = text_w(lbl, 64)
                hud.append((lbl, ((w - lw) // 2, h - 150), 64, col))
                if last_event['label'] != '인식 실패':
                    sub = f"{last_event['conf'] * 100:.0f}%"
                    hud.append((sub, ((w - text_w(sub, 28)) // 2, h - 78),
                                28, C_TX))

        # 상단 안내
        for item in status:
            if len(item) == 3 and isinstance(item[2], tuple):
                txt, col, pos = item
                hud.append((txt, pos, 18, col))
            else:
                txt, col = item
                hud.append((txt, ((w - text_w(txt, 22)) // 2, 20), 22, col))

        # 중앙 안내 박스
        hud.extend(overlay)

        # 상태 줄
        fps_n += 1
        if now - fps_t >= 0.5:
            fps = fps_n / (now - fps_t)
            fps_t, fps_n = now, 0
        bar_txt = (f'{model_name}  {fps:4.1f}fps   '
                   f'Q 종료  C 재보정  R 단계다시  S 좌우교정')
        hud.append((bar_txt, (16, h - 26), 15, C_DIM))

        draw_texts(frame, hud)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('c'):
            reset_all()
        if key == ord('r') and phase == 'CALIB':
            stage_count = 0
            calib_samples = [s for s in calib_samples
                             if (s[1], s[0]) != CALIB_SEQUENCE[calib_idx]]
            trackers['left'].hard_reset()
            trackers['right'].hard_reset()
        if key == ord('s'):
            swapped = not swapped
            kp_filter = Kp3Filter()
            for t in trackers.values():
                t.hard_reset()
        if key == ord('l') and phase == 'STANCE' and have_saved:
            try:
                classifier, swapped = load_calibration(CALIB_PATH, trackers)
                kp_filter = Kp3Filter()
                for t in trackers.values():
                    t.hard_reset()
                counts.clear()
                phase = 'PLAY'
                print('저장된 보정을 불러왔습니다.')
            except (OSError, KeyError, ValueError) as exc:
                print(f'보정 불러오기 실패: {exc}')
                have_saved = False

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
