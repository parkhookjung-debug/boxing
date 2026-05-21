"""
lim_punch_features.py — 윈도우 펀치 분류기 공유 피처 모듈

학습(LIM_train_window.py) · 평가(eval_window.py) · 앱(lim_pose_coach_app.py)이
모두 이 모듈을 import 해서 동일한 피처를 사용한다 (single source of truth).

좌표는 어깨폭으로 정규화 → 픽셀/정규 좌표 무관, 평행이동 불변.
"""

import csv
import math
import numpy as np

# ── COCO 17 ────────────────────────────────────────────────
COCO_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HI, R_HI = 11, 12

# ── 윈도우 설정 ─────────────────────────────────────────────
HALF = 7                 # 중심 프레임 기준 ±HALF
WINDOW = HALF * 2 + 1     # = 15
CLASSES = ['none', 'jab', 'cross', 'hook', 'uppercut']
PUNCH_CLASSES = ['jab', 'cross', 'hook', 'uppercut']

# 단일 프레임에서 위치를 읽는 키포인트 (어깨중심 기준 정규화)
_POS_KP = (NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HI, R_HI)


def shoulder_width(kp):
    return float(np.linalg.norm(kp[R_SH, :2] - kp[L_SH, :2])) + 1e-6


def body_scale(kp):
    """뷰 불변 정규화 스케일.

    측면 뷰는 어깨가 겹쳐 어깨폭이 붕괴하므로, 카메라 각도에 둔감한
    몸통 높이(어깨중심↔엉덩이중심)를 우선 사용. 엉덩이가 안 보이면 어깨폭.
    """
    sh_c = (kp[L_SH, :2] + kp[R_SH, :2]) * 0.5
    hi_c = (kp[L_HI, :2] + kp[R_HI, :2]) * 0.5
    torso = float(np.linalg.norm(sh_c - hi_c))
    sw = float(np.linalg.norm(kp[R_SH, :2] - kp[L_SH, :2]))
    return max(torso, sw, 1e-6)


def view_ratio(kp):
    """어깨폭 / 몸통높이. 작으면 측면, 크면 정면. 뷰 인지용 피처."""
    sh_c = (kp[L_SH, :2] + kp[R_SH, :2]) * 0.5
    hi_c = (kp[L_HI, :2] + kp[R_HI, :2]) * 0.5
    torso = float(np.linalg.norm(sh_c - hi_c)) + 1e-6
    sw = float(np.linalg.norm(kp[R_SH, :2] - kp[L_SH, :2]))
    return min(2.0, sw / torso)


def _angle(kp, a, b, c):
    """관절 각도 (도) — b 가 꼭짓점."""
    ba = kp[a, :2] - kp[b, :2]
    bc = kp[c, :2] - kp[b, :2]
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cos_v = max(-1.0, min(1.0, float(np.dot(ba, bc)) / denom))
    return math.degrees(math.acos(cos_v))


def frame_feat(kp):
    """단일 프레임 정규화 피처 (23차원). 몸통높이 정규화 — 뷰 불변."""
    s = body_scale(kp)
    center = (kp[L_SH, :2] + kp[R_SH, :2]) * 0.5
    f = []
    for idx in _POS_KP:
        f.append(float((kp[idx, 0] - center[0]) / s))
        f.append(float((kp[idx, 1] - center[1]) / s))
    # 관절 각도 (0~1 정규화)
    f.append(_angle(kp, L_SH, L_EL, L_WR) / 180.0)
    f.append(_angle(kp, R_SH, R_EL, R_WR) / 180.0)
    # 팔 신전 (어깨→손목 / 몸통높이)
    f.append(float(np.linalg.norm(kp[L_WR, :2] - kp[L_SH, :2]) / s))
    f.append(float(np.linalg.norm(kp[R_WR, :2] - kp[R_SH, :2]) / s))
    # 뷰 인지 (정면/측면)
    f.append(view_ratio(kp))
    return f


FRAME_FEAT_DIM = 23


def window_feat(kps, i):
    """
    kps: indexable sequence of kp arrays (각 (17, >=2)).
    i:   중심 프레임 인덱스.
    반환: 윈도우 피처 리스트 (FRAME_FEAT_DIM*WINDOW + 동역학 10).
    경계는 clamp.
    """
    n = len(kps)
    feat = []
    for off in range(-HALF, HALF + 1):
        j = min(n - 1, max(0, i + off))
        feat.extend(frame_feat(kps[j]))

    # 윈도우 내 손목 동역학 요약 (좌/우)
    s = body_scale(kps[min(n - 1, max(0, i))])
    for wr in (L_WR, R_WR):
        arr = np.array([kps[min(n - 1, max(0, i + o))][wr, :2]
                        for o in range(-HALF, HALF + 1)], dtype=float)
        d = np.diff(arr, axis=0)
        spd = np.linalg.norm(d, axis=1) / s
        feat.append(float(spd.max()))
        feat.append(float(spd.sum()))
        feat.append(float((arr[:, 0].max() - arr[:, 0].min()) / s))
        feat.append(float((arr[:, 1].max() - arr[:, 1].min()) / s))
        feat.append(float((arr[0, 1] - arr[:, 1].min()) / s))  # 상승량
    return feat


WINDOW_FEAT_DIM = FRAME_FEAT_DIM * WINDOW + 10


# ── CSV 로더 ────────────────────────────────────────────────
def load_kp_csv(path):
    """LIM_full_data*.csv → (frame_numbers list, kps list of (17,2) ndarray)."""
    frame_nums, kps = [], []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            kp = np.zeros((17, 2), dtype=float)
            for i, name in enumerate(COCO_NAMES):
                kp[i, 0] = float(row[f'{name}_x'])
                kp[i, 1] = float(row[f'{name}_y'])
            frame_nums.append(int(row['frame_number']))
            kps.append(kp)
    return frame_nums, kps


def load_label_csv(path):
    """LIM*_labels.csv → [(frame_number, punch_type), ...] 정렬."""
    events = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            events.append((int(row['frame_number']),
                           row['punch_type'].strip().lower()))
    return sorted(events)


# ── 이벤트 트래커 (앱·평가 공유) ─────────────────────────────
# 프레임별 분류 확률을 받아 펀치 이벤트로 묶는다.
# 같은 펀치 클래스가 연속 min_run 프레임 이상 → 1개 이벤트.
# 클래스가 바뀌거나 none 으로 떨어지면 직전 run 을 마감/방출.
DEFAULT_ENTER_THR = 0.50
DEFAULT_MIN_RUN = 3
DEFAULT_MAX_RUN = 22
DEFAULT_REFRACTORY = 6


class PunchEventTracker:
    """프레임별 확률 → 펀치 이벤트 스트림. 앱(온라인)·평가(오프라인) 공용."""

    def __init__(self, enter_thr=DEFAULT_ENTER_THR, min_run=DEFAULT_MIN_RUN,
                 max_run=DEFAULT_MAX_RUN, refractory=DEFAULT_REFRACTORY):
        self.enter_thr = enter_thr
        self.min_run = min_run
        self.max_run = max_run
        self.refractory = refractory
        self._run_cls = None          # 현재 run 의 펀치 클래스 인덱스 (1..4)
        self._run = []                # [(frame_no, probs ndarray), ...]
        self._last_emit = -10 ** 9

    def _close(self):
        """현재 run 을 마감해 이벤트(또는 None) 반환."""
        run, cls = self._run, self._run_cls
        self._run, self._run_cls = [], None
        if cls is None or len(run) < self.min_run:
            return None
        prob_sum = np.sum([p for _, p in run], axis=0)
        prob_sum[0] = -1.0  # none 제외
        best = int(np.argmax(prob_sum))
        # 이벤트 프레임 = punch 확률 최대 프레임
        peak_fno = max(run, key=lambda fp: 1.0 - fp[1][0])[0]
        if peak_fno - self._last_emit < self.refractory:
            return None
        self._last_emit = peak_fno
        conf = float(prob_sum[best] / len(run))
        return {'frame': peak_fno, 'class_idx': best,
                'punch_type': CLASSES[best], 'confidence': conf,
                'run_len': len(run)}

    def push(self, frame_no, probs):
        """probs: CLASSES 순서 확률. 이벤트 발생 시 dict, 아니면 None."""
        probs = np.asarray(probs, dtype=float)
        punch_p = 1.0 - probs[0]
        punch_idx = 1 + int(np.argmax(probs[1:]))
        emitted = None
        if punch_p >= self.enter_thr:
            if self._run_cls == punch_idx:
                self._run.append((frame_no, probs))
                if len(self._run) >= self.max_run:
                    emitted = self._close()
            else:
                emitted = self._close()
                self._run_cls = punch_idx
                self._run = [(frame_no, probs)]
        else:
            emitted = self._close()
        return emitted

    def flush(self):
        """시퀀스 종료 시 남은 run 마감 (오프라인 평가용)."""
        return self._close()
