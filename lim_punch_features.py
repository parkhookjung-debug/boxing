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


# ── 모션 게이트 ─────────────────────────────────────────────
# 정적인 자세는 펀치가 될 수 없다 (펀치 = 손목이 어깨 대비 빠르게 이동).
# 단일 복서(LIM) 학습 모델은 LIM 의 "쉬는" 프레임도 늘 움직임이 있어,
# 신규 사용자의 '완전 정적' 가드를 학습한 적이 없다 → 가장 가까운 펀치로
# 오인(특히 어퍼). 어깨중심 기준 상대 손목속도가 임계 미만이면 none 으로
# 강제하는 런타임 안전장치. 상대속도라 풋워크/바디무빙엔 둔감.
MOTION_GATE_THR = 0.04    # 윈도우 최대 어깨상대 손목속도 < 이 값 → none


def wrist_gate_speed(window):
    """window: WINDOW개 kp (각 (17,>=2)).
    어깨중심 기준 좌/우 손목의 윈도우 내 최대 상대 이동속도(몸통높이 정규화)."""
    w = np.asarray(window, dtype=float)
    sh_c = (w[:, L_SH, :2] + w[:, R_SH, :2]) * 0.5
    s = body_scale(w[HALF])
    g = 0.0
    for wr in (L_WR, R_WR):
        rel = w[:, wr, :2] - sh_c
        spd = np.linalg.norm(np.diff(rel, axis=0), axis=1) / s
        g = max(g, float(spd.max()))
    return g


# ── 1€ 필터 — 키포인트 지터링 제거 ──────────────────────────
# 경량 포즈 모델은 정지 상태에서도 좌표가 미세하게 떨린다(jittering).
# 단순 이동평균은 떨림과 함께 빠른 펀치 동작까지 뭉개지만, 1€ 필터는
# 손이 느릴 때(가드)는 강하게 / 빠를 때(펀치)는 약하게 평활하므로
# 펀치 속도 신호를 살리면서 떨림만 제거한다. dt 기반이라 모바일의
# 가변 FPS 에도 강건하다. (Casiez, Roussel, Vogel 2012)
#
# 학습(LIM_train_window)·평가(eval_window)·앱이 모두 같은 인과적 필터를
# 거치게 해 학습/런타임 좌표 분포를 일치시킨다 (load_kp_csv 가 기본 적용).
# 값은 tune_filter.py 스윕으로 확정 (이벤트 LOVO Macro-F1 0.787→0.797).
SMOOTH_MIN_CUTOFF = 1.0   # 정지 시 컷오프(Hz) — 낮을수록 더 부드러움
SMOOTH_BETA = 10.0        # 속도 반응 계수 — 높을수록 빠른 동작의 지연↓
SMOOTH_D_CUTOFF = 1.0     # 미분(속도) 추정용 컷오프
DEFAULT_FPS = 30.0


def _euro_alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """스칼라 신호용 1€ 필터. 가변 dt 지원, 인과적(online)."""

    def __init__(self, min_cutoff=SMOOTH_MIN_CUTOFF, beta=SMOOTH_BETA,
                 d_cutoff=SMOOTH_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None       # 직전 필터링 값
        self._dx = 0.0       # 직전 필터링 미분

    def __call__(self, x, dt):
        if self._x is None or dt <= 0:
            self._x = x
            return x
        dx = (x - self._x) / dt
        a_d = _euro_alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _euro_alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x
        self._x, self._dx = x_hat, dx_hat
        return x_hat


class KeypointFilter:
    """키포인트 x/y 좌표마다 1€ 필터 적용. 시퀀스(영상)마다 새로 생성."""

    def __init__(self, n_kp=17, min_cutoff=SMOOTH_MIN_CUTOFF,
                 beta=SMOOTH_BETA, d_cutoff=SMOOTH_D_CUTOFF):
        self.n_kp = n_kp
        self._fx = [OneEuroFilter(min_cutoff, beta, d_cutoff)
                    for _ in range(n_kp)]
        self._fy = [OneEuroFilter(min_cutoff, beta, d_cutoff)
                    for _ in range(n_kp)]

    def apply(self, kp, dt):
        """kp: (n_kp,>=2) → 평활된 (n_kp,2) ndarray."""
        kp = np.asarray(kp, dtype=float)
        out = np.array(kp[:, :2], dtype=float)
        for i in range(self.n_kp):
            out[i, 0] = self._fx[i](float(kp[i, 0]), dt)
            out[i, 1] = self._fy[i](float(kp[i, 1]), dt)
        return out


def filter_kp_sequence(frame_nums, kps, fps=DEFAULT_FPS,
                       min_cutoff=None, beta=None):
    """CSV 키포인트 시퀀스를 런타임과 동일한 인과적 1€ 필터로 평활.

    프레임 누락 구간은 frame_number 차이로 dt 를 보정한다.
    min_cutoff/beta 를 생략하면 모듈 전역값(SMOOTH_*)을 호출 시점에 읽는다."""
    if not kps:
        return kps
    if min_cutoff is None:
        min_cutoff = SMOOTH_MIN_CUTOFF
    if beta is None:
        beta = SMOOTH_BETA
    kf = KeypointFilter(n_kp=np.asarray(kps[0]).shape[0],
                        min_cutoff=min_cutoff, beta=beta)
    out, prev_fn = [], None
    for fn, kp in zip(frame_nums, kps):
        dt = 1.0 / fps if prev_fn is None else max(1, fn - prev_fn) / fps
        out.append(kf.apply(kp, dt))
        prev_fn = fn
    return out


# ── 수평반전 증강 ───────────────────────────────────────────
# 펀치 종류(잽/크로스/훅/어퍼)는 손잡이와 무관하다 — 잽은 어느 손으로 쳐도
# 잽이다. 좌우를 뒤집은 표본을 학습에 더하면 모델이 "LIM 의 왼손이 여기"
# 같은 손잡이 단서를 외우지 못하고 동작 패턴 자체로 분류하게 된다. 결과:
# 사우스포·거울모드·신규 사용자에 강건 (단일 복서 학습의 OOD '어퍼 폴백'
# 완화 — 학습 표본도 사실상 2배).
# COCO-17 좌우 대칭쌍 — 수평반전 시 교환할 인덱스.
COCO_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def flip_kp(kp):
    """키포인트 (17,>=2) 수평반전 — 좌우 랜드마크 교환 + x 미러(1-x).

    정규화 좌표(0~1) 가정. 다운스트림 피처는 어깨중심 기준 상대좌표라
    미러 기준점(1.0)은 상쇄되므로 펀치 분류 결과에 치우침을 주지 않는다."""
    kp = np.asarray(kp, dtype=float)
    out = kp[COCO_FLIP_IDX].copy()
    out[:, 0] = 1.0 - out[:, 0]
    return out


def flip_kp_sequence(kps):
    """kp 시퀀스 전체를 수평반전.

    1€ 필터는 수평반전과 가환이다 — 필터(1-x) = 1-필터(x) 가 정확히
    성립(필터는 입력의 볼록결합). 따라서 평활이 끝난 시퀀스를 반전해도
    '반전 후 평활'과 결과가 같아, load_kp_csv 출력을 그대로 증강에 써도
    학습 분포가 일치한다."""
    return [flip_kp(kp) for kp in kps]


# ── CSV 로더 ────────────────────────────────────────────────
def load_kp_csv(path, smooth=True, fps=DEFAULT_FPS):
    """LIM_full_data*.csv → (frame_numbers list, kps list of (17,2) ndarray).

    smooth=True 면 런타임 앱과 동일한 1€ 필터를 통과시켜 반환한다
    (학습·평가·런타임 좌표 분포 일치 — 지터링 보정 포함)."""
    frame_nums, kps = [], []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            kp = np.zeros((17, 2), dtype=float)
            for i, name in enumerate(COCO_NAMES):
                kp[i, 0] = float(row[f'{name}_x'])
                kp[i, 1] = float(row[f'{name}_y'])
            frame_nums.append(int(row['frame_number']))
            kps.append(kp)
    if smooth and kps:
        kps = filter_kp_sequence(frame_nums, kps, fps)
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
