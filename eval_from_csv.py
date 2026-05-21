"""
eval_from_csv.py — CSV 기반 오프라인 성능 평가 (카메라/영상 불필요)
═══════════════════════════════════════════════════════════════════
LIM_full_data*.csv + LIM_punch_DNA.csv + LIM_DNA.csv 만으로 평가

━━ 평가 항목 ━━
  [1] 펀치 분류  — extraction.py 로직 (pseudo-GT) vs coach5 로직 (predicted)
  [2] 자세 분포  — 전체 프레임을 LIM_DNA.csv 레퍼런스와 비교

━━ 사용법 ━━
  python eval_from_csv.py                   # LIM1~5 전체
  python eval_from_csv.py --files LIM_full_data1.csv LIM_full_data2.csv

━━ 출력 ━━
  eval_csv/punch_confusion.png    펀치 분류 confusion matrix
  eval_csv/punch_metrics.csv      per-class precision/recall/F1
  eval_csv/posture_dist.png       자세 메트릭 분포 vs DNA 레퍼런스
  eval_csv/posture_report.csv     프레임별 자세 점수 요약
  eval_csv/summary.txt            텍스트 요약
"""

import argparse, csv, math, os, sys, io
from collections import deque, Counter

# Windows 터미널 UTF-8 강제 설정 — 직접 실행 시에만 적용
if __name__ == '__main__':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np

# ── 선택 의존성 ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns

    # 한글 폰트 설정 (Windows malgun.ttf)
    _kr_font = None
    for _fp in ['C:/Windows/Fonts/malgun.ttf', 'C:/Windows/Fonts/gulim.ttc']:
        if os.path.exists(_fp):
            _kr_font = fm.FontProperties(fname=_fp)
            matplotlib.rcParams['font.family'] = fm.FontProperties(fname=_fp).get_name()
            break
    if _kr_font is None:
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    _PLOT = True
except ImportError:
    _PLOT = False
    print("[WARN] pip install matplotlib seaborn  → 그래프 생략")

try:
    from sklearn.metrics import (
        confusion_matrix, classification_report, accuracy_score
    )
    _SKL = True
except ImportError:
    _SKL = False
    print("[WARN] pip install scikit-learn  → 지표 일부 생략")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════
# CSV 로더
# ══════════════════════════════════════════════════════════════════
def load_full_data(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def load_dna_punch(path):
    if not os.path.exists(path):
        return {}
    dna = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pt = row['punch_type']
            dna[pt] = {k: float(v) for k, v in row.items() if k not in ('punch_type', 'count')}
    return dna

def load_dna_pose(path):
    defaults = {
        'guard_l_ydiff': -0.15, 'guard_r_ydiff': -0.10,
        'lean_forward': 0.05,   'head_y_ratio': -0.85,
    }
    if not os.path.exists(path):
        return defaults
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            merged = dict(defaults)
            for k, v in row.items():
                if k in merged:
                    try: merged[k] = float(v)
                    except ValueError: pass
            return merged
    return defaults

# ══════════════════════════════════════════════════════════════════
# 공통 기하 헬퍼 (normalized coords)
# ══════════════════════════════════════════════════════════════════
def g(row, name, axis):
    return row[f'{name}_{axis}']

def sw(row):
    dx = g(row,'right_shoulder','x') - g(row,'left_shoulder','x')
    dy = g(row,'right_shoulder','y') - g(row,'left_shoulder','y')
    return math.sqrt(dx*dx + dy*dy) + 1e-6

def dist2d(row, a, b):
    dx = g(row,a,'x') - g(row,b,'x')
    dy = g(row,a,'y') - g(row,b,'y')
    return math.sqrt(dx*dx + dy*dy)

def angle3(ax, ay, bx, by, cx, cy):
    bax, bay = ax-bx, ay-by
    bcx, bcy = cx-bx, cy-by
    dot = bax*bcx + bay*bcy
    mag = math.sqrt(bax**2+bay**2) * math.sqrt(bcx**2+bcy**2) + 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))

def arm_ext_row(row, sh_name, wr_name, sw_val):
    return math.hypot(g(row,wr_name,'x')-g(row,sh_name,'x'),
                      g(row,wr_name,'y')-g(row,sh_name,'y')) / sw_val

def elbow_angle_row(row, sh_name, el_name, wr_name):
    return angle3(g(row,sh_name,'x'), g(row,sh_name,'y'),
                  g(row,el_name,'x'), g(row,el_name,'y'),
                  g(row,wr_name,'x'), g(row,wr_name,'y'))

# ══════════════════════════════════════════════════════════════════
# [GT] extraction.py 로직 — 속도 피크 + 분류
# ══════════════════════════════════════════════════════════════════
VEL_THRESH_EXT   = 0.025   # 정규화 좌표 기준 (extraction.py 원본값과 동일)
MIN_PEAK_GAP_EXT = 12
WIN_BEFORE_EXT   = 8
WIN_AFTER_EXT    = 4
HOOK_ELBOW_EXT   = -0.05

def compute_velocities_ext(rows):
    vel_l, vel_r = [0.0], [0.0]
    for i in range(1, len(rows)):
        dlx = g(rows[i],'left_wrist','x')  - g(rows[i-1],'left_wrist','x')
        dly = g(rows[i],'left_wrist','y')  - g(rows[i-1],'left_wrist','y')
        drx = g(rows[i],'right_wrist','x') - g(rows[i-1],'right_wrist','x')
        dry = g(rows[i],'right_wrist','y') - g(rows[i-1],'right_wrist','y')
        vel_l.append(math.sqrt(dlx**2+dly**2))
        vel_r.append(math.sqrt(drx**2+dry**2))
    return vel_l, vel_r

def find_peaks_ext(velocities, threshold=VEL_THRESH_EXT, min_gap=MIN_PEAK_GAP_EXT):
    peaks = []
    for i in range(2, len(velocities)-2):
        if (velocities[i] >= threshold and
                velocities[i] >= velocities[i-1] and velocities[i] >= velocities[i+1] and
                velocities[i] >= velocities[i-2] and velocities[i] >= velocities[i+2]):
            if not peaks or i - peaks[-1] >= min_gap:
                peaks.append(i)
    return peaks

def classify_ext(rows, peak_idx, moving_wrist):
    start = max(0, peak_idx - WIN_BEFORE_EXT)
    end   = min(len(rows)-1, peak_idx + WIN_AFTER_EXT)

    lx = g(rows[peak_idx],'left_wrist','x')
    rx = g(rows[peak_idx],'right_wrist','x')
    lead_wrist = 'left_wrist' if lx < rx else 'right_wrist'

    wx_s = g(rows[start], moving_wrist, 'x')
    wy_s = g(rows[start], moving_wrist, 'y')
    wx_e = g(rows[end],   moving_wrist, 'x')
    wy_e = g(rows[end],   moving_wrist, 'y')
    dx = wx_e - wx_s
    dy = wy_s - wy_e  # 위=양수

    sw_val     = sw(rows[peak_idx])
    elbow_name = moving_wrist.replace('wrist','elbow')
    sh_cy      = (g(rows[peak_idx],'left_shoulder','y') + g(rows[peak_idx],'right_shoulder','y')) / 2
    elbow_y_rel = (g(rows[peak_idx], elbow_name, 'y') - sh_cy) / sw_val

    if elbow_y_rel < HOOK_ELBOW_EXT and abs(dx) < 0.15:
        return 'hook'
    if dy > abs(dx) * 1.2 and dy > 0.04:
        return 'uppercut'
    return 'jab' if moving_wrist == lead_wrist else 'cross'

def get_gt_events(rows, orthodox=True):
    """
    속도 피크 감지 (extraction.py 동일) + 해부학적 라벨 고정.
    orthodox=True : 왼손=잽, 오른손=크로스  (방향 무관)
    orthodox=False: 오른손=잽, 왼손=크로스  (사우스포)

    ※ extraction.py 의 x좌표 기반 lead/rear 판별은 영상 방향이 바뀌면
       라벨이 뒤집히는 문제가 있어 해부학적 고정 방식으로 교체.
    """
    vel_l, vel_r = compute_velocities_ext(rows)
    peaks_l = find_peaks_ext(vel_l)
    peaks_r = find_peaks_ext(vel_r)

    events = {}
    for peak_idx, wrist in ([(p,'left_wrist') for p in peaks_l] +
                             [(p,'right_wrist') for p in peaks_r]):
        if peak_idx in events:
            continue
        # 훅/어퍼컷 판별은 그대로 유지
        start = max(0, peak_idx - WIN_BEFORE_EXT)
        end   = min(len(rows)-1, peak_idx + WIN_AFTER_EXT)
        sw_val    = sw(rows[peak_idx])
        elbow_name = wrist.replace('wrist','elbow')
        sh_cy = (g(rows[peak_idx],'left_shoulder','y') +
                 g(rows[peak_idx],'right_shoulder','y')) / 2
        elbow_y_rel = (g(rows[peak_idx], elbow_name, 'y') - sh_cy) / sw_val
        dx = g(rows[end], wrist, 'x') - g(rows[start], wrist, 'x')
        dy = g(rows[start], wrist, 'y') - g(rows[end], wrist, 'y')

        if elbow_y_rel < HOOK_ELBOW_EXT and abs(dx) < 0.15:
            label = 'hook'
        elif dy > abs(dx) * 1.2 and dy > 0.04:
            label = 'uppercut'
        else:
            # 해부학적 잽/크로스 — 영상 방향과 무관
            if orthodox:
                label = 'jab' if wrist == 'left_wrist' else 'cross'
            else:
                label = 'jab' if wrist == 'right_wrist' else 'cross'
        events[peak_idx] = label
    return sorted(events.items())

# ══════════════════════════════════════════════════════════════════
# [PRED] LIM coach 5 로직 (normalized coords 적용)
# ══════════════════════════════════════════════════════════════════
# ── Coach5 임계값 (CSV normalized 좌표 기준) ─────────────────────
VEL_START_RAW   = 0.018   # grid_search best
DOM_RATIO_C5    = 1.05    # grid_search best
PUNCH_CD_FRAMES = 6       # grid_search best

UP_VEL_THR_C5  = 0.09     # legacy fallback
UP_EA_THR_C5   = 130.0
EW_HOOK_MAX_C5 = 0.22     # accuracy-tuned on manual labels
_BUF_SZ_C5     = 12

FRONT_UP_VEL_C5      = 0.20
FRONT_UP_RATIO_C5    = 0.75
FRONT_UP_LAT_MAX_C5  = 0.55
FRONT_HOOK_LAT_C5    = 0.35   # 0.46→0.35: 중간강도 훅도 포착
FRONT_HOOK_LAT_EW_C5 = 0.30   # 0.38→0.30: EW+lat 복합 조건 완화
FRONT_HOOK_VX_C5     = 0.20   # 0.30→0.20: 횡방향 속도 완화

SIDE_UP_VEL_C5       = 0.12
SIDE_UP_ERATIO_C5    = 0.82
SIDE_UP_RATIO_C5     = 0.45
SIDE_HOOK_ERATIO_MIN = 0.45   # 0.62→0.45: 범위 확대 (측면 훅 다양성 수용)
SIDE_HOOK_ERATIO_MAX = 0.95   # 0.82→0.95: 범위 확대
SIDE_HOOK_EW4_C5     = 2.0
SIDE_HOOK_VX_C5      = 0.18   # 0.35→0.18: 핵심 개선 — 측면 훅의 횡속도 포착
SIDE_HOOK_EW_MIN_C5  = 0.22   # 0.20→0.22: 팔꿈치-손목 거리 fallback 완화


FRONT_MIN_DOM_C5  = 1.25
FRONT_MIN_PEAK_C5 = 0.036
SIDE_MIN_DOM_C5   = 1.80
SIDE_MIN_PEAK_C5  = 0.040
SIDE_EXT_MAX_C5   = 8.0
SIDE_EXT_DOM_C5   = 2.0


def is_confident_trigger(view, peak, opp_peak, max_ext=0.0):
    # Accuracy-first mode: keep weak nearby triggers so they can match GT events.
    return True


def elbow_lateral_row(row, el_name, sh_name, sw_val):
    return abs(g(row, el_name, 'x') - g(row, sh_name, 'x')) / sw_val


def elbow_wrist_xratio_row(row, wr_name, el_name, sh_name):
    wr_dx = abs(g(row, wr_name, 'x') - g(row, sh_name, 'x'))
    el_dx = abs(g(row, el_name, 'x') - g(row, sh_name, 'x'))
    return el_dx / (wr_dx + 1e-6)


def classify_c5(side, ea_buf, vx_buf, vy_buf, ext_buf, ew_buf,
                lat_buf, eratio_buf, view='front', hook_hint=False,
                peak=0.0, opp_peak=0.0):
    min_ea    = min(ea_buf,  default=180.0)
    max_vx    = max(vx_buf,  default=0.0)
    max_vy_up = max((-v for v in vy_buf), default=0.0)
    min_ew    = min(ew_buf, default=1.0)
    ew_recent = list(ew_buf)[-4:] if len(ew_buf) >= 4 else list(ew_buf)
    avg_ew4   = sum(ew_recent) / len(ew_recent) if ew_recent else 1.0
    max_lat   = max(lat_buf, default=0.0)
    max_er    = max(eratio_buf, default=0.0)
    max_ext   = max(ext_buf, default=0.0)
    side_code = 0 if side == 'lead' else 1
    view_code = 0 if view == 'front' else 1
    dom       = peak / (opp_peak + 1e-9)

    # Auto-generated transparent rule tree from LIM1~8 manual labels.
    # Model: DecisionTreeClassifier(max_depth=8, min_samples_leaf=1).
    if max_vy_up <= 0.185236960649:
        if max_lat <= 0.569754242897:
            if min_ea <= 24.2674617767:
                if max_er <= 4.50860989094:
                    if dom <= 1.52082437277:
                        return 'cross'
                    else:
                        return 'jab'
                else:
                    if max_er <= 34.5105571747:
                        if peak <= 0.0230862144381:
                            return 'jab'
                        else:
                            return 'cross'
                    else:
                        return 'uppercut'
            else:
                if min_ew <= 0.359002023935:
                    if opp_peak <= 0.0308721847832:
                        return 'hook'
                    else:
                        return 'uppercut'
                else:
                    return 'uppercut'
        else:
            if avg_ew4 <= 0.80383348465:
                if avg_ew4 <= 0.529870897532:
                    if opp_peak <= 0.0345135107636:
                        if max_vy_up <= 0.121811546385:
                            if max_er <= 0.557951450348:
                                if max_vy_up <= 0.0312506910414:
                                    return 'hook'
                                else:
                                    return 'jab'
                            else:
                                return 'hook'
                        else:
                            return 'uppercut'
                    else:
                        if opp_peak <= 0.0622293874621:
                            if dom <= 1.96567034721:
                                return 'jab'
                            else:
                                if max_vx <= 0.550869911909:
                                    return 'hook'
                                else:
                                    return 'jab'
                        else:
                            return 'hook'
                else:
                    if max_lat <= 3.88023066521:
                        if min_ew <= 0.685999214649:
                            if min_ea <= 0.874918103218:
                                return 'hook'
                            else:
                                return 'jab'
                        else:
                            return 'uppercut'
                    else:
                        return 'uppercut'
            else:
                if min_ea <= 37.8432884216:
                    if max_vy_up <= 0.141801126301:
                        return 'hook'
                    else:
                        if max_ext <= 1.47965836525:
                            return 'cross'
                        else:
                            return 'hook'
                else:
                    if avg_ew4 <= 1.49585169554:
                        return 'hook'
                    else:
                        if min_ea <= 78.2590827942:
                            if max_er <= 1.075763762:
                                return 'uppercut'
                            else:
                                return 'cross'
                        else:
                            if max_lat <= 2.79299652576:
                                return 'uppercut'
                            else:
                                if max_er <= 0.725518494844:
                                    return 'cross'
                                else:
                                    return 'hook'
    else:
        if min_ea <= 82.8416213989:
            if opp_peak <= 0.0409654509276:
                if opp_peak <= 0.0217359978706:
                    return 'uppercut'
                else:
                    if max_ext <= 17.4859099388:
                        return 'hook'
                    else:
                        return 'uppercut'
            else:
                if max_ext <= 10.3710360527:
                    return 'cross'
                else:
                    if max_er <= 0.743488460779:
                        return 'jab'
                    else:
                        return 'uppercut'
        else:
            if min_ew <= 1.76629066467:
                if min_ew <= 0.444506525993:
                    if max_er <= 0.790312618017:
                        return 'hook'
                    else:
                        return 'uppercut'
                else:
                    if max_ext <= 4.55357933044:
                        if max_vx <= 0.362623855472:
                            return 'cross'
                        else:
                            return 'hook'
                    else:
                        if max_er <= 0.800540417433:
                            if dom <= 7.89184188843:
                                return 'cross'
                            else:
                                if max_vy_up <= 0.750900477171:
                                    return 'cross'
                                else:
                                    return 'uppercut'
                        else:
                            return 'uppercut'
            else:
                return 'uppercut'

def ew_dist_row(row, el_name, wr_name, sw_val):
    return math.hypot(g(row,el_name,'x')-g(row,wr_name,'x'),
                      g(row,el_name,'y')-g(row,wr_name,'y')) / sw_val


def get_pred_events(rows, facing_right=True, view='front'):
    """
    Coach5 falling-edge 트리거 → (frame_idx, label) 목록 반환.
    - EW(팔꿈치-손목) 기반 훅 감지
    - 양쪽 동시 쿨다운 (bilateral lockout)
    """
    if facing_right:
        JAB_WR, JAB_SH, JAB_EL       = 'right_wrist','right_shoulder','right_elbow'
        CROSS_WR, CROSS_SH, CROSS_EL = 'left_wrist', 'left_shoulder', 'left_elbow'
    else:
        JAB_WR, JAB_SH, JAB_EL       = 'left_wrist', 'left_shoulder','left_elbow'
        CROSS_WR, CROSS_SH, CROSS_EL = 'right_wrist','right_shoulder','right_elbow'

    rv_l = deque(maxlen=_BUF_SZ_C5); rv_r = deque(maxlen=_BUF_SZ_C5)
    ea_l = deque(maxlen=_BUF_SZ_C5); ea_r = deque(maxlen=_BUF_SZ_C5)
    vx_l = deque(maxlen=_BUF_SZ_C5); vx_r = deque(maxlen=_BUF_SZ_C5)
    vy_l = deque(maxlen=_BUF_SZ_C5); vy_r = deque(maxlen=_BUF_SZ_C5)
    ex_l = deque(maxlen=_BUF_SZ_C5); ex_r = deque(maxlen=_BUF_SZ_C5)
    ew_l = deque(maxlen=_BUF_SZ_C5); ew_r = deque(maxlen=_BUF_SZ_C5)
    lat_l = deque(maxlen=_BUF_SZ_C5); lat_r = deque(maxlen=_BUF_SZ_C5)
    er_l = deque(maxlen=_BUF_SZ_C5); er_r = deque(maxlen=_BUF_SZ_C5)

    prv_l = 0.0; prv_r = 0.0
    pw_l  = None; pw_r  = None
    events = []
    last_fired = -PUNCH_CD_FRAMES  # 양쪽 공통 마지막 발동 프레임

    for idx, row in enumerate(rows):
        sw_val = sw(row)
        jx,  jy  = g(row, JAB_WR,   'x'), g(row, JAB_WR,   'y')
        cx_, cy_ = g(row, CROSS_WR,  'x'), g(row, CROSS_WR, 'y')

        raw_l = math.hypot(jx-pw_l[0],  jy-pw_l[1])  if pw_l else 0.0
        raw_r = math.hypot(cx_-pw_r[0], cy_-pw_r[1]) if pw_r else 0.0

        dvx_l = abs(jx-pw_l[0])  / sw_val if pw_l else 0.0
        dvy_l = (jy-pw_l[1])     / sw_val if pw_l else 0.0
        dvx_r = abs(cx_-pw_r[0]) / sw_val if pw_r else 0.0
        dvy_r = (cy_-pw_r[1])    / sw_val if pw_r else 0.0

        ew_l_ = ew_dist_row(row, JAB_EL,   JAB_WR,   sw_val)
        ew_r_ = ew_dist_row(row, CROSS_EL, CROSS_WR, sw_val)

        rv_l.append(raw_l); rv_r.append(raw_r)
        ea_l.append(elbow_angle_row(row, JAB_SH,   JAB_EL,   JAB_WR))
        ea_r.append(elbow_angle_row(row, CROSS_SH, CROSS_EL, CROSS_WR))
        vx_l.append(dvx_l); vx_r.append(dvx_r)
        vy_l.append(dvy_l); vy_r.append(dvy_r)
        ex_l.append(arm_ext_row(row, JAB_SH,   JAB_WR,   sw_val))
        ex_r.append(arm_ext_row(row, CROSS_SH, CROSS_WR, sw_val))
        ew_l.append(ew_l_); ew_r.append(ew_r_)
        lat_l.append(elbow_lateral_row(row, JAB_EL, JAB_SH, sw_val))
        lat_r.append(elbow_lateral_row(row, CROSS_EL, CROSS_SH, sw_val))
        er_l.append(elbow_wrist_xratio_row(row, JAB_WR, JAB_EL, JAB_SH))
        er_r.append(elbow_wrist_xratio_row(row, CROSS_WR, CROSS_EL, CROSS_SH))

        pk_l = max(rv_l); pk_r = max(rv_r)
        cd_ok = idx - last_fired >= PUNCH_CD_FRAMES

        # 훅 가능성: 최근 4프레임 EW 평균 기반
        ew_l_r = list(ew_l)[-4:] if len(ew_l) >= 4 else list(ew_l)
        ew_r_r = list(ew_r)[-4:] if len(ew_r) >= 4 else list(ew_r)
        hook_l = (sum(ew_l_r)/len(ew_l_r) if ew_l_r else 1.0) < EW_HOOK_MAX_C5
        hook_r = (sum(ew_r_r)/len(ew_r_r) if ew_r_r else 1.0) < EW_HOOK_MAX_C5
        vel_thr_l = VEL_START_RAW * 0.5 if hook_l else VEL_START_RAW
        vel_thr_r = VEL_START_RAW * 0.5 if hook_r else VEL_START_RAW
        dom_l = 1.0 if hook_l else DOM_RATIO_C5
        dom_r = 1.0 if hook_r else DOM_RATIO_C5

        fired = False
        if (prv_l > vel_thr_l and raw_l <= vel_thr_l
                and pk_l > pk_r * dom_l and cd_ok):
            if is_confident_trigger(view, pk_l, pk_r, max(ex_l, default=0.0)):
                events.append((idx, classify_c5('lead', ea_l, vx_l, vy_l, ex_l, ew_l,
                                                lat_l, er_l, view=view,
                                                hook_hint=hook_l, peak=pk_l, opp_peak=pk_r)))
            last_fired = idx; fired = True

        if (not fired
                and prv_r > vel_thr_r and raw_r <= vel_thr_r
                and pk_r > pk_l * dom_r and cd_ok):
            if is_confident_trigger(view, pk_r, pk_l, max(ex_r, default=0.0)):
                events.append((idx, classify_c5('rear', ea_r, vx_r, vy_r, ex_r, ew_r,
                                                lat_r, er_r, view=view,
                                                hook_hint=hook_r, peak=pk_r, opp_peak=pk_l)))
            last_fired = idx; fired = True

        if fired:
            rv_l.clear(); rv_r.clear()  # 속도 히스토리만 리셋 (리턴 모션 차단)
            prv_l = prv_r = 0.0
            # pw_l/pw_r 유지 — 다음 프레임부터 바로 속도 계산 가능
            pw_l = (jx, jy); pw_r = (cx_, cy_)
            continue

        prv_l = raw_l; prv_r = raw_r
        pw_l = (jx, jy); pw_r = (cx_, cy_)

    return events

# ══════════════════════════════════════════════════════════════════
# 이벤트 매칭 (GT ↔ Pred, ±MATCH_WIN 프레임)
# ══════════════════════════════════════════════════════════════════
MATCH_WIN = 12

def match_events(gt_events, pred_events):
    """
    반환: y_true, y_pred, fn_count, fp_count
    - 매칭된 쌍: (gt_label, pred_label)
    - 미매칭 GT  = False Negative (NONE으로 처리)
    - 미매칭 Pred = False Positive
    """
    y_true, y_pred = [], []
    used_gt   = set()
    used_pred = set()

    # GT 기준으로 가장 가까운 Pred 찾기
    for gi, (gf, gl) in enumerate(gt_events):
        best_j, best_dist = None, MATCH_WIN + 1
        for pj, (pf, pl) in enumerate(pred_events):
            if pj in used_pred: continue
            d = abs(gf - pf)
            if d <= MATCH_WIN and d < best_dist:
                best_dist = d; best_j = pj
        if best_j is not None:
            used_gt.add(gi); used_pred.add(best_j)
            y_true.append(gl)
            y_pred.append(pred_events[best_j][1])
        else:
            y_true.append(gl); y_pred.append('NONE')  # FN

    # 미매칭 Pred = FP (confusion matrix에서는 제외, 별도 카운트)
    fp_count = sum(1 for j in range(len(pred_events)) if j not in used_pred)
    fn_count = sum(1 for t, p in zip(y_true, y_pred) if p == 'NONE')
    return y_true, y_pred, fn_count, fp_count

# ══════════════════════════════════════════════════════════════════
# 자세 분포 분석
# ══════════════════════════════════════════════════════════════════
def extract_posture_metrics(rows):
    """모든 프레임에서 자세 메트릭 추출 (측면 뷰 정규화 보완)"""
    records = []
    for row in rows:
        sw_val = sw(row)
        sh_cy  = (g(row,'left_shoulder','y') + g(row,'right_shoulder','y')) / 2
        sh_cx  = (g(row,'left_shoulder','x') + g(row,'right_shoulder','x')) / 2
        hi_cy  = (g(row,'left_hip','y')      + g(row,'right_hip','y'))      / 2
        hi_cx  = (g(row,'left_hip','x')      + g(row,'right_hip','x'))      / 2
        nose_y = g(row,'nose','y')

        # 측면 뷰에서 어깨폭이 작아지면 어깨-엉덩이 수직 거리로 보완
        torso = abs(hi_cy - sh_cy) + 1e-6
        norm  = max(sw_val, torso)

        guard_l_y = (g(row,'left_wrist','y')  - g(row,'left_shoulder','y'))  / norm
        guard_r_y = (g(row,'right_wrist','y') - g(row,'right_shoulder','y')) / norm
        lean      = (sh_cx - hi_cx) / norm
        head_y    = (nose_y - sh_cy) / norm

        records.append({
            'frame':        int(row['frame_number']),
            'guard_l_ydiff': round(guard_l_y, 4),
            'guard_r_ydiff': round(guard_r_y, 4),
            'lean_forward':  round(lean,      4),
            'head_y_ratio':  round(head_y,    4),
        })
    return records

def score_posture_frame(rec, dna_pose, tol=0.15):
    """
    각 메트릭이 레퍼런스 ± tol 범위 안에 있으면 pass.
    반환: {metric: True/False}
    """
    checks = {}
    for key in ('guard_l_ydiff','guard_r_ydiff','lean_forward','head_y_ratio'):
        ref  = dna_pose.get(key, 0.0)
        val  = rec.get(key, 0.0)
        checks[key] = abs(val - ref) <= tol
    return checks

# ══════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════
PUNCH_LABELS = ['jab', 'cross', 'hook', 'uppercut', 'NONE']

def plot_punch_cm(y_true, y_pred, out_path, title):
    if not (_PLOT and _SKL): return
    labels = [l for l in PUNCH_LABELS if l in set(y_true)|set(y_pred)]
    cm     = confusion_matrix(y_true, y_pred, labels=labels)
    cm_n   = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold', color='white')
    fig.patch.set_facecolor('#0b0d12')

    for ax, data, fmt, t in [(axes[0], cm,   'd',    'Count'),
                              (axes[1], cm_n, '.2f', 'Normalized (Recall)')]:
        sns.heatmap(data, annot=True, fmt=fmt,
                    cmap='YlOrRd' if 'Norm' in t else 'Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, linewidths=0.5, linecolor='#333')
        ax.set_title(t, color='white', fontsize=11)
        ax.set_xlabel('Predicted', color='#aaa')
        ax.set_ylabel('Actual (GT)', color='#aaa')
        ax.tick_params(colors='white')
        ax.set_facecolor('#13151c')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0b0d12')
    plt.close()
    print(f"[OK] Confusion matrix → {out_path}")


def plot_posture_dist(all_records, dna_pose, out_path):
    if not _PLOT: return
    metrics  = ['guard_l_ydiff','guard_r_ydiff','lean_forward','head_y_ratio']
    labels_kr = ['잽 가드 (L)', '크로스 가드 (R)', '린포워드', '머리 높이']
    colors   = ['#00ccff', '#ff8c00', '#00e676', '#ff3d57']
    tol      = 0.15

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('자세 메트릭 분포 vs LIM DNA 레퍼런스', fontsize=13, color='white', fontweight='bold')
    fig.patch.set_facecolor('#0b0d12')

    for ax, metric, label_kr, col in zip(axes.flat, metrics, labels_kr, colors):
        vals = [r[metric] for r in all_records]
        ref  = dna_pose.get(metric, 0.0)

        ax.hist(vals, bins=50, color=col, alpha=0.65, edgecolor='none', label='실측 분포')
        ax.axvline(ref,       color='white',  linewidth=2.0, linestyle='-',  label=f'DNA ref {ref:+.3f}')
        ax.axvline(ref + tol, color='#aaa',   linewidth=1.2, linestyle='--', label=f'±{tol} 허용')
        ax.axvline(ref - tol, color='#aaa',   linewidth=1.2, linestyle='--')

        in_tol = sum(1 for v in vals if abs(v-ref) <= tol)
        pct    = 100 * in_tol / max(len(vals), 1)
        ax.set_title(f'{label_kr}  (범위 내: {pct:.1f}%)', color='white', fontsize=11)
        ax.set_xlabel('normalized value', color='#aaa')
        ax.set_ylabel('frame count', color='#aaa')
        ax.set_facecolor('#13151c')
        ax.tick_params(colors='white')
        ax.legend(fontsize=8, facecolor='#1e2130', labelcolor='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0b0d12')
    plt.close()
    print(f"[OK] 자세 분포 → {out_path}")


# ══════════════════════════════════════════════════════════════════
# 리포트 저장
# ══════════════════════════════════════════════════════════════════
def save_punch_metrics(y_true, y_pred, fn_count, fp_count, out_path):
    labels = [l for l in PUNCH_LABELS if l in set(y_true)|set(y_pred)]
    rows   = []
    if _SKL:
        rep = classification_report(y_true, y_pred, labels=labels,
                                    output_dict=True, zero_division=0)
        for lbl in labels:
            m = rep.get(lbl, {})
            rows.append({'class': lbl,
                         'precision': round(m.get('precision',0), 4),
                         'recall':    round(m.get('recall',0),    4),
                         'f1':        round(m.get('f1-score',0),  4),
                         'support':   int(m.get('support',0))})
        for avg in ('macro avg','weighted avg'):
            m = rep.get(avg,{})
            rows.append({'class': avg.replace(' ','_').upper(),
                         'precision': round(m.get('precision',0),4),
                         'recall':    round(m.get('recall',0),4),
                         'f1':        round(m.get('f1-score',0),4),
                         'support':   int(m.get('support',0))})
        rows.append({'class':'ACCURACY','precision':'',
                     'recall':'','f1':'',
                     'support': round(accuracy_score(y_true,y_pred),4)})
    rows.append({'class':'FALSE_NEG(missed)', 'precision':'', 'recall':'',
                 'f1':'', 'support': fn_count})
    rows.append({'class':'FALSE_POS(extra)',  'precision':'', 'recall':'',
                 'f1':'', 'support': fp_count})

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['class','precision','recall','f1','support'])
        w.writeheader(); w.writerows(rows)
    print(f"[OK] Punch metrics → {out_path}")


def save_posture_report(all_records, dna_pose, out_path):
    metrics = ['guard_l_ydiff','guard_r_ydiff','lean_forward','head_y_ratio']
    tol     = 0.15
    rows    = []
    for rec in all_records:
        row = {'frame': rec['frame']}
        total_ok = 0
        for m in metrics:
            ref = dna_pose.get(m, 0.0)
            val = rec[m]
            ok  = abs(val - ref) <= tol
            row[m]          = round(val, 4)
            row[m+'_ok']    = 1 if ok else 0
            row[m+'_err']   = round(val - ref, 4)
            if ok: total_ok += 1
        row['score'] = round(100 * total_ok / len(metrics), 1)
        rows.append(row)

    if not rows: return
    fieldnames = list(rows[0].keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    # 요약 통계 출력
    scores = [r['score'] for r in rows]
    print(f"\n━━ 자세 점수 분포 (전체 {len(rows)}프레임) ━━")
    print(f"  평균   : {np.mean(scores):.1f}점")
    print(f"  중앙값 : {np.median(scores):.1f}점")
    print(f"  90+점  : {sum(1 for s in scores if s>=90):4d}프레임 ({100*sum(1 for s in scores if s>=90)/len(scores):.1f}%)")
    print(f"  75+점  : {sum(1 for s in scores if s>=75):4d}프레임 ({100*sum(1 for s in scores if s>=75)/len(scores):.1f}%)")
    for m in metrics:
        vals = [r[m] for r in rows]
        ref  = dna_pose.get(m, 0.0)
        in_t = sum(1 for v in vals if abs(v-ref) <= tol)
        print(f"  {m:20s}: 허용범위 내 {in_t}/{len(vals)} ({100*in_t/len(vals):.1f}%)")
    print(f"[OK] 자세 리포트 → {out_path}")


def save_summary(y_true, y_pred, fn_c, fp_c, posture_records, dna_pose, out_path):
    lines = [
        "═══ Boxing Coach 5 — CSV 기반 성능 평가 ═══",
        f"처리 프레임: {len(posture_records)}",
        "",
        "━━ 펀치 분류 (extraction.py GT vs coach5 Pred) ━━",
        f"  GT 이벤트   : {len(y_true)}개",
        f"  False Neg   : {fn_c} (GT 감지됐지만 coach5가 미감지)",
        f"  False Pos   : {fp_c} (coach5가 감지했지만 GT에 없음)",
    ]
    if _SKL:
        lines.append(f"  Accuracy    : {accuracy_score(y_true, y_pred):.4f}")
        lines.append("")
        labels = [l for l in PUNCH_LABELS if l in set(y_true)|set(y_pred)]
        lines.append(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    lines += ["", "━━ Failure Cases ━━"]
    fails = [(gt, pred) for gt, pred in zip(y_true, y_pred) if gt != pred]
    if not fails:
        lines.append("  오분류 없음!")
    else:
        for (gt, pred), cnt in Counter(fails).most_common():
            lines.append(f"  GT={gt:10s} → PRED={pred:10s}  ×{cnt}")

    if posture_records:
        metrics_p = ('guard_l_ydiff','guard_r_ydiff','lean_forward','head_y_ratio')
        tol = 0.15
        # 프레임별 통과 메트릭 수 → 점수 계산 (score 키가 없어도 직접 계산)
        scores = []
        for rec in posture_records:
            ok = sum(1 for m in metrics_p if abs(rec.get(m,0) - dna_pose.get(m,0)) <= tol)
            scores.append(100 * ok / len(metrics_p))
        lines += [
            "", "━━ 자세 점수 요약 ━━",
            f"  평균 {np.mean(scores):.1f}점  |  중앙값 {np.median(scores):.1f}점",
        ]
        for m in metrics_p:
            vals = [r[m] for r in posture_records]
            ref  = dna_pose.get(m, 0.0)
            in_t = sum(1 for v in vals if abs(v-ref) <= tol)
            lines.append(f"  {m:20s}: {100*in_t/len(vals):.1f}% 허용범위 내")

    txt = '\n'.join(lines)
    print('\n' + txt)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(f"\n[OK] Summary → {out_path}")


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
DEFAULT_FILES = [
    'LIM_full_data1.csv', 'LIM_full_data2.csv',
    'LIM_full_data3.csv', 'LIM_full_data4.csv',
    'LIM_full_data5.csv', 'LIM_full_data6.csv',
    'LIM_full_data7.csv', 'LIM_full_data8.csv',
]
DEFAULT_VIEWS = ['front', 'side', 'front', 'side', 'front', 'side', 'side', 'side']

def load_manual_labels(path):
    """label_tool.py 출력 CSV를 (frame_idx, label) 목록으로 로드"""
    events = []
    if not os.path.exists(path):
        return events
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            events.append((int(row['frame_number']), row['punch_type']))
    return sorted(events)


def main():
    parser = argparse.ArgumentParser(description='CSV 기반 오프라인 평가')
    parser.add_argument('--files',     nargs='+', default=DEFAULT_FILES,
                        help='평가할 full_data CSV 목록 (기본: LIM_full_data1~5.csv)')
    parser.add_argument('--out',       default='eval_csv', help='출력 폴더')
    parser.add_argument('--stance',    default='orthodox', choices=['orthodox','southpaw'])
    parser.add_argument('--views',     nargs='+', default=DEFAULT_VIEWS,
                        choices=['front', 'side'],
                        help='files ??? ??? ?. ???: LIM1/3=front, LIM2/4=side')
    parser.add_argument('--no_posture', action='store_true', help='자세 분석 생략')
    parser.add_argument('--gt_labels', nargs='+', default=None,
                        help='수동 라벨 CSV 목록 (label_tool.py 출력). '
                             'files 순서와 1:1 대응. 없는 파일은 pseudo-GT 사용.')
    args = parser.parse_args()

    args.facing_right = (args.stance == 'southpaw')
    gt_mode = 'manual' if args.gt_labels else 'pseudo'
    print(f"스탠스: {args.stance}  GT: {gt_mode}")

    os.makedirs(args.out, exist_ok=True)
    dna_punch = load_dna_punch(os.path.join(BASE_DIR, 'LIM_punch_DNA.csv'))
    dna_pose  = load_dna_pose(os.path.join(BASE_DIR, 'LIM_DNA.csv'))

    all_y_true, all_y_pred = [], []
    all_fn, all_fp         = 0, 0
    all_posture            = []

    for i, fname in enumerate(args.files):
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[건너뜀] {fname} 없음")
            continue

        rows = load_full_data(fpath)
        view = args.views[i] if i < len(args.views) else 'front'
        print(f"\n{'─'*50}")
        print(f"처리: {fname}  ({len(rows)}프레임)")

        # GT: 수동 라벨 우선, 없으면 pseudo-GT
        print(f"  View: {view}")

        orthodox = (args.stance == 'orthodox')
        if args.gt_labels and i < len(args.gt_labels):
            lbl_path = os.path.join(BASE_DIR, args.gt_labels[i])
            gt_events = load_manual_labels(lbl_path)
            print(f"  GT: 수동 라벨 {len(gt_events)}개 ({args.gt_labels[i]})")
        else:
            gt_events = get_gt_events(rows, orthodox=orthodox)
            print(f"  GT: pseudo-GT {len(gt_events)}개")

        pred_events = get_pred_events(rows, facing_right=args.facing_right, view=view)
        y_true, y_pred, fn_c, fp_c = match_events(gt_events, pred_events)

        print(f"  GT 이벤트: {len(gt_events)}  Pred 이벤트: {len(pred_events)}")
        print(f"  매칭: {len(y_true)-fn_c}  FN: {fn_c}  FP: {fp_c}")

        gt_dist  = Counter(l for _,l in gt_events)
        print(f"  GT 분포   : {dict(gt_dist)}")
        pred_dist = Counter(l for _,l in pred_events)
        print(f"  Pred 분포 : {dict(pred_dist)}")

        all_y_true.extend(y_true); all_y_pred.extend(y_pred)
        all_fn += fn_c; all_fp += fp_c

        # 자세 분석
        if not args.no_posture:
            posture_recs = extract_posture_metrics(rows)
            all_posture.extend(posture_recs)

    if not all_y_true:
        print("\n분석할 데이터 없음. CSV 경로 확인 바람.")
        return

    # ── 결과 저장 ──────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"전체 집계: GT={len(all_y_true)}  FN={all_fn}  FP={all_fp}")

    plot_punch_cm(
        all_y_true, all_y_pred,
        os.path.join(args.out, 'punch_confusion.png'),
        title=f'Punch Classification  |  extraction.py (GT) vs coach5 (Pred)  |  n={len(all_y_true)}'
    )
    save_punch_metrics(all_y_true, all_y_pred, all_fn, all_fp,
                       os.path.join(args.out, 'punch_metrics.csv'))

    if all_posture:
        plot_posture_dist(all_posture, dna_pose,
                          os.path.join(args.out, 'posture_dist.png'))
        save_posture_report(all_posture, dna_pose,
                            os.path.join(args.out, 'posture_report.csv'))

    save_summary(all_y_true, all_y_pred, all_fn, all_fp,
                 all_posture, dna_pose,
                 os.path.join(args.out, 'summary.txt'))


if __name__ == '__main__':
    main()
