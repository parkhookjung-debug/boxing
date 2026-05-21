"""
LIM coach 5.py — 웹 UI 스타일 로컬 복싱 코치 (펀치 인식 개선)
──────────────────────────────────────────────────────────────
LIM coach web + 주먹 인식률 대폭 개선

개선 사항
  ● 임계값 완화   PUNCH_CD 0.55→0.40 / VEL_START 0.15→0.08 / DOM_RATIO 1.35→1.15
  ● 속도 분리     총 속도 외에 vx(수평) · vy_up(상향) · d_ext(팔 신전) 별도 추적
  ● 펀치 분류 개선
      - 어퍼컷: vy_up 강함 + 팔꿈치 굽음 → 기존 절대좌표 방식보다 빠르게 반응
      - 훅:     ea <112° 프레임 ≥3 + ext_delta 작음 → 신전 없이 스윙만 할 때
      - 잽/크로스: 나머지 (팔 뻗음 + ea 큼)
  ● DirectML GPU 자동 감지

단축키  Q/ESC 종료  R 초기화  D 방향전환  G 가이드  V 음성  S 스냅샷
"""

import cv2, numpy as np, math, os, csv, time, threading
from collections import deque
from PIL import ImageFont, ImageDraw, Image as PILImage

try:
    from rtmlib import RTMO
except ImportError:
    raise SystemExit("pip install rtmlib onnxruntime")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════
# 색상 팔레트 (BGR)
# ══════════════════════════════════════════════════════
C_INK     = (18,  13,  11)
C_CANVAS  = (28,  21,  19)
C_BORDER  = (40,  35,  33)
C_ACCENT  = (87,  61, 255)
C_TEXT    = (245, 244, 244)
C_MUTED   = (170, 161, 161)
C_DIM     = (122, 113, 113)
C_GREEN   = (100, 220,   0)
C_AMBER   = ( 50, 165, 255)

PUNCH_BGR = {
    'jab':      (  0, 220, 255),
    'cross':    (255, 160,  40),
    'hook':     (200,  50, 255),
    'uppercut': (100, 255,  50),
}

# ══════════════════════════════════════════════════════
# 폰트
# ══════════════════════════════════════════════════════
def _font(size):
    for p in [
        "C:/Windows/Fonts/BebasNeue-Regular.ttf",
        os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts/BebasNeue-Regular.ttf"),
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()

def _font_kr(size):
    for p in ["C:/Windows/Fonts/malgun.ttf","C:/Windows/Fonts/gulim.ttc","C:/Windows/Fonts/NanumGothic.ttf"]:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()

F_DISPLAY = _font(52);   F_TITLE = _font(32);   F_LABEL = _font(20)
F_SMALL   = _font_kr(15); F_MEDIUM = _font_kr(18); F_LARGE = _font_kr(22)

# ══════════════════════════════════════════════════════
# TTS
# ══════════════════════════════════════════════════════
_tts_ok = False; _tts_busy = False; _last_tts = 0.0; TTS_INTERVAL = 5.0
try:
    import pyttsx3 as _p3; _tts_ok = True
except ImportError:
    pass

def speak(text):
    global _tts_busy, _last_tts
    if not _tts_ok or _tts_busy or not _voice_on: return
    if time.time() - _last_tts < TTS_INTERVAL: return
    _last_tts = time.time(); _tts_busy = True
    def _run():
        global _tts_busy
        try:
            e = _p3.init()
            for v in e.getProperty('voices'):
                if any(k in str(v.id) for k in ['Ko','Korean','Heami','ko_','ko-']):
                    e.setProperty('voice', v.id); break
            e.setProperty('rate', 165); e.say(text); e.runAndWait()
        except: pass
        finally: _tts_busy = False
    threading.Thread(target=_run, daemon=True).start()

# ══════════════════════════════════════════════════════
# RTMPose (DirectML 자동 감지)
# ══════════════════════════════════════════════════════
RTMO_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
    'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip'
)
print("RTMPose (RTMO-s) 로드 중...")
try:
    import onnxruntime as ort
    _device = 'dml' if 'DmlExecutionProvider' in ort.get_available_providers() else 'cpu'
except Exception:
    _device = 'cpu'
print(f"  device: {_device}")
pose_model = RTMO(RTMO_URL, backend='onnxruntime', device=_device)
print("모델 준비 완료\n")

# ══════════════════════════════════════════════════════
# COCO 17 키포인트
# ══════════════════════════════════════════════════════
KP_NOSE=0; KP_L_SH=5; KP_R_SH=6; KP_L_EL=7; KP_R_EL=8
KP_L_WR=9; KP_R_WR=10; KP_L_HI=11; KP_R_HI=12
KP_L_KN=13; KP_R_KN=14; KP_L_AN=15; KP_R_AN=16

COCO_CONN = [
    (KP_L_SH,KP_R_SH),(KP_L_SH,KP_L_EL),(KP_L_EL,KP_L_WR),
    (KP_R_SH,KP_R_EL),(KP_R_EL,KP_R_WR),
    (KP_L_SH,KP_L_HI),(KP_R_SH,KP_R_HI),(KP_L_HI,KP_R_HI),
    (KP_L_HI,KP_L_KN),(KP_L_KN,KP_L_AN),(KP_R_HI,KP_R_KN),(KP_R_KN,KP_R_AN),
    (KP_NOSE,KP_L_SH),(KP_NOSE,KP_R_SH),
]
VIS_MIN   = 0.30
NEEDED_KP = [KP_L_SH, KP_R_SH, KP_L_WR, KP_R_WR, KP_L_EL, KP_R_EL, KP_NOSE]

_view_mode = 'side'  # 'side' or 'front'; press C to toggle
_orthodox = True  # True=orthodox(왼손 잽), False=southpaw(오른손 잽)

def get_arm_indices():
    if _orthodox:
        # orthodox: 왼손=잽(lead), 오른손=크로스(rear) — 방향과 무관
        return KP_L_WR, KP_L_SH, KP_L_EL, KP_R_WR, KP_R_SH, KP_R_EL
    else:
        # southpaw: 오른손=잽(lead), 왼손=크로스(rear)
        return KP_R_WR, KP_R_SH, KP_R_EL, KP_L_WR, KP_L_SH, KP_L_EL

# ══════════════════════════════════════════════════════
# DNA
# ══════════════════════════════════════════════════════
PUNCH_DEFAULTS = {
    'jab':      {'arm_extension_avg':1.10,'elbow_angle_avg':158,'elbow_height_avg':0.12,'lean_forward_avg':0.05},
    'cross':    {'arm_extension_avg':1.20,'elbow_angle_avg':162,'elbow_height_avg':0.08,'lean_forward_avg':0.08},
    'hook':     {'arm_extension_avg':0.80,'elbow_angle_avg': 95,'elbow_height_avg':-0.05,'lean_forward_avg':0.05},
    'uppercut': {'arm_extension_avg':0.90,'elbow_angle_avg':115,'elbow_height_avg':0.20,'lean_forward_avg':0.10},
}
POSE_DEFAULTS = {
    'guard_l_ydiff':-0.15,'guard_r_ydiff':-0.10,
    'head_y_ratio':-0.85,'lean_forward':0.05,'stance_3d_ratio':1.20,
}
TOL_PUNCH = {'arm_extension':0.12,'elbow_angle':28.0,'lean_forward':0.10}

def load_punch_dna(path):
    if not os.path.exists(path): return dict(PUNCH_DEFAULTS)
    res = dict(PUNCH_DEFAULTS)
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pt = row['punch_type']
            res[pt] = {k:float(v) for k,v in row.items() if k not in ('punch_type','count')}
    return res

def load_pose_dna(path):
    if not os.path.exists(path): return dict(POSE_DEFAULTS)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            merged = dict(POSE_DEFAULTS)
            for k,v in row.items():
                if k.endswith('_std'): continue
                try:
                    if k in merged: merged[k] = float(v)
                except ValueError: pass
            return merged
    return dict(POSE_DEFAULTS)

_dna_side = os.path.join(BASE_DIR, 'LIM_punch_DNA_side.csv')
_dna_old  = os.path.join(BASE_DIR, 'LIM_punch_DNA.csv')
_dna_src  = _dna_side if os.path.exists(_dna_side) else _dna_old
PUNCH_DNA = load_punch_dna(_dna_src)
POSE_DNA  = load_pose_dna(os.path.join(BASE_DIR, 'LIM_DNA.csv'))


# ══════════════════════════════════════════════════════
# 펀치 템플릿 (스켈레톤 1-NN 매칭)
# ══════════════════════════════════════════════════════
TEMPLATE_KP_NAMES = [
    'left_wrist',  'right_wrist',
    'left_elbow',  'right_elbow',
    'left_shoulder','right_shoulder',
    'nose',
    'left_hip',    'right_hip',
]

def load_punch_templates(path):
    """punch_templates.csv 로드. (vec, type, side, view) 리스트 반환."""
    tpls = []
    if not os.path.exists(path):
        return tpls
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            vec = []
            for name in TEMPLATE_KP_NAMES:
                vec.append(float(row[f'{name}_dx']))
                vec.append(float(row[f'{name}_dy']))
            tpls.append({
                'punch_type': row['punch_type'],
                'side':       row['side'],
                'view':       row['view'],
                'vec':        vec,
            })
    return tpls

PUNCH_TEMPLATES = load_punch_templates(os.path.join(BASE_DIR, 'punch_templates.csv'))
print(f"펀치 템플릿: {len(PUNCH_TEMPLATES)}개 로드")

# ══════════════════════════════════════════════════════
# 감지 상수 — 완화된 임계값
# ══════════════════════════════════════════════════════
PUNCH_CD       = 0.30   # 발동한 쪽 자체 쿨다운 (복귀 모션 차단)
PUNCH_CD_CROSS = 0.10   # 반대쪽 쿨다운 (콤보 허용 — 잽→크로스 가능하게 짧게)
VEL_START = 0.04   # 약한 펀치도 감지
DOM_RATIO = 1.0
DETECT_DROP_RATIO = 0.55
DETECT_MAX_ACTIVE = 0.18
DETECT_MIN_PEAK   = 0.05    # 최소 피크 (잔진동 차단)
SETTLE_RATIO      = 0.75    # v < VEL_START*0.75 = 0.03 아래로 떨어져야 재무장

# 분류 임계값 (단순 변위 기반)
UPPERCUT_DY    = 0.18   # 손목이 위로 net 0.18 sw 이상 → 어퍼컷
HOOK_SUM_VX    = 0.30   # 횡방향 누적 0.30 sw 이상 → 훅

# 분류 임계값
UP_VEL_THR  = 0.12   # 어퍼컷 상향 속도 임계 (올림: 오분류 방지)
UP_EA_THR   = 130.0  # 어퍼컷 팔꿈치 각도 상한
HOOK_EA_THR  = 112.0  # (미사용, 하위호환)
HOOK_FRAMES  = 3
HOOK_EXT_MAX = 0.38
EW_HOOK_MAX  = 0.22   # min_ew 방식 기준 — 잽/크로스 오분류 방지

# View-aware punch classification thresholds from CSV evaluation.
FRONT_UP_VEL      = 0.20
FRONT_UP_RATIO    = 0.75
FRONT_UP_LAT_MAX  = 0.55
FRONT_HOOK_LAT    = 0.46
FRONT_HOOK_LAT_EW = 0.38
FRONT_HOOK_VX     = 0.30

SIDE_UP_VEL       = 0.12
SIDE_UP_ERATIO    = 0.82
SIDE_UP_RATIO     = 0.45
SIDE_HOOK_ER_MIN  = 0.62
SIDE_HOOK_ER_MAX  = 0.82
SIDE_HOOK_EW4     = 2.0
SIDE_HOOK_VX      = 0.35
SIDE_HOOK_EW_MIN  = 0.20

# ══════════════════════════════════════════════════════
# 상태
# ══════════════════════════════════════════════════════
_count       = {t:0 for t in PUNCH_BGR}
_trail_lead  = deque(maxlen=22)
_trail_rear  = deque(maxlen=22)
_last_pt     = {'lead':'jab','rear':'cross'}
_lm_buf      = deque(maxlen=30)
_pending     = []
EXT_DELAY    = 5

_BUF_SZ      = 12
# 기존 버퍼
_v_buf_lead  = deque(maxlen=_BUF_SZ); _v_buf_rear  = deque(maxlen=_BUF_SZ)
_el_buf_lead = deque(maxlen=_BUF_SZ); _el_buf_rear = deque(maxlen=_BUF_SZ)
_ea_buf_lead = deque(maxlen=_BUF_SZ); _ea_buf_rear = deque(maxlen=_BUF_SZ)
_wy_buf_lead = deque(maxlen=_BUF_SZ); _wy_buf_rear = deque(maxlen=_BUF_SZ)
# 신규 버퍼 — 방향별 속도 + 팔 신전
_vx_buf_lead  = deque(maxlen=_BUF_SZ); _vx_buf_rear  = deque(maxlen=_BUF_SZ)
_vy_buf_lead  = deque(maxlen=_BUF_SZ); _vy_buf_rear  = deque(maxlen=_BUF_SZ)
_ext_buf_lead = deque(maxlen=_BUF_SZ); _ext_buf_rear = deque(maxlen=_BUF_SZ)
_ew_buf_lead  = deque(maxlen=_BUF_SZ); _ew_buf_rear  = deque(maxlen=_BUF_SZ)  # elbow-wrist dist / sw
_lat_buf_lead = deque(maxlen=_BUF_SZ); _lat_buf_rear = deque(maxlen=_BUF_SZ)
_er_buf_lead  = deque(maxlen=_BUF_SZ); _er_buf_rear  = deque(maxlen=_BUF_SZ)

_prev_v_lead  = 0.0; _prev_v_rear  = 0.0
_punch_cd     = {'lead':0.0,'rear':0.0}
_prev_lead_wr = None; _prev_rear_wr = None
_det_lead     = {'active': False, 'start': 0.0, 'peak': 0.0, 'opp_peak': 0.0}
_det_rear     = {'active': False, 'start': 0.0, 'peak': 0.0, 'opp_peak': 0.0}
_settled      = {'lead': True, 'rear': True}  # 발동 후 손목이 가라앉아야 True

_report       = None; REPORT_DUR = 4.0
_posture_data = None
_flash        = None
_show_ghost   = True
_voice_on     = True
_snap_count   = 0
_pose_stable  = 0
_pose_ready   = False
READY_FRAMES  = 20

# 사이드바 디버그 표시용 최근 신호
_dbg_signal   = {'lead': '', 'rear': ''}

# ══════════════════════════════════════════════════════
# 기하 헬퍼
# ══════════════════════════════════════════════════════
def sw_px(kp):
    dx=kp[KP_R_SH][0]-kp[KP_L_SH][0]; dy=kp[KP_R_SH][1]-kp[KP_L_SH][1]
    return math.sqrt(dx*dx+dy*dy)+1e-6

def angle3pt(ax,ay,bx,by,cx,cy):
    bax,bay=ax-bx,ay-by; bcx,bcy=cx-bx,cy-by
    dot=bax*bcx+bay*bcy
    mag=math.sqrt(bax**2+bay**2)*math.sqrt(bcx**2+bcy**2)+1e-9
    return math.degrees(math.acos(max(-1,min(1,dot/mag))))

def arm_ext(kp, sh_i, wr_i, sw):
    return math.hypot(kp[wr_i][0]-kp[sh_i][0], kp[wr_i][1]-kp[sh_i][1]) / sw

def elbow_lat(kp, el_i, sh_i, sw):
    return abs(kp[el_i][0] - kp[sh_i][0]) / sw

def elbow_wrist_xratio(kp, wr_i, el_i, sh_i):
    wr_dx = abs(kp[wr_i][0] - kp[sh_i][0])
    el_dx = abs(kp[el_i][0] - kp[sh_i][0])
    return el_dx / (wr_dx + 1e-6)


def sw_robust_kp(kp):
    """측면 카메라용 sw 보완 — 어깨가 겹치면 토르소(어깨↔엉덩이)로 대체."""
    sw_v = sw_px(kp)
    if kp[KP_L_HI][1] > 0 and kp[KP_R_HI][1] > 0:
        sh_y = (kp[KP_L_SH][1] + kp[KP_R_SH][1]) / 2
        hi_y = (kp[KP_L_HI][1] + kp[KP_R_HI][1]) / 2
        torso = abs(hi_y - sh_y) + 1e-6
        return max(sw_v, torso)
    return sw_v


_KP_TEMPLATE_INDICES = [
    (KP_L_WR, KP_R_WR),
    (KP_L_EL, KP_R_EL),
    (KP_L_SH, KP_R_SH),
    (KP_NOSE,),
    (KP_L_HI, KP_R_HI),
]
# 평탄화해서 9개 인덱스 (TEMPLATE_KP_NAMES 순서와 일치)
_KP_FLAT = [KP_L_WR, KP_R_WR, KP_L_EL, KP_R_EL,
            KP_L_SH, KP_R_SH, KP_NOSE, KP_L_HI, KP_R_HI]


def current_pose_vector(kp, sw_v):
    """현재 kp에서 어깨 중심 기준 정규화 벡터 (18차원)."""
    sh_cx = (kp[KP_L_SH][0] + kp[KP_R_SH][0]) / 2
    sh_cy = (kp[KP_L_SH][1] + kp[KP_R_SH][1]) / 2
    vec = []
    for idx in _KP_FLAT:
        vec.append((kp[idx][0] - sh_cx) / sw_v)
        vec.append((kp[idx][1] - sh_cy) / sw_v)
    return vec


def match_pose_template(kp, sw_v, view, side):
    """현재 포즈를 punch_templates 와 1-NN 매칭.
    반환: (best_punch_type, best_distance, similarity_percent) 또는 (None, inf, 0)
    """
    if not PUNCH_TEMPLATES:
        return None, float('inf'), 0.0

    cur_vec = current_pose_vector(kp, sw_v)

    best_type = None
    best_dist = float('inf')
    # 1차: 같은 view + 같은 side
    for t in PUNCH_TEMPLATES:
        if t['side'] != side: continue
        if t['view'] != view: continue
        d = 0.0
        tv = t['vec']
        for i in range(len(cur_vec)):
            diff = cur_vec[i] - tv[i]
            d += diff * diff
        if d < best_dist:
            best_dist = d
            best_type = t['punch_type']

    # 매칭 실패 시 view 무시하고 같은 side 만 검색
    if best_type is None:
        for t in PUNCH_TEMPLATES:
            if t['side'] != side: continue
            d = 0.0
            tv = t['vec']
            for i in range(len(cur_vec)):
                diff = cur_vec[i] - tv[i]
                d += diff * diff
            if d < best_dist:
                best_dist = d
                best_type = t['punch_type']

    best_dist = math.sqrt(best_dist)
    # similarity %: 거리 0 → 100%, 거리 ~2 → 0%
    similarity = max(0.0, min(100.0, math.exp(-best_dist * 0.8) * 100))
    return best_type, best_dist, similarity

# ══════════════════════════════════════════════════════
# 펀치 분류 — 단순 X/Y 변위 기반 4-way classifier
# ══════════════════════════════════════════════════════
def classify_from_buf(side, sw, ea_buf, el_buf, wy_buf, vx_buf, vy_buf, ext_buf, ew_buf, lat_buf, er_buf,
                      peak=0.0, opp_peak=0.0, kp=None):
    """
    1순위: 스켈레톤 1-NN 매칭 (punch_templates.csv)
    2순위: 단순 X/Y 변위 기반 규칙 (템플릿 없거나 매칭 실패 시)
    """
    global _dbg_signal

    # ── 1순위: 템플릿 매칭 ──────────────────────────────────────────
    if kp is not None and PUNCH_TEMPLATES:
        sw_v = sw_robust_kp(kp) if _view_mode == 'side' else sw_px(kp)
        best_type, best_dist, similarity = match_pose_template(
            kp, sw_v, view=_view_mode, side=side
        )
        if best_type is not None:
            _dbg_signal[side] = (f"{_view_mode} {side} → {best_type} "
                                 f"sim={similarity:.0f}% d={best_dist:.2f}")
            return best_type

    # ── 2순위: 변위 기반 fallback ───────────────────────────────────
    net_vy = sum(vy_buf)   # 음수면 손목이 위로 net 이동
    sum_vx = sum(vx_buf)   # 횡방향 누적 이동거리

    if net_vy < -UPPERCUT_DY:
        return 'uppercut'
    if sum_vx > HOOK_SUM_VX:
        return 'hook'
    return 'jab' if side == 'lead' else 'cross'


def heuristic_hook_uppercut(ew_buf, wy_buf, ea_buf, sw):
    """
    훅/어퍼컷 휴리스틱 — 보수적으로 (잽/크로스 오인 방지).

    훅:     손목-팔꿈치(EW) 매우 가까움 + 팔꿈치 강하게 굽음
    어퍼컷: 손목이 시작점에서 큰 폭으로 위로 + 팔꿈치 굽음 + dip-rise 패턴

    반환: 'hook' / 'uppercut' / None
    """
    if len(ea_buf) < 3 or len(wy_buf) < 3:
        return None
    min_ew = min(ew_buf, default=1.0)
    min_ea = min(ea_buf, default=180.0)
    wy_list = list(wy_buf)

    wy_start = wy_list[0]
    wy_top   = min(wy_list)
    wy_bot   = max(wy_list)
    rise = (wy_start - wy_top) / (sw + 1e-6)
    dip  = (wy_bot - wy_start) / (sw + 1e-6)

    # 어퍼컷: rise 크고 + 팔꿈치 매우 굽음 + dip(내려갔다 올라옴) 패턴
    if rise > 0.22 and min_ea < 120.0 and dip > 0.04:
        return 'uppercut'

    # 훅: EW 매우 가까움 + 팔꿈치 강하게 굽음
    if min_ew < 0.22 and min_ea < 115.0:
        return 'hook'

    return None


def is_extended_punch(kp, sc, side):
    """확장 시점에 팔이 거의 펴졌는지 — 잽/크로스 강제 판별용.
    팔꿈치 각도 > 155° 이면 일자로 뻗은 펀치로 간주."""
    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()
    if side == 'lead':
        wr_i, sh_i, el_i = JAB_WR, JAB_SH, JAB_EL
    else:
        wr_i, sh_i, el_i = CROSS_WR, CROSS_SH, CROSS_EL
    if sc[el_i] < VIS_MIN:
        return False
    ea = angle3pt(kp[sh_i][0], kp[sh_i][1],
                  kp[el_i][0], kp[el_i][1],
                  kp[wr_i][0], kp[wr_i][1])
    return ea > 155.0


# ══════════════════════════════════════════════════════
# (참고용 보관) 이전 의사결정 트리 — min_samples_leaf=1 오버핏 가능성
# ══════════════════════════════════════════════════════
def _classify_legacy_tree(side, sw, ea_buf, el_buf, wy_buf, vx_buf, vy_buf, ext_buf, ew_buf, lat_buf, er_buf,
                          peak=0.0, opp_peak=0.0):
    min_ea    = min(ea_buf,  default=180.0)
    max_vx    = max(vx_buf,  default=0.0)
    max_vy_up = max((-v for v in vy_buf), default=0.0)
    min_ew    = min(ew_buf, default=1.0)
    ew_recent = list(ew_buf)[-4:] if len(ew_buf) >= 4 else list(ew_buf)
    avg_ew4   = sum(ew_recent) / len(ew_recent) if ew_recent else 1.0
    max_lat   = max(lat_buf, default=0.0)
    max_er    = max(er_buf, default=0.0)
    max_ext   = max(ext_buf, default=0.0)
    side_code = 0 if side == 'lead' else 1
    view_code = 0 if _view_mode == 'front' else 1
    dom       = peak / (opp_peak + 1e-9)

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

def update_punch_detect(kp, sc, now):
    global _prev_v_lead, _prev_v_rear, _prev_lead_wr, _prev_rear_wr

    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()
    if sc[JAB_WR] < VIS_MIN or sc[CROSS_WR] < VIS_MIN:
        return

    jx, jy = kp[JAB_WR][0], kp[JAB_WR][1]
    cx_, cy_ = kp[CROSS_WR][0], kp[CROSS_WR][1]
    _trail_lead.append((jx, jy))
    _trail_rear.append((cx_, cy_))

    sw = sw_px(kp)
    v_lead = math.hypot(jx - _prev_lead_wr[0], jy - _prev_lead_wr[1]) / sw if _prev_lead_wr else 0.0
    v_rear = math.hypot(cx_ - _prev_rear_wr[0], cy_ - _prev_rear_wr[1]) / sw if _prev_rear_wr else 0.0
    vx_l = abs(jx - _prev_lead_wr[0]) / sw if _prev_lead_wr else 0.0
    vy_l = (jy - _prev_lead_wr[1]) / sw if _prev_lead_wr else 0.0
    vx_r = abs(cx_ - _prev_rear_wr[0]) / sw if _prev_rear_wr else 0.0
    vy_r = (cy_ - _prev_rear_wr[1]) / sw if _prev_rear_wr else 0.0

    sh_y = (kp[KP_L_SH][1] + kp[KP_R_SH][1]) / 2
    jel_vis = sc[JAB_EL]   > VIS_MIN
    rel_vis = sc[CROSS_EL] > VIS_MIN
    jea  = angle3pt(kp[JAB_SH][0],   kp[JAB_SH][1],   kp[JAB_EL][0],   kp[JAB_EL][1],   kp[JAB_WR][0],   kp[JAB_WR][1])   if jel_vis else 180.0
    rea  = angle3pt(kp[CROSS_SH][0], kp[CROSS_SH][1], kp[CROSS_EL][0], kp[CROSS_EL][1], kp[CROSS_WR][0], kp[CROSS_WR][1]) if rel_vis else 180.0
    jel  = (kp[JAB_EL][1]   - sh_y) / sw if jel_vis else 1.0
    rel  = (kp[CROSS_EL][1] - sh_y) / sw if rel_vis else 1.0
    ext_l = arm_ext(kp, JAB_SH,   JAB_WR,   sw)
    ext_r = arm_ext(kp, CROSS_SH, CROSS_WR, sw)
    ew_l  = math.hypot(kp[JAB_EL][0]   - kp[JAB_WR][0],   kp[JAB_EL][1]   - kp[JAB_WR][1])   / sw if jel_vis else 1.0
    ew_r  = math.hypot(kp[CROSS_EL][0] - kp[CROSS_WR][0], kp[CROSS_EL][1] - kp[CROSS_WR][1]) / sw if rel_vis else 1.0
    lat_l = elbow_lat(kp, JAB_EL,   JAB_SH,   sw) if jel_vis else 0.0
    lat_r = elbow_lat(kp, CROSS_EL, CROSS_SH, sw) if rel_vis else 0.0
    er_l  = elbow_wrist_xratio(kp, JAB_WR,   JAB_EL,   JAB_SH)   if jel_vis else 0.5
    er_r  = elbow_wrist_xratio(kp, CROSS_WR, CROSS_EL, CROSS_SH) if rel_vis else 0.5

    _v_buf_lead.append(v_lead);   _v_buf_rear.append(v_rear)
    _el_buf_lead.append(jel);     _el_buf_rear.append(rel)
    _ea_buf_lead.append(jea);     _ea_buf_rear.append(rea)
    _wy_buf_lead.append(jy);      _wy_buf_rear.append(cy_)
    _vx_buf_lead.append(vx_l);    _vx_buf_rear.append(vx_r)
    _vy_buf_lead.append(vy_l);    _vy_buf_rear.append(vy_r)
    _ext_buf_lead.append(ext_l);  _ext_buf_rear.append(ext_r)
    _ew_buf_lead.append(ew_l);    _ew_buf_rear.append(ew_r)
    _lat_buf_lead.append(lat_l);  _lat_buf_rear.append(lat_r)
    _er_buf_lead.append(er_l);    _er_buf_rear.append(er_r)

    peak_lead = max(_v_buf_lead) if _v_buf_lead else 0.0
    peak_rear = max(_v_buf_rear) if _v_buf_rear else 0.0
    ew_l_r = list(_ew_buf_lead)[-4:] if len(_ew_buf_lead) >= 4 else list(_ew_buf_lead)
    ew_r_r = list(_ew_buf_rear)[-4:] if len(_ew_buf_rear) >= 4 else list(_ew_buf_rear)
    hook_possible_lead = (sum(ew_l_r) / len(ew_l_r) if ew_l_r else 1.0) < EW_HOOK_MAX
    hook_possible_rear = (sum(ew_r_r) / len(ew_r_r) if ew_r_r else 1.0) < EW_HOOK_MAX
    vel_thr_lead = VEL_START * 0.5 if hook_possible_lead else VEL_START
    vel_thr_rear = VEL_START * 0.5 if hook_possible_rear else VEL_START
    dom_thr_lead = 0.85 if hook_possible_lead else DOM_RATIO
    dom_thr_rear = 0.85 if hook_possible_rear else DOM_RATIO

    # side별 쿨다운 — 콤보(원투) 허용
    cd_ok_lead = (now - _punch_cd['lead'] > PUNCH_CD)
    cd_ok_rear = (now - _punch_cd['rear'] > PUNCH_CD)

    # 손목이 충분히 가라앉으면 settled = True (재무장 허용)
    settle_thr = VEL_START * SETTLE_RATIO
    if v_lead < settle_thr: _settled['lead'] = True
    if v_rear < settle_thr: _settled['rear'] = True

    def _update_state(state, speed, peak, opp_peak, vel_thr, settled, cd_ok):
        # 자기 쪽 쿨다운 OK + 손목이 한번 가라앉음 = 새 상태 누적 허용
        if speed > vel_thr and cd_ok and settled:
            if not state['active']:
                state['active'] = True
                state['start'] = now
                state['peak'] = peak
                state['opp_peak'] = opp_peak
            else:
                state['peak'] = max(state['peak'], peak)
                state['opp_peak'] = max(state['opp_peak'], opp_peak)

    def _candidate(state, speed, prev_speed, vel_thr, dom_thr):
        if not state['active']:
            return False
        if state['peak'] < max(DETECT_MIN_PEAK, vel_thr * 1.15):
            return False
        if state['peak'] < state['opp_peak'] * dom_thr:
            return False
        falling_edge = prev_speed > vel_thr and speed <= vel_thr
        peak_drop = speed <= max(vel_thr * 0.75, state['peak'] * DETECT_DROP_RATIO)
        timeout = now - state['start'] >= DETECT_MAX_ACTIVE
        return falling_edge or peak_drop or timeout

    def _reset_feature_buffers():
        _v_buf_lead.clear(); _v_buf_rear.clear()
        _el_buf_lead.clear(); _el_buf_rear.clear()
        _ea_buf_lead.clear(); _ea_buf_rear.clear()
        _wy_buf_lead.clear(); _wy_buf_rear.clear()
        _vx_buf_lead.clear(); _vx_buf_rear.clear()
        _vy_buf_lead.clear(); _vy_buf_rear.clear()
        _ext_buf_lead.clear(); _ext_buf_rear.clear()
        _ew_buf_lead.clear(); _ew_buf_rear.clear()
        _lat_buf_lead.clear(); _lat_buf_rear.clear()
        _er_buf_lead.clear(); _er_buf_rear.clear()

    _update_state(_det_lead, v_lead, peak_lead, peak_rear, vel_thr_lead, _settled['lead'], cd_ok_lead)
    _update_state(_det_rear, v_rear, peak_rear, peak_lead, vel_thr_rear, _settled['rear'], cd_ok_rear)

    lead_ok = cd_ok_lead and _candidate(_det_lead, v_lead, _prev_v_lead, vel_thr_lead, dom_thr_lead)
    rear_ok = cd_ok_rear and _candidate(_det_rear, v_rear, _prev_v_rear, vel_thr_rear, dom_thr_rear)

    fired_side = None
    choose_rear = rear_ok and (not lead_ok or _det_rear['peak'] > _det_lead['peak'])
    if lead_ok and not choose_rear:
        # 트리거 시점의 휴리스틱 계산 (훅/어퍼컷 보조 — EW + dip/rise 패턴)
        heur = heuristic_hook_uppercut(_ew_buf_lead, _wy_buf_lead, _ea_buf_lead, sw)
        _last_pt['lead'] = 'jab'  # 임시값
        _punch_cd['lead'] = now
        _punch_cd['rear'] = max(_punch_cd['rear'], now - PUNCH_CD + PUNCH_CD_CROSS)
        _pending.append([EXT_DELAY, None, 'lead', heur])
        fired_side = 'lead'
    elif choose_rear:
        heur = heuristic_hook_uppercut(_ew_buf_rear, _wy_buf_rear, _ea_buf_rear, sw)
        _last_pt['rear'] = 'cross'
        _punch_cd['rear'] = now
        _punch_cd['lead'] = max(_punch_cd['lead'], now - PUNCH_CD + PUNCH_CD_CROSS)
        _pending.append([EXT_DELAY, None, 'rear', heur])
        fired_side = 'rear'

    if fired_side is not None:
        _reset_feature_buffers()
        _det_lead.update({'active': False, 'start': 0.0, 'peak': 0.0, 'opp_peak': 0.0})
        _det_rear.update({'active': False, 'start': 0.0, 'peak': 0.0, 'opp_peak': 0.0})
        # 발동한 쪽만 settled=False (반대쪽은 콤보 위해 그대로 유지)
        _settled[fired_side] = False
        _prev_v_lead = 0.0
        _prev_v_rear = 0.0
        _prev_lead_wr = (jx, jy)
        _prev_rear_wr = (cx_, cy_)
        return

    # If a candidate died out without firing, release it so the next punch can arm quickly.
    if _det_lead['active'] and v_lead <= vel_thr_lead * 0.5 and now - _det_lead['start'] > DETECT_MAX_ACTIVE:
        _det_lead.update({'active': False, 'start': 0.0, 'peak': 0.0, 'opp_peak': 0.0})
    if _det_rear['active'] and v_rear <= vel_thr_rear * 0.5 and now - _det_rear['start'] > DETECT_MAX_ACTIVE:
        _det_rear.update({'active': False, 'start': 0.0, 'peak': 0.0, 'opp_peak': 0.0})

    _prev_v_lead = v_lead
    _prev_v_rear = v_rear
    _prev_lead_wr = (jx, jy)
    _prev_rear_wr = (cx_, cy_)

def _fmt_signal(v_total, vx_buf, vy_buf, ew_buf, lat_buf, er_buf):
    """Sidebar debug signal summary."""
    max_vx = max(vx_buf, default=0.0)
    max_vy_u = max((-v for v in vy_buf), default=0.0)
    min_ew = min(ew_buf, default=1.0)
    max_lat = max(lat_buf, default=0.0)
    max_er = max(er_buf, default=0.0)
    return f"{_view_mode} v{v_total:.2f} x{max_vx:.2f} u{max_vy_u:.2f} ew{min_ew:.2f} lat{max_lat:.2f} er{max_er:.2f}"

# ?? ?? (DNA ??)
# ?????????????????????????????????????????????????????
def analyse_punch(punch_type, side):
    global _report
    buf = list(_lm_buf)
    if len(buf) < 3 or punch_type not in PUNCH_DNA: return
    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()
    wr_i = JAB_WR if side=='lead' else CROSS_WR
    sh_i = JAB_SH if side=='lead' else CROSS_SH
    el_i = JAB_EL if side=='lead' else CROSS_EL
    best_kp, best_d = buf[-1][0], 0.0
    for kp_b, sc_b in buf:
        if sc_b[wr_i] < VIS_MIN or sc_b[sh_i] < VIS_MIN: continue
        d = math.hypot(kp_b[wr_i][0]-kp_b[sh_i][0], kp_b[wr_i][1]-kp_b[sh_i][1])
        if d > best_d: best_d = d; best_kp = kp_b
    kp = best_kp; sw = sw_px(kp)
    sh_cx = (kp[KP_L_SH][0]+kp[KP_R_SH][0])/2
    hi_ok = kp[KP_L_HI][0]>0 and kp[KP_R_HI][0]>0
    lean  = (sh_cx-(kp[KP_L_HI][0]+kp[KP_R_HI][0])/2)/sw if hi_ok else 0.0
    _report = {
        'type':    punch_type,
        'arm_ext': math.hypot(kp[wr_i][0]-kp[sh_i][0], kp[wr_i][1]-kp[sh_i][1]) / sw,
        'el_ang':  angle3pt(kp[sh_i][0],kp[sh_i][1],kp[el_i][0],kp[el_i][1],kp[wr_i][0],kp[wr_i][1]),
        'lean':    lean, 'time': time.time(),
    }
    _count[punch_type] += 1

# ══════════════════════════════════════════════════════
# 자세 점수
# ══════════════════════════════════════════════════════
def calc_posture(kp, sc, sw):
    # 측면 뷰에서 어깨폭(sw)이 작아지면 어깨-엉덩이 수직 거리로 보완
    if sc[KP_L_HI] > VIS_MIN and sc[KP_R_HI] > VIS_MIN:
        sh_y_ = (kp[KP_L_SH][1] + kp[KP_R_SH][1]) / 2
        hi_y_ = (kp[KP_L_HI][1] + kp[KP_R_HI][1]) / 2
        torso  = abs(hi_y_ - sh_y_) + 1e-6
        sw = max(sw, torso)
    issues = []
    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()
    sh_y   = (kp[KP_L_SH][1]+kp[KP_R_SH][1])/2
    l_ydiff = (kp[JAB_WR][1]   - kp[JAB_SH][1])  / sw
    r_ydiff = (kp[CROSS_WR][1] - kp[CROSS_SH][1]) / sw
    ref_l = POSE_DNA['guard_l_ydiff']; ref_r = POSE_DNA['guard_r_ydiff']
    tol_g = 0.25

    def _partial(err, mx):
        if abs(err) <= tol_g: return mx
        return max(0, int(mx * max(0.0, 1.0-(abs(err)-tol_g)/tol_g)))

    l_err = l_ydiff - ref_l; r_err = r_ydiff - ref_r
    l_score = _partial(l_err, 18); r_score = _partial(r_err, 17)
    guard_score = l_score + r_score
    if l_score < 18: issues.append((0,'잽 가드 올려' if l_err>0 else '잽 가드 내려', 18-l_score))
    if r_score < 17: issues.append((0,'크로스 가드 올려' if r_err>0 else '크로스 가드 내려', 17-r_score))

    hi_ok = sc[KP_L_HI]>VIS_MIN and sc[KP_R_HI]>VIS_MIN
    lean_score = 0; lean_msg = '엉덩이 미감지'
    if hi_ok:
        sh_cx  = (kp[KP_L_SH][0]+kp[KP_R_SH][0])/2
        hi_cx  = (kp[KP_L_HI][0]+kp[KP_R_HI][0])/2
        lean   = (sh_cx - hi_cx) / sw
        l_err2 = lean - POSE_DNA.get('lean_forward', 0.05)
        tol_l  = 0.12
        def _lp(err, mx):
            if abs(err) <= tol_l: return mx
            return max(0, int(mx * max(0.0, 1.0-(abs(err)-tol_l)/tol_l)))
        lean_score = _lp(l_err2, 25)
        lean_msg   = (f'기울기 좋아! ({lean:+.2f})' if lean_score == 25
                      else ('앞으로 기울여' if l_err2<0 else '상체 세워'))
        if lean_score < 25: issues.append((1, lean_msg, 25-lean_score))

    head_y    = (kp[KP_NOSE][1]-sh_y)/sw
    head_err  = head_y - POSE_DNA['head_y_ratio']
    head_score = 20 if abs(head_err)<0.18 else 10 if abs(head_err)<0.30 else 0
    head_msg   = ('머리 좋아!' if abs(head_err)<0.18
                  else ('고개 들어!' if head_err>0 else '턱 당겨!'))
    if abs(head_err) >= 0.18: issues.append((2, head_msg, 20-head_score))

    l_el_y = (kp[JAB_EL][1]  -kp[JAB_SH][1])  /sw
    r_el_y = (kp[CROSS_EL][1]-kp[CROSS_SH][1])/sw
    el_ok_l = (l_el_y > l_ydiff) and (l_el_y < 0.60)
    el_ok_r = (r_el_y > r_ydiff) and (r_el_y < 0.60)
    elbow_score = (10 if el_ok_l else 0) + (10 if el_ok_r else 0)
    el_msg = ('팔꿈치 좋아!' if el_ok_l and el_ok_r
              else ('잽 팔꿈치 조정' if not el_ok_l else '크로스 팔꿈치 조정'))

    total = guard_score + lean_score + head_score + elbow_score
    grade = 'S' if total>=85 else 'A' if total>=70 else 'B' if total>=50 else 'C'
    items = [
        ('가드',     guard_score,  35, '올려!' if guard_score<35 else '좋아!'),
        ('린포워드', lean_score,   25, lean_msg),
        ('머리',     head_score,   20, head_msg),
        ('팔꿈치',   elbow_score,  20, el_msg),
    ]
    if issues: speak(min(issues, key=lambda x:x[0])[1])
    else:       speak('자세가 좋아요')
    return total, grade, items

# ══════════════════════════════════════════════════════
# PIL / 카드 헬퍼
# ══════════════════════════════════════════════════════
def bgr2pil(img): return PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def pil2bgr(pil): return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
def draw_text_pil(img, text, pos, font, color_rgb):
    pil = bgr2pil(img); ImageDraw.Draw(pil).text(pos, text, font=font, fill=color_rgb); img[:] = pil2bgr(pil)
def rgb(bgr): return (bgr[2], bgr[1], bgr[0])

def draw_card(img, x, y, w, h, color=C_CANVAS, border=C_BORDER, radius=12):
    ov = img.copy()
    cv2.rectangle(ov, (x+radius,y), (x+w-radius,y+h), color, -1)
    cv2.rectangle(ov, (x,y+radius), (x+w,y+h-radius), color, -1)
    for cx2,cy2 in [(x+radius,y+radius),(x+w-radius,y+radius),(x+radius,y+h-radius),(x+w-radius,y+h-radius)]:
        cv2.circle(ov, (cx2,cy2), radius, color, -1)
    cv2.addWeighted(ov, 1.0, img, 0.0, 0, img)
    cv2.rectangle(img, (x+radius,y),   (x+w-radius,y),   border, 1)
    cv2.rectangle(img, (x+radius,y+h), (x+w-radius,y+h), border, 1)
    cv2.rectangle(img, (x,y+radius),   (x,y+h-radius),   border, 1)
    cv2.rectangle(img, (x+w,y+radius), (x+w,y+h-radius), border, 1)

def draw_bar(img, x, y, w, h, ratio, color=C_ACCENT, bg=C_BORDER):
    cv2.rectangle(img, (x,y), (x+w,y+h), bg, -1)
    fw = max(0, int(w * min(ratio, 1.0)))
    if fw > 0: cv2.rectangle(img, (x,y), (x+fw,y+h), color, -1)
    r = h//2
    cv2.circle(img, (x+r,y+r), r, bg, -1)
    if fw > r: cv2.circle(img, (x+fw-r,y+r), r, color, -1)

# ══════════════════════════════════════════════════════
# 레이아웃 상수
# ══════════════════════════════════════════════════════
WIN_W, WIN_H = 1280, 720
CAM_W = 853
SB_X  = CAM_W + 1
SB_W  = WIN_W - SB_X
PAD   = 14

# ══════════════════════════════════════════════════════
# 스켈레톤 / 트레일 / 고스트
# ══════════════════════════════════════════════════════
def draw_skeleton(frame, kp, sc):
    for a,b in COCO_CONN:
        if sc[a]>VIS_MIN and sc[b]>VIS_MIN:
            cv2.line(frame,(int(kp[a][0]),int(kp[a][1])),(int(kp[b][0]),int(kp[b][1])),(60,60,80),2)
    for i in range(17):
        if sc[i]>VIS_MIN:
            cv2.circle(frame,(int(kp[i][0]),int(kp[i][1])),4,(87,61,255),-1)

def draw_trail(cam):
    for trail, side in [(_trail_lead,'lead'),(_trail_rear,'rear')]:
        col = PUNCH_BGR.get(_last_pt[side],(0,200,200))
        pts = list(trail); n = len(pts)
        if n < 2: continue
        for i in range(1, n):
            a = i/n; c = tuple(int(x*a) for x in col)
            cv2.line(cam,(int(pts[i-1][0]),int(pts[i-1][1])),(int(pts[i][0]),int(pts[i][1])),c,max(1,int(4*a)))
        cv2.circle(cam,(int(pts[-1][0]),int(pts[-1][1])),7,col,-1)

def draw_ghost(cam, kp, sw):
    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()
    targets = [
        (kp[JAB_SH][0],   kp[JAB_SH][1]+POSE_DNA['guard_l_ydiff']*sw, PUNCH_BGR['jab']),
        (kp[CROSS_SH][0], kp[CROSS_SH][1]+POSE_DNA['guard_r_ydiff']*sw, PUNCH_BGR['cross']),
    ]
    ov = cam.copy()
    for ix,iy,col in targets:
        cx2,cy2 = int(ix),int(iy)
        cv2.circle(ov,(cx2,cy2),22,col,2); cv2.circle(ov,(cx2,cy2),5,col,-1)
        cv2.line(ov,(cx2-13,cy2),(cx2+13,cy2),col,1); cv2.line(ov,(cx2,cy2-13),(cx2,cy2+13),col,1)
    cv2.addWeighted(ov,0.4,cam,0.6,0,cam)

# ══════════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════════
def render_sidebar(canvas, posture, now):
    x0 = SB_X + PAD
    cw = SB_W - PAD*2

    # 헤더
    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((x0, 12), "BOXING", font=F_DISPLAY, fill=rgb(C_TEXT))
    bbox = d.textbbox((0,0), "BOXING", font=F_DISPLAY)
    bx = x0 + (bbox[2]-bbox[0]) + 4
    d.text((bx, 12), ".", font=F_DISPLAY, fill=rgb(C_ACCENT))
    bx2 = bx + d.textlength(".", font=F_DISPLAY) + 2
    d.text((bx2, 12), "COACH", font=F_DISPLAY, fill=rgb(C_TEXT))
    canvas[:] = pil2bgr(pil)

    cy = 70

    # 코치 정보 카드
    draw_card(canvas, SB_X+PAD, cy, cw, 52)
    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((x0+10, cy+8),  "LIM KWANWOO", font=F_TITLE, fill=rgb(C_TEXT))
    d.text((x0+10, cy+34), f"{'Orthodox (왼손 잽)' if _orthodox else 'Southpaw (오른손 잽)'}", font=F_SMALL, fill=rgb(C_ACCENT))
    canvas[:] = pil2bgr(pil)
    cy += 60

    # 펀치 카운트 카드
    draw_card(canvas, SB_X+PAD, cy, cw, 158)
    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((x0+10, cy+10), "PUNCH COUNT", font=F_LABEL, fill=rgb(C_DIM))
    canvas[:] = pil2bgr(pil)
    cy2 = cy + 36
    total_punches = sum(_count.values())
    for i, pt in enumerate(PUNCH_BGR):
        col = PUNCH_BGR[pt]
        row = i // 2; col_i = i % 2
        bx_ = SB_X+PAD+10 + col_i*(cw//2)
        by_ = cy2 + row * 52
        draw_card(canvas, bx_-4, by_-4, cw//2-8, 44, color=(38,33,31), border=C_BORDER, radius=8)
        pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
        d.text((bx_+4, by_+2),  pt.upper(), font=F_SMALL, fill=rgb(C_MUTED))
        d.text((bx_+4, by_+18), f"{_count[pt]:03d}", font=F_TITLE, fill=rgb(col))
        canvas[:] = pil2bgr(pil)
    cy += 166

    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((x0+10, cy-6), f"총 {total_punches}회", font=F_SMALL, fill=rgb(C_DIM))
    canvas[:] = pil2bgr(pil)
    cy += 10

    # 자세 점수 카드
    if posture:
        total_s, grade, items = posture
        grade_col = C_GREEN if total_s>=85 else C_ACCENT if total_s>=70 else C_AMBER
        card_h = 36 + len(items)*42 + 10
        draw_card(canvas, SB_X+PAD, cy, cw, card_h)
        pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
        d.text((x0+10,    cy+10), "POSTURE",          font=F_LABEL, fill=rgb(C_DIM))
        d.text((x0+cw-90, cy+6),  f"{total_s}/100",  font=F_TITLE, fill=rgb(C_TEXT))
        d.text((x0+cw-30, cy+10), f"[{grade}]",      font=F_LABEL, fill=rgb(grade_col))
        canvas[:] = pil2bgr(pil)
        iy = cy + 38
        for label, score, mx, msg in items:
            ratio   = score/mx if mx else 0
            bar_col = C_GREEN if ratio>0.85 else C_ACCENT if ratio>0.5 else C_AMBER
            pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
            d.text((x0+10,    iy),    label,       font=F_SMALL, fill=rgb(C_MUTED))
            d.text((x0+cw-50, iy),    f"{score}/{mx}", font=F_SMALL, fill=rgb(C_TEXT))
            canvas[:] = pil2bgr(pil)
            draw_bar(canvas, x0+10, iy+18, cw-20, 6, ratio, bar_col)
            pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
            d.text((x0+10, iy+26), msg, font=F_SMALL, fill=rgb(bar_col if ratio<0.85 else C_DIM))
            canvas[:] = pil2bgr(pil)
            iy += 42
        cy += card_h + 8

    # 펀치 리포트
    if _report:
        age = now - _report['time']
        if age < REPORT_DUR and cy + 108 < WIN_H - 50:
            pt  = _report['type']
            col = PUNCH_BGR.get(pt, (0,200,200))
            dna = PUNCH_DNA.get(pt, {})
            draw_card(canvas, SB_X+PAD, cy, cw, 108)
            pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
            d.text((x0+10, cy+8), f"[ {pt.upper()} ]", font=F_TITLE, fill=rgb(col))
            metrics = [
                ('팔 뻗음',  _report['arm_ext'], dna.get('arm_extension_avg',0), TOL_PUNCH['arm_extension']),
                ('팔꿈치°',  _report['el_ang'],  dna.get('elbow_angle_avg',0),   TOL_PUNCH['elbow_angle']),
            ]
            for mi,(name,val,ref,tol) in enumerate(metrics):
                ok   = abs(val-ref) <= tol
                icon = '✓' if ok else ('▲' if val<ref else '▼')
                fc   = C_GREEN if ok else C_AMBER
                d.text((x0+10,  cy+36+mi*26), name,                   font=F_SMALL, fill=rgb(C_MUTED))
                d.text((x0+100, cy+36+mi*26), f"{val:.2f}/{ref:.2f} {icon}", font=F_SMALL, fill=rgb(fc))
            # 감지 신호 (디버그)
            sig_side = 'lead' if pt in ('jab','hook','uppercut') else 'rear'
            sig = _dbg_signal.get(sig_side, '')
            if sig:
                d.text((x0+10, cy+88), sig, font=F_SMALL, fill=rgb(C_DIM))
            canvas[:] = pil2bgr(pil)
            cy += 116

    # 단축키 힌트
    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    hints = [("R","초기화"),("D","방향"),("G","가이드"),("V",f"음성{'ON' if _voice_on else 'OFF'}"),("Q","종료")]
    hx = x0
    for key, label in hints:
        d.text((hx, WIN_H-28), key,   font=F_SMALL, fill=rgb(C_ACCENT))
        tw = int(d.textlength(key, font=F_SMALL))
        d.text((hx+tw+3, WIN_H-28), label, font=F_SMALL, fill=rgb(C_DIM))
        hx += tw + int(d.textlength(label+" ", font=F_SMALL)) + 10
    canvas[:] = pil2bgr(pil)

# ══════════════════════════════════════════════════════
# 펀치 플래시
# ══════════════════════════════════════════════════════
def draw_punch_flash(cam, now):
    if _flash and now < _flash[1]:
        col   = PUNCH_BGR.get(_flash[0], (0,200,200))
        age   = now - (_flash[1] - 0.18)
        alpha = max(0.0, 1.0 - age/0.18)
        ov = cam.copy()
        cv2.rectangle(ov, (0,0),(cam.shape[1]-1,cam.shape[0]-1), col, 8)
        cv2.addWeighted(ov, alpha*0.8, cam, 1-alpha*0.8, 0, cam)
        pil = bgr2pil(cam); d = ImageDraw.Draw(pil)
        d.text((cam.shape[1]//2-100, 20), _flash[0].upper(), font=F_DISPLAY, fill=(*rgb(col), int(alpha*220)))
        cam[:] = pil2bgr(pil)

# ══════════════════════════════════════════════════════
# 준비 화면
# ══════════════════════════════════════════════════════
def draw_ready_screen(canvas, pose_ok, stable):
    canvas[:] = C_INK
    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((WIN_W//2-200, WIN_H//2-160), "BOXING", font=F_DISPLAY, fill=rgb(C_TEXT))
    d.text((WIN_W//2-200+d.textlength("BOXING",font=F_DISPLAY)+4, WIN_H//2-160), ".", font=F_DISPLAY, fill=rgb(C_ACCENT))
    d.text((WIN_W//2-50, WIN_H//2-100), "COACH", font=F_TITLE, fill=rgb(C_MUTED))
    d.text((WIN_W//2-170, WIN_H//2-50), "LIM KWANWOO — 측면 카메라 모드", font=F_MEDIUM, fill=rgb(C_MUTED))
    canvas[:] = pil2bgr(pil)
    if pose_ok:
        prog = min(stable/READY_FRAMES, 1.0)
        draw_bar(canvas, WIN_W//2-180, WIN_H//2+20, 360, 8, prog, C_ACCENT)
        pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
        d.text((WIN_W//2-180, WIN_H//2-6), f"자세 인식 중… {int(prog*100)}%", font=F_MEDIUM, fill=rgb(C_GREEN))
        canvas[:] = pil2bgr(pil)
    else:
        draw_text_pil(canvas, "전신이 보이도록 서주세요", (WIN_W//2-130, WIN_H//2+20), F_MEDIUM, rgb(C_DIM))
    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    for i, t in enumerate(["측면 카메라 | 오르토독스 스탠스", "D키 방향전환  G가이드  V음성  R초기화  Q종료"]):
        d.text((WIN_W//2-180, WIN_H//2+60+i*26), t, font=F_SMALL, fill=rgb(C_DIM))
    canvas[:] = pil2bgr(pil)

# ══════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('BOXING COACH', cv2.WINDOW_NORMAL)
cv2.resizeWindow('BOXING COACH', WIN_W, WIN_H)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    now = time.time()

    fh, fw   = frame.shape[:2]
    cam_frame = cv2.resize(frame, (CAM_W, WIN_H))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    kps, scs  = pose_model(rgb_frame)

    pose_ok = len(kps) > 0
    kp = kps[0] if pose_ok else None
    sc = scs[0] if pose_ok else None

    kp_disp = None
    if pose_ok:
        pose_ok = all(sc[i] > VIS_MIN for i in NEEDED_KP)
        if pose_ok:
            kp_disp = kp.copy().astype(float)
            kp_disp[:, 0] *= CAM_W / fw
            kp_disp[:, 1] *= WIN_H / fh

    sw = sw_px(kp_disp) if kp_disp is not None else None

    canvas = np.full((WIN_H, WIN_W, 3), C_INK, dtype=np.uint8)

    if not _pose_ready:
        if pose_ok: _pose_stable = min(_pose_stable+1, READY_FRAMES)
        else:       _pose_stable = max(_pose_stable-2, 0)
        draw_ready_screen(canvas, pose_ok, _pose_stable)
        if _pose_stable >= READY_FRAMES: _pose_ready = True
        cv2.imshow('BOXING COACH', canvas)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
        continue

    if pose_ok and kp_disp is not None:
        _lm_buf.append((kp, sc))
        for entry in list(_pending):
            entry[0] -= 1
            if entry[0] <= 0:
                _pending.remove(entry)
                pt, side = entry[1], entry[2]
                heur = entry[3] if len(entry) > 3 else None
                # 분류 미수행이면 지금(확장 피크 시점) 매칭 수행
                if pt is None:
                    view = _view_mode
                    sw_v = sw_robust_kp(kp) if view == 'side' else sw_px(kp)
                    best_type, dist, sim = match_pose_template(kp, sw_v, view, side)

                    # 확장 시점 팔이 거의 펴짐 (ea > 155°) → 직선 펀치
                    extended = is_extended_punch(kp, sc, side)

                    # ── 우선순위 기반 분류 ──────────────────────────────
                    # 1) 팔 펴짐 + 휴리스틱 hook/uppercut 아님  → 잽/크로스 확정
                    # 2) 휴리스틱 발동 (hook/uppercut)         → 휴리스틱
                    # 3) 템플릿 매칭 성공                       → 템플릿
                    # 4) fallback                              → 잽/크로스
                    if extended and heur is None:
                        pt = 'jab' if side == 'lead' else 'cross'
                        src = 'extended'
                    elif heur:
                        pt = heur
                        src = 'heur'
                    elif best_type:
                        pt = best_type
                        src = 'tpl'
                    else:
                        pt = 'jab' if side == 'lead' else 'cross'
                        src = 'default'

                    _dbg_signal[side] = (f"{view} {side} → {pt} "
                                         f"[{src}] sim={sim:.0f}% d={dist:.2f}")
                    _last_pt[side] = pt
                analyse_punch(pt, side)
                _flash = (pt, now+0.18)
        update_punch_detect(kp, sc, now)
        _posture_data = calc_posture(kp_disp, sc, sw)
    else:
        _pose_stable = max(_pose_stable-2, 0)
        if _pose_stable < READY_FRAMES//2: _pose_ready = False

    canvas[:, :CAM_W] = cam_frame
    if kp_disp is not None:
        draw_skeleton(canvas[:, :CAM_W], kp_disp, sc)
        draw_trail(canvas[:, :CAM_W])
        if _show_ghost: draw_ghost(canvas[:, :CAM_W], kp_disp, sw)
    draw_punch_flash(canvas[:, :CAM_W], now)

    cv2.line(canvas, (CAM_W, 0), (CAM_W, WIN_H), C_BORDER, 1)
    render_sidebar(canvas, _posture_data, now)

    cv2.imshow('BOXING COACH', canvas)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27): break
    elif key == ord('r'):
        for t in _count: _count[t] = 0
        _report = None; _posture_data = None
        _dbg_signal['lead'] = ''; _dbg_signal['rear'] = ''
        _settled['lead'] = True; _settled['rear'] = True
        _det_lead.update({'active': False, 'start': 0.0, 'peak': 0.0, 'opp_peak': 0.0})
        _det_rear.update({'active': False, 'start': 0.0, 'peak': 0.0, 'opp_peak': 0.0})
    elif key == ord('g'): _show_ghost = not _show_ghost
    elif key == ord('v'): _voice_on   = not _voice_on
    elif key == ord('c'):
        _view_mode = 'front' if _view_mode == 'side' else 'side'
        _dbg_signal['lead'] = f'view={_view_mode}'; _dbg_signal['rear'] = ''
    elif key == ord('d'):
        _orthodox = not _orthodox
        _prev_lead_wr = None; _prev_rear_wr = None
        _det_lead.update({'active': False, 'start': 0.0, 'peak': 0.0, 'opp_peak': 0.0})
        _det_rear.update({'active': False, 'start': 0.0, 'peak': 0.0, 'opp_peak': 0.0})
        _settled['lead'] = True; _settled['rear'] = True
        _v_buf_lead.clear();  _v_buf_rear.clear()
        _vx_buf_lead.clear(); _vx_buf_rear.clear()
        _vy_buf_lead.clear(); _vy_buf_rear.clear()
        _ext_buf_lead.clear();_ext_buf_rear.clear()
        _ew_buf_lead.clear(); _ew_buf_rear.clear()
        _lat_buf_lead.clear(); _lat_buf_rear.clear()
        _er_buf_lead.clear();  _er_buf_rear.clear()
    elif key == ord('s'):
        _snap_count += 1
        fn = os.path.join(BASE_DIR, f'coach_snap_{_snap_count:03d}.png')
        cv2.imwrite(fn, canvas); print(f'스냅샷: {fn}')

cap.release()
cv2.destroyAllWindows()
