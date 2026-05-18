"""
boxing game 5.py — 웹 UI 스타일 방어 & 카운터 복싱 게임
─────────────────────────────────────────────────────────
boxing game 4 + 웹 디자인 (LIM coach web.py 참고)

변경 사항
  ● 웹 컬러 시스템  ink=#0b0d12 / canvas=#13151c / accent=#ff3d57
  ● 좌우 분할 레이아웃 — 좌 2/3 카메라, 우 1/3 사이드바
  ● DirectML GPU 자동 감지
  ● 펀치 감지 임계값 완화 (웹 detection.ts 동기화)
  ● Bebas Neue 폰트 (없으면 malgun.ttf 폴백)

단축키  1~4 난이도 / P 일시정지 / R 재시작 / Q 종료
"""

import cv2, numpy as np, math, os, time, random
from collections import deque
from PIL import ImageFont, ImageDraw, Image as PILImage

try:
    from rtmlib import RTMO
except ImportError:
    raise SystemExit("pip install rtmlib onnxruntime")

try:
    import sounddevice as _sd
    _sd.query_devices()
    _audio_ok = True
except Exception:
    _audio_ok = False

_SR = 44100

def _make_snd(freqs, dur, vol=0.7, decay_k=4.0, wave='sine'):
    n = int(_SR * dur)
    t = np.linspace(0, dur, n, False)
    w = np.zeros(n, dtype=np.float64)
    for f, amp in freqs:
        if wave == 'sine':   w += np.sin(2*np.pi*f*t) * amp
        elif wave == 'square': w += np.sign(np.sin(2*np.pi*f*t)) * amp
        elif wave == 'noise':  w += np.random.uniform(-1,1,n) * amp
    env = np.exp(-decay_k * t / dur)
    env[:max(1,int(0.005*_SR))] *= np.linspace(0,1,max(1,int(0.005*_SR)))
    return np.clip(w * env * vol, -1, 1).astype(np.float32)

def _snd(arr):
    if not _audio_ok or arr is None: return
    try: _sd.play(arr, _SR)
    except: pass

SND_BELL    = _make_snd([(880,0.6),(1760,0.3),(2640,0.1)], 0.9,  vol=0.6,  decay_k=3.0)
SND_DEFEND  = _make_snd([(440,0.7),(660,0.3)],              0.12, vol=0.45, decay_k=8.0)
SND_HIT     = _make_snd([(180,1.0)],                        0.18, vol=0.75, decay_k=5.0, wave='noise')
SND_PERFECT = _make_snd([(1046,0.6),(1318,0.3),(1568,0.1)], 0.40, vol=0.65, decay_k=2.5)
SND_FAIL    = _make_snd([(140,0.8),(105,0.2)],              0.45, vol=0.65, decay_k=2.0, wave='square')
SND_SPEED   = _make_snd([(660,0.5),(880,0.5)],              0.25, vol=0.7,  decay_k=1.5, wave='square')

# ══════════════════════════════════════════════════════════════════
# 색상 (BGR)
# ══════════════════════════════════════════════════════════════════
C_INK     = (18,  13,  11)   # #0b0d12
C_CANVAS  = (28,  21,  19)   # #13151c
C_CARD    = (48,  33,  30)   # #1e2130
C_BORDER  = (60,  45,  42)   # card border
C_ACCENT  = (87,  61, 255)   # #ff3d57  (BGR)
C_WHITE   = (230,230,230)
C_GRAY    = (120,120,120)
C_DGRAY   = (60,  60,  60)
C_GREEN   = (80, 210,  80)
C_CYAN    = (220,190,  50)   # ~#32bedc BGR
C_YELLOW  = (50, 230, 230)

# ══════════════════════════════════════════════════════════════════
# 모델
# ══════════════════════════════════════════════════════════════════
RTMO_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
    'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip'
)
print("RTMPose (RTMO-s) 로드 중...")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    device = 'dml' if 'DmlExecutionProvider' in providers else 'cpu'
except Exception:
    device = 'cpu'
print(f"  device: {device}")
pose_model = RTMO(RTMO_URL, backend='onnxruntime', device=device)
print("준비 완료\n")

# ══════════════════════════════════════════════════════════════════
# COCO 17 키포인트
# ══════════════════════════════════════════════════════════════════
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
VIS_MIN = 0.30
NEEDED  = [KP_L_SH, KP_R_SH, KP_L_WR, KP_R_WR, KP_L_EL, KP_R_EL, KP_NOSE]

# ══════════════════════════════════════════════════════════════════
# 펀치 감지 임계값 (웹 detection.ts 동기화)
# ══════════════════════════════════════════════════════════════════
BLOCK_NOSE_THRESH = -0.15   # 코 아래 15% 어깨너비까지 허용
SLIP_THRESH       =  0.20
GUARD_DROP_THRESH =  0.65
PUNCH_VEL         =  0.06
PUNCH_EXTEND      =  0.12
PUNCH_DOM         =  1.05
ARM_EXT_FULL      =  0.78
ARM_EXT_DELTA     =  0.18
VEL_BUF_LEN       =  8
COUNTER_DELAY     =  0.45

VALID_DEF = {
    'LEFT':  {'BLOCK_L': 'LEFT',  'SLIP_R': 'RIGHT'},
    'RIGHT': {'BLOCK_R': 'RIGHT', 'SLIP_L': 'LEFT'},
}

# ══════════════════════════════════════════════════════════════════
# 난이도
# ══════════════════════════════════════════════════════════════════
DIFF = {
    'EASY':    {'ai_hp':3,  'p_hp':100,'p_dmg':0,   'col':C_GREEN,  'label':'EASY'},
    'NORMAL':  {'ai_hp':5,  'p_hp':100,'p_dmg':20,  'col':C_CYAN,   'label':'NORMAL'},
    'HARD':    {'ai_hp':7,  'p_hp':100,'p_dmg':34,  'col':C_YELLOW, 'label':'HARD'},
    'EXTREME': {'ai_hp':10, 'p_hp':100,'p_dmg':100, 'col':C_ACCENT, 'label':'EXTREME'},
}

AI_PATTERNS = {
    'EASY':    None,
    'NORMAL':  [['LEFT','RIGHT'],['RIGHT','LEFT'],['LEFT'],['RIGHT'],['LEFT','RIGHT','LEFT']],
    'HARD':    [['LEFT','LEFT'],['RIGHT','RIGHT'],['LEFT','RIGHT','LEFT'],
                ['RIGHT','LEFT','RIGHT'],['LEFT','RIGHT','RIGHT'],['RIGHT','LEFT','LEFT']],
    'EXTREME': [['LEFT','LEFT','LEFT'],['RIGHT','RIGHT','RIGHT'],
                ['LEFT','RIGHT','LEFT'],['RIGHT','LEFT','RIGHT'],
                ['LEFT','LEFT','RIGHT'],['RIGHT','RIGHT','LEFT'],
                ['RIGHT','LEFT','LEFT'],['LEFT','RIGHT','RIGHT']],
}

TOTAL_ROUNDS     = 10
WARN_SINGLE      = (1.4, 2.0)
DEF_SINGLE       = (1.1, 1.5)
WARN_COMBO       = (0.6, 1.0)
DEF_COMBO        = (0.7, 1.0)
COUNTER_DUR      = 1.5
RESULT_DUR       = 0.8
SPEED_ALERT_DUR  = 2.5

# ══════════════════════════════════════════════════════════════════
# 폰트
# ══════════════════════════════════════════════════════════════════
def _find_font(size):
    candidates = [
        "C:/Windows/Fonts/BebasNeue-Regular.ttf",
        "C:/Windows/Fonts/Bebas_Neue.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]
    for p in candidates:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()

F_XS  = _find_font(16)
F_SM  = _find_font(22)
F_MD  = _find_font(32)
F_LG  = _find_font(52)
F_XL  = _find_font(80)
F_XXL = _find_font(120)

def put_text(img, text, pos, col, font=None, anchor='lt'):
    font = font or F_MD
    pil  = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d    = ImageDraw.Draw(pil)
    if anchor == 'center':
        bb = d.textbbox((0, 0), text, font=font)
        tw = bb[2] - bb[0]
        pos = (pos[0] - tw // 2, pos[1])
    d.text(pos, text, font=font, fill=(col[2], col[1], col[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ══════════════════════════════════════════════════════════════════
# 게임 상태
# ══════════════════════════════════════════════════════════════════
_gstate      = 'DIFF_SELECT'
_sub         = 'WARN'
_diff        = 'NORMAL'
_ai_hp       = 5;  _ai_hp_max  = 5
_p_hp        = 100; _p_hp_max  = 100; _p_dmg = 20
_round_num   = 0;  _combo = []; _combo_idx = 0
_defended    = False; _counter_arm = None
_countered   = False; _result_ok   = False
_phase_start = 0.0;  _cntdn = 3;  _score = 0
_paused      = False

_prev_kp_m          = None
_vel_buf_r          = deque(maxlen=VEL_BUF_LEN)
_vel_buf_l          = deque(maxlen=VEL_BUF_LEN)
_ext_buf_r          = deque(maxlen=VEL_BUF_LEN)
_ext_buf_l          = deque(maxlen=VEL_BUF_LEN)
_punch_base_r       = None
_punch_base_l       = None
_base_ext_r         = None
_base_ext_l         = None
_warn_dur           = 1.8
_defend_dur         = 1.4
_defend_phase_start = 0.0
_slip_buf           = deque(maxlen=3)

_speed_mode  = False; _speed_mult = 1.0; _speed_round = 0
_react_times = []

_shake_frames = 0; _shake_mag = 0
_ai_wobble    = 0.0; _ai_wobble_vel = 0.0; _ai_hit_flash = 0

# ══════════════════════════════════════════════════════════════════
# 기하 헬퍼
# ══════════════════════════════════════════════════════════════════
def spatial_arms(kp_m):
    if kp_m[KP_L_SH][0] > kp_m[KP_R_SH][0]:
        return {'r': [KP_L_SH, KP_L_EL, KP_L_WR], 'l': [KP_R_SH, KP_R_EL, KP_R_WR]}
    return {'r': [KP_R_SH, KP_R_EL, KP_R_WR], 'l': [KP_L_SH, KP_L_EL, KP_L_WR]}

def sw_m(kp_m):
    dx = kp_m[KP_R_SH][0] - kp_m[KP_L_SH][0]
    dy = kp_m[KP_R_SH][1] - kp_m[KP_L_SH][1]
    return math.sqrt(dx*dx + dy*dy) + 1e-6

def mirror_kp(kp, w):
    m = kp.copy(); m[:, 0] = w - kp[:, 0]; return m

def arm_extension(kp_m, sh_idx, wr_idx, sw):
    return math.hypot(kp_m[wr_idx][0]-kp_m[sh_idx][0], kp_m[wr_idx][1]-kp_m[sh_idx][1]) / sw

# ══════════════════════════════════════════════════════════════════
# 감지
# ══════════════════════════════════════════════════════════════════
def check_guard(kp_m, sw):
    sh_y = (kp_m[KP_L_SH][1] + kp_m[KP_R_SH][1]) / 2
    thr  = sh_y + GUARD_DROP_THRESH * sw
    arms = spatial_arms(kp_m)
    return kp_m[arms['r'][2]][1] < thr and kp_m[arms['l'][2]][1] < thr

def detect_block(kp_m, sw):
    nose_y = kp_m[KP_NOSE][1]
    thr    = nose_y - BLOCK_NOSE_THRESH * sw
    arms   = spatial_arms(kp_m)
    r_up   = kp_m[arms['r'][2]][1] < thr
    l_up   = kp_m[arms['l'][2]][1] < thr
    if r_up and not l_up: return 'RIGHT'
    if l_up and not r_up: return 'LEFT'
    return None

def detect_slip(kp_m, sw):
    sh_cx = (kp_m[KP_L_SH][0] + kp_m[KP_R_SH][0]) / 2
    dev   = (kp_m[KP_NOSE][0] - sh_cx) / sw
    if dev >  SLIP_THRESH: return 'RIGHT'
    if dev < -SLIP_THRESH: return 'LEFT'
    return None

def get_defense(kp_m, sw):
    b = detect_block(kp_m, sw)
    if b == 'RIGHT': return 'BLOCK_R'
    if b == 'LEFT':  return 'BLOCK_L'
    s = detect_slip(kp_m, sw)
    if s == 'RIGHT': return 'SLIP_R'
    if s == 'LEFT':  return 'SLIP_L'
    return None

def set_punch_baseline(kp_m, sw):
    global _punch_base_r, _punch_base_l, _base_ext_r, _base_ext_l
    arms         = spatial_arms(kp_m)
    _punch_base_r = kp_m[arms['r'][2]][:2].copy()
    _punch_base_l = kp_m[arms['l'][2]][:2].copy()
    _base_ext_r   = arm_extension(kp_m, arms['r'][0], arms['r'][2], sw)
    _base_ext_l   = arm_extension(kp_m, arms['l'][0], arms['l'][2], sw)

def detect_punch(kp_m, sw):
    global _prev_kp_m
    if _prev_kp_m is None or _punch_base_r is None or _base_ext_r is None:
        _prev_kp_m = kp_m.copy(); return None

    arms   = spatial_arms(kp_m)
    r_sh   = arms['r'][0]; l_sh = arms['l'][0]
    r_wr   = arms['r'][2]; l_wr = arms['l'][2]

    drx = kp_m[r_wr][0] - _prev_kp_m[r_wr][0]
    dlx = kp_m[l_wr][0] - _prev_kp_m[l_wr][0]
    vr  = max(0.0, -drx) / sw
    vl  = max(0.0,  dlx) / sw

    er     = math.hypot(kp_m[r_wr][0]-_punch_base_r[0], kp_m[r_wr][1]-_punch_base_r[1]) / sw
    el     = math.hypot(kp_m[l_wr][0]-_punch_base_l[0], kp_m[l_wr][1]-_punch_base_l[1]) / sw
    ext_r  = arm_extension(kp_m, r_sh, r_wr, sw)
    ext_l  = arm_extension(kp_m, l_sh, l_wr, sw)
    dext_r = ext_r - _base_ext_r
    dext_l = ext_l - _base_ext_l

    _prev_kp_m = kp_m.copy()
    _vel_buf_r.append(vr); _vel_buf_l.append(vl)
    _ext_buf_r.append(ext_r); _ext_buf_l.append(ext_l)

    pr       = max(_vel_buf_r)
    pl       = max(_vel_buf_l)
    peak_er  = max(_ext_buf_r)
    peak_el  = max(_ext_buf_l)
    pdext_r  = peak_er - _base_ext_r
    pdext_l  = peak_el - _base_ext_l

    horiz_r  = pr > PUNCH_VEL and er > PUNCH_EXTEND and pr > pl * PUNCH_DOM
    horiz_l  = pl > PUNCH_VEL and el > PUNCH_EXTEND and pl > pr * PUNCH_DOM
    ext_p_r  = peak_er > ARM_EXT_FULL and pdext_r > ARM_EXT_DELTA and pdext_r > pdext_l * PUNCH_DOM
    ext_p_l  = peak_el > ARM_EXT_FULL and pdext_l > ARM_EXT_DELTA and pdext_l > pdext_r * PUNCH_DOM

    if horiz_r or ext_p_r: return 'RIGHT'
    if horiz_l or ext_p_l: return 'LEFT'
    return None

# ══════════════════════════════════════════════════════════════════
# 이펙트
# ══════════════════════════════════════════════════════════════════
def trigger_shake(mag=14, frames=7):
    global _shake_frames, _shake_mag
    _shake_frames = max(_shake_frames, frames)
    _shake_mag    = max(_shake_mag, mag)

def trigger_ai_hit():
    global _ai_wobble_vel, _ai_hit_flash
    _ai_wobble_vel = random.choice([-2.8, 2.8])
    _ai_hit_flash  = 10

def apply_shake(frame, w, h):
    global _shake_frames, _shake_mag
    if _shake_frames <= 0: return frame
    ox  = random.randint(-_shake_mag, _shake_mag)
    oy  = random.randint(-_shake_mag//2, _shake_mag//2)
    M   = np.float32([[1,0,ox],[0,1,oy]])
    out = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    _shake_frames -= 1
    _shake_mag     = max(0, _shake_mag - 2)
    return out

# ══════════════════════════════════════════════════════════════════
# 스켈레톤
# ══════════════════════════════════════════════════════════════════
def draw_skeleton(frame, kp_m, sc):
    for a, b in COCO_CONN:
        if sc[a] > VIS_MIN and sc[b] > VIS_MIN:
            cv2.line(frame,
                     (int(kp_m[a][0]), int(kp_m[a][1])),
                     (int(kp_m[b][0]), int(kp_m[b][1])),
                     C_DGRAY, 2)
    for i in range(17):
        if sc[i] > VIS_MIN:
            cv2.circle(frame, (int(kp_m[i][0]), int(kp_m[i][1])), 4, C_CYAN, -1)

def highlight_arm(frame, kp_m, sc, arm, col):
    idxs = spatial_arms(kp_m)['r' if arm == 'RIGHT' else 'l']
    for i in range(len(idxs) - 1):
        a, b = idxs[i], idxs[i+1]
        if sc[a] > VIS_MIN and sc[b] > VIS_MIN:
            cv2.line(frame,
                     (int(kp_m[a][0]), int(kp_m[a][1])),
                     (int(kp_m[b][0]), int(kp_m[b][1])),
                     col, 6)
    if sc[idxs[2]] > VIS_MIN:
        cv2.circle(frame, (int(kp_m[idxs[2]][0]), int(kp_m[idxs[2]][1])), 12, col, -1)

def highlight_nose(frame, kp_m, sc, col):
    if sc[KP_NOSE] > VIS_MIN:
        nx, ny = int(kp_m[KP_NOSE][0]), int(kp_m[KP_NOSE][1])
        cv2.circle(frame, (nx, ny), 22, col, 4)
        cv2.circle(frame, (nx, ny), 10, col, -1)
        for sh in [KP_L_SH, KP_R_SH]:
            if sc[sh] > VIS_MIN:
                cv2.line(frame, (nx, ny), (int(kp_m[sh][0]), int(kp_m[sh][1])), col, 4)

def draw_center_line(frame, kp_m, sw, h):
    sh_cx = int((kp_m[KP_L_SH][0] + kp_m[KP_R_SH][0]) / 2)
    for y in range(0, h, 24):
        cv2.line(frame, (sh_cx, y), (sh_cx, min(y+12, h)), C_DGRAY, 1)
    nx, ny = int(kp_m[KP_NOSE][0]), int(kp_m[KP_NOSE][1])
    dev    = (nx - sh_cx) / sw
    col    = C_ACCENT if abs(dev) >= SLIP_THRESH else C_GREEN
    cv2.circle(frame, (nx, ny), 10, col, 2)
    if abs(dev) > 0.10:
        cv2.arrowedLine(frame, (sh_cx, ny), (nx, ny), col, 2, tipLength=0.3)

# ══════════════════════════════════════════════════════════════════
# AI 복서 실루엣
# ══════════════════════════════════════════════════════════════════
def draw_ai_boxer(frame, w, h, phase='IDLE'):
    global _ai_wobble, _ai_wobble_vel, _ai_hit_flash
    _ai_wobble     += _ai_wobble_vel * 0.07
    _ai_wobble_vel *= 0.80
    _ai_wobble     *= 0.90
    if _ai_hit_flash > 0: _ai_hit_flash -= 1

    cx  = int(w // 2 + math.sin(_ai_wobble) * 40)
    cy  = h // 6
    hit = _ai_hit_flash > 0

    body_col = (50, 60, 200) if hit else (65, 55, 45)
    glv_col  = (30, 30, 160) if hit else (25, 25, 80)

    ov = frame.copy()
    cv2.circle(ov,    (cx, cy-30), 22, body_col, -1)
    cv2.circle(ov,    (cx, cy-30), 22, (110,110,110), 2)
    cv2.rectangle(ov, (cx-24, cy-8), (cx+24, cy+52), body_col, -1)
    if phase == 'WARN':
        cv2.line(ov, (cx-24,cy+5), (cx-38,cy-8), body_col, 11)
        cv2.line(ov, (cx+24,cy+5), (cx+38,cy-8), body_col, 11)
        cv2.circle(ov, (cx-38,cy-8),  9, glv_col, -1)
        cv2.circle(ov, (cx+38,cy-8),  9, glv_col, -1)
    else:
        cv2.line(ov, (cx-24,cy+5), (cx-48,cy-20), body_col, 11)
        cv2.line(ov, (cx+24,cy+5), (cx+48,cy-20), body_col, 11)
        cv2.circle(ov, (cx-48,cy-20), 10, glv_col, -1)
        cv2.circle(ov, (cx+48,cy-20), 10, glv_col, -1)
    cv2.rectangle(ov, (cx-21,cy+52),(cx-7, cy+95), body_col, -1)
    cv2.rectangle(ov, (cx+7, cy+52),(cx+21,cy+95), body_col, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

# ══════════════════════════════════════════════════════════════════
# 공격 화살표
# ══════════════════════════════════════════════════════════════════
def draw_attack_arrow(frame, w, h, attack):
    cx, cy = w // 2, h // 2 - 30
    if attack == 'LEFT':
        pts = np.array([[cx-140,cy],[cx-60,cy-55],[cx-60,cy-18],
                         [cx+120,cy-18],[cx+120,cy+18],[cx-60,cy+18],[cx-60,cy+55]], np.int32)
        lbl = '← 왼쪽 공격!'
    else:
        pts = np.array([[cx+140,cy],[cx+60,cy-55],[cx+60,cy-18],
                         [cx-120,cy-18],[cx-120,cy+18],[cx+60,cy+18],[cx+60,cy+55]], np.int32)
        lbl = '오른쪽 공격! →'
    ov = frame.copy()
    cv2.fillPoly(ov, [pts], C_ACCENT)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    cv2.polylines(frame, [pts], True, C_ACCENT, 2)
    put_text(frame, lbl, (w//2, h//2+80), C_ACCENT, F_LG, anchor='center')

# ══════════════════════════════════════════════════════════════════
# 타이머 바
# ══════════════════════════════════════════════════════════════════
def timer_bar(frame, w, h, elapsed, total, col, bar_h=8):
    ratio = max(0.0, 1.0 - elapsed / total)
    cv2.rectangle(frame, (0, h-bar_h), (w, h), C_CARD, -1)
    cv2.rectangle(frame, (0, h-bar_h), (int(w*ratio), h), col, -1)

# ══════════════════════════════════════════════════════════════════
# 게임 오버레이 (카메라 영역에 그림)
# ══════════════════════════════════════════════════════════════════
def overlay_rect(frame, alpha, col=C_INK):
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (frame.shape[1], frame.shape[0]), col, -1)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

def draw_warn_phase(cam, cw, ch, attack):
    overlay_rect(cam, 0.20, (20, 0, 0))
    draw_attack_arrow(cam, cw, ch, attack)

def draw_defend_phase(cam, cw, ch, attack, elapsed):
    overlay_rect(cam, 0.25, (0, 10, 30))
    draw_attack_arrow(cam, cw, ch, attack)
    h1, h2 = (('← 왼팔 올려서 막기', '→ 오른쪽으로 슬립') if attack == 'LEFT'
               else ('오른팔 올려서 막기 →', '← 왼쪽으로 슬립'))
    put_text(cam, '[ 방어하세요! ]', (cw//2, ch//2-130), C_CYAN, F_MD, anchor='center')
    put_text(cam, h1, (cw//2, ch//2+130), C_CYAN, F_SM, anchor='center')
    put_text(cam, h2, (cw//2, ch//2+160), C_GRAY, F_SM, anchor='center')
    timer_bar(cam, cw, ch, elapsed, _defend_dur, C_CYAN)

def draw_counter_phase(cam, cw, ch, counter_arm, elapsed):
    overlay_rect(cam, 0.20, (0, 30, 0))
    arm_kr = '오른손' if counter_arm == 'RIGHT' else '왼손'
    put_text(cam, '카운터!', (cw//2, ch//2-100), C_GREEN, F_XL, anchor='center')
    put_text(cam, f'{arm_kr}으로 펀치!', (cw//2, ch//2+10), C_GREEN, F_LG, anchor='center')
    put_text(cam, '(안 쳐도 OK)', (cw//2, ch//2+80), C_GRAY, F_SM, anchor='center')
    timer_bar(cam, cw, ch, elapsed, COUNTER_DUR, C_GREEN)

def draw_result_flash(cam, cw, ch, ok, perfect=False):
    if ok and perfect:
        overlay_rect(cam, 0.15, (0, 20, 10))
        for i in range(3):
            cv2.rectangle(cam, (3+i*4,3+i*4),(cw-3-i*4,ch-3-i*4),(0,220,180),2)
        put_text(cam, 'PERFECT!', (cw//2, ch//2-60), (180,255,200), F_XL, anchor='center')
    elif ok:
        overlay_rect(cam, 0.18, (0, 30, 0))
        put_text(cam, '성공!', (cw//2, ch//2-50), C_GREEN, F_XL, anchor='center')
    else:
        overlay_rect(cam, 0.30, (30, 0, 0))
        put_text(cam, '실패!', (cw//2, ch//2-50), C_ACCENT, F_XL, anchor='center')

# ══════════════════════════════════════════════════════════════════
# 게임 로직
# ══════════════════════════════════════════════════════════════════
def gen_combo():
    patterns = AI_PATTERNS.get(_diff)
    if patterns is None or random.random() < 0.3:
        return [random.choice(['LEFT','RIGHT']) for _ in range(random.randint(1,3))]
    return list(random.choice(patterns))

def start_attack():
    global _sub, _phase_start, _defended, _counter_arm, _countered, _result_ok
    global _prev_kp_m, _vel_buf_r, _vel_buf_l, _ext_buf_r, _ext_buf_l
    global _punch_base_r, _punch_base_l, _base_ext_r, _base_ext_l
    _sub = 'WARN'; _phase_start = time.time()
    _defended = False; _counter_arm = None; _countered = False; _result_ok = False
    _prev_kp_m = None
    _vel_buf_r.clear(); _vel_buf_l.clear()
    _ext_buf_r.clear(); _ext_buf_l.clear()
    _punch_base_r = None; _punch_base_l = None
    _base_ext_r   = None; _base_ext_l   = None
    _slip_buf.clear()

def start_round_combo():
    global _combo, _combo_idx, _warn_dur, _defend_dur
    _combo = gen_combo(); _combo_idx = 0
    mult = _speed_mult
    if len(_combo) == 1:
        _warn_dur   = random.uniform(*WARN_SINGLE) / mult
        _defend_dur = random.uniform(*DEF_SINGLE)  / mult
    else:
        _warn_dur   = random.uniform(*WARN_COMBO) / mult
        _defend_dur = random.uniform(*DEF_COMBO)  / mult
    start_attack()

def start_game(diff_key):
    global _gstate, _diff, _ai_hp, _ai_hp_max, _p_hp, _p_hp_max, _p_dmg
    global _round_num, _cntdn, _score, _speed_mode, _speed_mult, _speed_round, _react_times
    global _shake_frames, _shake_mag, _ai_wobble, _ai_wobble_vel, _ai_hit_flash, _phase_start
    _diff = diff_key; cfg = DIFF[diff_key]
    _ai_hp = _ai_hp_max = cfg['ai_hp']
    _p_hp  = _p_hp_max  = cfg['p_hp']
    _p_dmg = cfg['p_dmg']
    _round_num = 0; _score = 0
    _speed_mode = False; _speed_mult = 1.0; _speed_round = 0; _react_times = []
    _shake_frames = 0; _shake_mag = 0; _ai_wobble = 0.0; _ai_wobble_vel = 0.0; _ai_hit_flash = 0
    _gstate = 'COUNTDOWN'; _cntdn = 3; _phase_start = time.time()

def advance():
    global _round_num, _gstate, _speed_mode, _speed_mult, _speed_round, _phase_start
    _round_num += 1
    if _round_num >= TOTAL_ROUNDS and not _speed_mode:
        _speed_mode = True; _speed_mult = 2.0; _speed_round = 0
        _gstate = 'SPEED_ALERT'; _phase_start = time.time()
        _snd(SND_SPEED); return
    if _speed_mode:
        _speed_round += 1; _speed_mult = 2.0 + _speed_round * 0.15
    start_round_combo()

# ══════════════════════════════════════════════════════════════════
# 사이드바 렌더링
# ══════════════════════════════════════════════════════════════════
def draw_sidebar(sidebar):
    sh, sw_px = sidebar.shape[:2]
    sidebar[:] = C_INK

    # ─── 헤더 ───────────────────────────────────────────────────
    cv2.rectangle(sidebar, (0,0), (sw_px, 52), C_CANVAS, -1)
    put_text(sidebar, 'BOXING.GAME', (sw_px//2, 10), C_WHITE, F_MD, anchor='center')
    cv2.line(sidebar, (0, 52), (sw_px, 52), C_BORDER, 1)

    y = 68

    # ─── 난이도 & 라운드 ─────────────────────────────────────────
    cfg = DIFF[_diff]
    cv2.rectangle(sidebar, (12, y), (sw_px-12, y+72), C_CARD, -1)
    cv2.rectangle(sidebar, (12, y), (sw_px-12, y+72), C_BORDER, 1)
    put_text(sidebar, cfg['label'], (24, y+6), cfg['col'], F_SM)
    if _speed_mode:
        info = f"SPEED x{_speed_mult:.1f}  #스피드{_speed_round+1}"
        put_text(sidebar, info, (24, y+36), C_ACCENT, F_XS)
    else:
        put_text(sidebar, f"Round  {_round_num+1} / {TOTAL_ROUNDS}", (24, y+36), C_GRAY, F_XS)
    put_text(sidebar, f"Score  {_score}", (sw_px-24-80, y+36), C_CYAN, F_XS)
    y += 84

    # ─── HP 바 (플레이어) ─────────────────────────────────────────
    cv2.rectangle(sidebar, (12, y), (sw_px-12, y+64), C_CARD, -1)
    cv2.rectangle(sidebar, (12, y), (sw_px-12, y+64), C_BORDER, 1)
    put_text(sidebar, 'HP', (24, y+6), C_GRAY, F_XS)
    hp_lbl = 'INF' if _p_dmg == 0 else str(_p_hp)
    put_text(sidebar, hp_lbl, (sw_px-24-60, y+6), C_WHITE, F_XS)
    ratio_p = max(0.0, _p_hp / _p_hp_max)
    bar_col = C_GREEN if ratio_p > 0.5 else C_YELLOW if ratio_p > 0.25 else C_ACCENT
    bx, by, bw2, bh2 = 24, y+34, sw_px-48, 16
    cv2.rectangle(sidebar, (bx, by), (bx+bw2, by+bh2), C_DGRAY, -1)
    cv2.rectangle(sidebar, (bx, by), (bx+int(bw2*ratio_p), by+bh2), bar_col, -1)
    y += 76

    # ─── AI HP (별) ──────────────────────────────────────────────
    cv2.rectangle(sidebar, (12, y), (sw_px-12, y+58), C_CARD, -1)
    cv2.rectangle(sidebar, (12, y), (sw_px-12, y+58), C_BORDER, 1)
    put_text(sidebar, 'AI', (24, y+6), C_GRAY, F_XS)
    ai_max  = DIFF[_diff]['ai_hp']
    stars   = '★' * _ai_hp + '☆' * (ai_max - _ai_hp)
    ai_col  = C_ACCENT if _ai_hp <= ai_max // 3 else C_CYAN
    put_text(sidebar, stars, (24, y+30), ai_col, F_SM)
    y += 70

    # ─── 현재 콤보 ───────────────────────────────────────────────
    if _gstate == 'PLAYING' and _combo:
        cv2.rectangle(sidebar, (12, y), (sw_px-12, y+62), C_CARD, -1)
        cv2.rectangle(sidebar, (12, y), (sw_px-12, y+62), C_BORDER, 1)
        put_text(sidebar, 'COMBO', (24, y+4), C_GRAY, F_XS)
        combo_str = '  '.join(
            ('◆' if i == _combo_idx else ('✓' if i < _combo_idx else '◇')) + ' ' + s[:1]
            for i, s in enumerate(_combo)
        )
        col_c = C_ACCENT if _sub == 'WARN' else C_CYAN if _sub == 'DEFEND' else C_GREEN
        put_text(sidebar, combo_str, (24, y+30), col_c, F_SM)
        y += 74

    # ─── 방어 힌트 ───────────────────────────────────────────────
    if _gstate == 'PLAYING' and _sub in ('WARN', 'DEFEND') and _combo:
        attack = _combo[_combo_idx]
        cv2.rectangle(sidebar, (12, y), (sw_px-12, y+90), C_CARD, -1)
        cv2.rectangle(sidebar, (12, y), (sw_px-12, y+90), C_BORDER, 1)
        put_text(sidebar, 'HOW TO DEFEND', (24, y+4), C_GRAY, F_XS)
        if attack == 'LEFT':
            put_text(sidebar, '왼팔 올려서 막기', (24, y+28), C_WHITE, F_SM)
            put_text(sidebar, '또는 오른쪽으로 슬립', (24, y+56), C_GRAY, F_SM)
        else:
            put_text(sidebar, '오른팔 올려서 막기', (24, y+28), C_WHITE, F_SM)
            put_text(sidebar, '또는 왼쪽으로 슬립', (24, y+56), C_GRAY, F_SM)
        y += 102

    # ─── 반응속도 ─────────────────────────────────────────────────
    if _react_times:
        avg_ms  = int(sum(_react_times) / len(_react_times) * 1000)
        best_ms = int(min(_react_times) * 1000)
        cv2.rectangle(sidebar, (12, y), (sw_px-12, y+58), C_CARD, -1)
        cv2.rectangle(sidebar, (12, y), (sw_px-12, y+58), C_BORDER, 1)
        put_text(sidebar, 'REACT', (24, y+4), C_GRAY, F_XS)
        put_text(sidebar, f'avg {avg_ms}ms  best {best_ms}ms', (24, y+28), C_GREEN, F_SM)
        y += 70

    # ─── 키 안내 ─────────────────────────────────────────────────
    ky = sh - 80
    cv2.line(sidebar, (12, ky-8), (sw_px-12, ky-8), C_BORDER, 1)
    put_text(sidebar, 'P 일시정지   R 재시작', (sw_px//2, ky+2), C_DGRAY, F_XS, anchor='center')
    put_text(sidebar, 'Q 종료   1~4 난이도',  (sw_px//2, ky+26), C_DGRAY, F_XS, anchor='center')

# ══════════════════════════════════════════════════════════════════
# 전체 화면 오버레이 (선택/카운트다운/스피드/승패)
# ══════════════════════════════════════════════════════════════════
def draw_diff_select(canvas, cw, ch):
    overlay_rect(canvas, 0.82)
    put_text(canvas, 'BOXING DEFENSE', (cw//2, ch//2-220), C_WHITE,  F_XL,  anchor='center')
    put_text(canvas, 'GAME',           (cw//2, ch//2-140), C_ACCENT, F_XL,  anchor='center')
    put_text(canvas, '난이도를 선택하세요  (1~4)', (cw//2, ch//2-70), C_GRAY, F_MD, anchor='center')
    rows = [
        ('1  EASY',    'AI 체력 3  /  HP 무한',     DIFF['EASY']['col']),
        ('2  NORMAL',  'AI 체력 5  /  HP 100',      DIFF['NORMAL']['col']),
        ('3  HARD',    'AI 체력 7  /  HP 100',      DIFF['HARD']['col']),
        ('4  EXTREME', 'AI 체력 10  /  즉사',        DIFF['EXTREME']['col']),
    ]
    for i, (lbl, desc, col) in enumerate(rows):
        y = ch//2 - 10 + i * 72
        cv2.rectangle(canvas, (cw//2-200, y), (cw//2+200, y+60), C_CARD, -1)
        cv2.rectangle(canvas, (cw//2-200, y), (cw//2+200, y+60), col, 1)
        put_text(canvas, lbl,  (cw//2-180, y+4),  col,    F_SM)
        put_text(canvas, desc, (cw//2-180, y+32), C_GRAY, F_XS)

def draw_countdown_overlay(canvas, cw, ch):
    overlay_rect(canvas, 0.65)
    put_text(canvas, DIFF[_diff]['label'], (cw//2, ch//2-200), DIFF[_diff]['col'], F_LG,  anchor='center')
    put_text(canvas, '준비!',              (cw//2, ch//2-120), C_GRAY,             F_MD,  anchor='center')
    n   = str(_cntdn) if _cntdn > 0 else 'GO!'
    col = C_CYAN if _cntdn > 0 else C_GREEN
    put_text(canvas, n, (cw//2, ch//2-40), col, F_XXL, anchor='center')

def draw_speed_overlay(canvas, cw, ch):
    overlay_rect(canvas, 0.80, (20, 0, 30))
    put_text(canvas, 'SPEED MODE!',   (cw//2, ch//2-140), C_CYAN,   F_XL, anchor='center')
    put_text(canvas, '2배속으로 버텨라!', (cw//2, ch//2+0),  C_WHITE,  F_LG, anchor='center')
    put_text(canvas, '방어 실패 = 게임 오버', (cw//2, ch//2+70), C_GRAY, F_MD, anchor='center')
    for i in range(3):
        cv2.rectangle(canvas, (3+i*4,3+i*4),(cw-3-i*4,ch-3-i*4),(0,80,255),2)

def draw_win_overlay(canvas, cw, ch):
    overlay_rect(canvas, 0.80, (0, 15, 0))
    put_text(canvas, '승리!',           (cw//2, ch//2-180), C_GREEN, F_XL,  anchor='center')
    put_text(canvas, f'카운터 {_score}번', (cw//2, ch//2-80),  C_WHITE, F_LG,  anchor='center')
    if _speed_mode and _speed_round > 0:
        put_text(canvas, f'스피드 {_speed_round}라운드 생존', (cw//2, ch//2+10), C_CYAN, F_MD, anchor='center')
    if _react_times:
        avg_ms  = int(sum(_react_times) / len(_react_times) * 1000)
        best_ms = int(min(_react_times) * 1000)
        put_text(canvas, f'평균 반응: {avg_ms}ms  최고: {best_ms}ms', (cw//2, ch//2+65), C_GREEN, F_SM, anchor='center')
    put_text(canvas, 'R = 재시작  /  Q = 종료', (cw//2, ch//2+140), C_GRAY, F_MD, anchor='center')

def draw_lose_overlay(canvas, cw, ch):
    overlay_rect(canvas, 0.80, (20, 0, 0))
    title = f'스피드 {_speed_round}라운드 생존' if _speed_mode else '패배'
    tcol  = C_CYAN if _speed_mode else C_ACCENT
    put_text(canvas, title,             (cw//2, ch//2-180), tcol,   F_XL, anchor='center')
    put_text(canvas, f'카운터 {_score}번', (cw//2, ch//2-80),  C_WHITE, F_LG, anchor='center')
    if _react_times:
        avg_ms = int(sum(_react_times) / len(_react_times) * 1000)
        put_text(canvas, f'평균 반응: {avg_ms}ms', (cw//2, ch//2+10), C_GRAY, F_SM, anchor='center')
    put_text(canvas, 'R = 재시작  /  Q = 종료', (cw//2, ch//2+100), C_GRAY, F_MD, anchor='center')

def draw_pause_overlay(canvas, cw, ch):
    overlay_rect(canvas, 0.70)
    put_text(canvas, 'PAUSE', (cw//2, ch//2-60), C_WHITE, F_XL, anchor='center')
    put_text(canvas, 'P 키로 재개',    (cw//2, ch//2+40), C_GRAY, F_MD, anchor='center')

# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
CAM_W  = 960
CAM_H  = 540
SIDE_W = 320
WIN_W  = CAM_W + SIDE_W
WIN_H  = CAM_H

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Boxing Game 5', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Boxing Game 5', WIN_W, WIN_H)

canvas   = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
sidebar  = np.zeros((WIN_H, SIDE_W, 3), dtype=np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h_raw, w_raw = frame.shape[:2]
    now = time.time()

    # ── 포즈 추정 ─────────────────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    kps, scs = pose_model(rgb)
    disp = cv2.flip(frame, 1)
    disp = cv2.resize(disp, (CAM_W, CAM_H))

    pose_ok = len(kps) > 0
    kp  = kps[0] if pose_ok else None
    sc  = scs[0] if pose_ok else None
    kp_m = None
    if pose_ok:
        pose_ok = all(sc[i] > VIS_MIN for i in NEEDED)
    if pose_ok:
        kp_m = mirror_kp(kp, w_raw)
        # 좌표를 표시 해상도로 스케일
        kp_m[:, 0] = kp_m[:, 0] * CAM_W / w_raw
        kp_m[:, 1] = kp_m[:, 1] * CAM_H / h_raw
        sw_val = sw_m(kp_m)
        draw_skeleton(disp, kp_m, sc)

    # ── 일시정지 ─────────────────────────────────────────────────
    if _paused and _gstate == 'PLAYING':
        draw_pause_overlay(disp, CAM_W, CAM_H)
        draw_sidebar(sidebar)
        canvas[:, :CAM_W]  = disp
        canvas[:, CAM_W:]  = sidebar
        cv2.imshow('Boxing Game 5', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        elif key == ord('p'):
            _paused = False
            _phase_start = time.time() - (time.time() - _phase_start)  # 시간 보정
        continue

    # ── 상태 머신 ─────────────────────────────────────────────────
    if _gstate == 'DIFF_SELECT':
        draw_diff_select(disp, CAM_W, CAM_H)

    elif _gstate == 'COUNTDOWN':
        draw_countdown_overlay(disp, CAM_W, CAM_H)
        elapsed = now - _phase_start
        new_n   = 3 - int(elapsed)
        if new_n != _cntdn and new_n >= 0:
            _cntdn = new_n
        if elapsed >= 4.0:
            _snd(SND_BELL)
            _gstate = 'PLAYING'; start_round_combo()

    elif _gstate == 'SPEED_ALERT':
        draw_speed_overlay(disp, CAM_W, CAM_H)
        if now - _phase_start >= SPEED_ALERT_DUR:
            _gstate = 'PLAYING'; start_round_combo()

    elif _gstate == 'PLAYING':
        attack  = _combo[_combo_idx]
        elapsed = now - _phase_start

        draw_ai_boxer(disp, CAM_W, CAM_H, phase=_sub)
        if kp_m is not None:
            draw_center_line(disp, kp_m, sw_val, CAM_H)

        # 가드 경고
        if kp_m is not None and _sub in ('WARN', 'DEFEND'):
            if not check_guard(kp_m, sw_val):
                put_text(disp, '★ 가드 올려! ★', (CAM_W//2, CAM_H-75), C_ACCENT, F_MD, anchor='center')

        if _sub == 'WARN':
            draw_warn_phase(disp, CAM_W, CAM_H, attack)
            timer_bar(disp, CAM_W, CAM_H, elapsed, _warn_dur, C_ACCENT)
            if elapsed >= _warn_dur:
                _sub = 'DEFEND'; _phase_start = now; _defend_phase_start = now

        elif _sub == 'DEFEND':
            draw_defend_phase(disp, CAM_W, CAM_H, attack, elapsed)
            if kp_m is not None and not _defended:
                _raw = get_defense(kp_m, sw_val)
                _slip_buf.append(_raw)
                if _raw and not _raw.startswith('SLIP'):
                    defense = _raw
                elif len(_slip_buf) >= 2 and _slip_buf[-1] == _slip_buf[-2] and _slip_buf[-1]:
                    defense = _slip_buf[-1]
                else:
                    defense = None
                if defense and defense in VALID_DEF.get(attack, {}):
                    _react_times.append(now - _defend_phase_start)
                    _defended = True; _counter_arm = VALID_DEF[attack][defense]
                    if defense.startswith('SLIP'):
                        highlight_nose(disp, kp_m, sc, C_GREEN)
                    else:
                        highlight_arm(disp, kp_m, sc, 'RIGHT' if 'R' in defense else 'LEFT', C_GREEN)
                    _snd(SND_DEFEND)
                    if _combo_idx < len(_combo) - 1:
                        _combo_idx += 1; start_attack()
                    else:
                        _sub = 'COUNTER'; _phase_start = now; _prev_kp_m = None
            if elapsed >= _defend_dur and _sub == 'DEFEND':
                _p_hp = max(0, _p_hp - _p_dmg); _result_ok = False
                _snd(SND_FAIL)
                _sub = 'RESULT'; _phase_start = now
                if _p_hp <= 0: _gstate = 'LOSE'

        elif _sub == 'COUNTER':
            draw_counter_phase(disp, CAM_W, CAM_H, _counter_arm, elapsed)
            if kp_m is not None:
                if elapsed < COUNTER_DELAY:
                    _prev_kp_m = kp_m.copy()
                    if _punch_base_r is None:
                        set_punch_baseline(kp_m, sw_val)
                elif not _countered:
                    punch = detect_punch(kp_m, sw_val)
                    if punch == _counter_arm:
                        _countered = True; _result_ok = True
                        _score += 1; _ai_hp = max(0, _ai_hp - 1)
                        highlight_arm(disp, kp_m, sc, _counter_arm, C_CYAN)
                        trigger_shake(mag=16, frames=8)
                        trigger_ai_hit()
                        _snd(SND_PERFECT)
                        _sub = 'RESULT'; _phase_start = now
                        if _ai_hp <= 0: _gstate = 'WIN'
                else:
                    detect_punch(kp_m, sw_val)
            if elapsed >= COUNTER_DUR and _sub == 'COUNTER':
                _result_ok = True; _sub = 'RESULT'; _phase_start = now

        elif _sub == 'RESULT':
            draw_result_flash(disp, CAM_W, CAM_H, _result_ok, perfect=(_result_ok and _countered))
            if now - _phase_start >= RESULT_DUR:
                if _gstate == 'PLAYING': advance()

    elif _gstate == 'WIN':
        if kp_m is not None: draw_skeleton(disp, kp_m, sc)
        draw_win_overlay(disp, CAM_W, CAM_H)

    elif _gstate == 'LOSE':
        draw_lose_overlay(disp, CAM_W, CAM_H)

    # ── 흔들림 + 합성 ─────────────────────────────────────────────
    disp = apply_shake(disp, CAM_W, CAM_H)
    draw_sidebar(sidebar)
    canvas[:, :CAM_W] = disp
    canvas[:, CAM_W:] = sidebar

    cv2.imshow('Boxing Game 5', canvas)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27): break
    elif key == ord('r'):
        _gstate = 'DIFF_SELECT'; _paused = False
    elif key == ord('p') and _gstate == 'PLAYING':
        _paused = not _paused
    elif _gstate == 'DIFF_SELECT':
        if   key == ord('1'): start_game('EASY')
        elif key == ord('2'): start_game('NORMAL')
        elif key == ord('3'): start_game('HARD')
        elif key == ord('4'): start_game('EXTREME')

cap.release()
cv2.destroyAllWindows()
