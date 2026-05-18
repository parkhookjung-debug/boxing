"""
LIM coach web.py — 웹 UI 스타일 로컬 복싱 코치
────────────────────────────────────────────────
RTMPose (RTMO-s) | 측면 카메라 | 오르토독스 스탠스

단축키
  Q / ESC   종료
  R         카운터 초기화
  D         방향 전환 (우향/좌향)
  G         가이드(고스트) 토글
  V         음성 ON/OFF
  S         스냅샷 저장
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
C_INK     = (18,  13,  11)   # #0b0d12
C_CANVAS  = (28,  21,  19)   # #13151c
C_BORDER  = (40,  35,  33)   # white/5
C_ACCENT  = (87,  61, 255)   # #ff3d57
C_TEXT    = (245, 244, 244)  # zinc-100
C_MUTED   = (170, 161, 161)  # zinc-400
C_DIM     = (122, 113, 113)  # zinc-500
C_GREEN   = (100, 220,   0)  # success
C_AMBER   = ( 50, 165, 255)  # warning

PUNCH_BGR = {
    'jab':      (  0, 220, 255),  # #ffdc00
    'cross':    (255, 160,  40),  # #28a0ff
    'hook':     (200,  50, 255),  # #ff32c8
    'uppercut': (100, 255,  50),  # #32ff64
}

# ══════════════════════════════════════════════════════
# 폰트
# ══════════════════════════════════════════════════════
def _font(size):
    candidates = [
        "C:/Windows/Fonts/BebasNeue-Regular.ttf",
        os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts/BebasNeue-Regular.ttf"),
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except:
            pass
    return ImageFont.load_default()

def _font_kr(size):
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "C:/Windows/Fonts/NanumGothic.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except:
            pass
    return ImageFont.load_default()

F_DISPLAY = _font(52)
F_TITLE   = _font(32)
F_LABEL   = _font(20)
F_SMALL   = _font_kr(15)
F_MEDIUM  = _font_kr(18)
F_LARGE   = _font_kr(22)

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
# RTMPose
# ══════════════════════════════════════════════════════
RTMO_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
    'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip'
)
print("RTMPose (RTMO-s) 로드 중...")
pose_model = RTMO(RTMO_URL, backend='onnxruntime', device='cpu')
print("모델 준비 완료")

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

_facing_right = True

def get_arm_indices():
    if _facing_right:
        return KP_R_WR, KP_R_SH, KP_R_EL, KP_L_WR, KP_L_SH, KP_L_EL
    return KP_L_WR, KP_L_SH, KP_L_EL, KP_R_WR, KP_R_SH, KP_R_EL

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
# 상태
# ══════════════════════════════════════════════════════
_count       = {t:0 for t in PUNCH_BGR}
_trail_lead  = deque(maxlen=22)
_trail_rear  = deque(maxlen=22)
_last_pt     = {'lead':'jab','rear':'cross'}
_lm_buf      = deque(maxlen=30)
_pending     = []
EXT_DELAY    = 8

_BUF_SZ      = 12
_v_buf_lead  = deque(maxlen=_BUF_SZ); _v_buf_rear  = deque(maxlen=_BUF_SZ)
_el_buf_lead = deque(maxlen=_BUF_SZ); _el_buf_rear = deque(maxlen=_BUF_SZ)
_ea_buf_lead = deque(maxlen=_BUF_SZ); _ea_buf_rear = deque(maxlen=_BUF_SZ)
_wy_buf_lead = deque(maxlen=_BUF_SZ); _wy_buf_rear = deque(maxlen=_BUF_SZ)

_prev_v_lead = 0.0; _prev_v_rear = 0.0
_punch_cd    = {'lead':0.0,'rear':0.0}

PUNCH_CD  = 0.55
VEL_START = 0.15
DOM_RATIO = 1.35

_prev_lead_wr = None; _prev_rear_wr = None
_report       = None; REPORT_DUR = 4.0
_posture_data = None
_flash        = None
_show_ghost   = True
_voice_on     = True
_snap_count   = 0
_pose_stable  = 0
_pose_ready   = False
READY_FRAMES  = 20

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

# ══════════════════════════════════════════════════════
# 펀치 감지 (원본 동일)
# ══════════════════════════════════════════════════════
def classify_from_buf(side, sw, el_buf, ea_buf, wy_buf):
    min_el = min(el_buf) if el_buf else 1.0
    min_ea = min(ea_buf) if ea_buf else 180.0
    wy_list = list(wy_buf)
    if wy_list:
        rise = (wy_list[0] - min(wy_list)) / (sw + 1e-6)
        if rise > 0.20 and min_ea < 120.0 and min_el > 0.15:
            return 'uppercut'
    hook_frames = sum(1 for el,ea in zip(el_buf,ea_buf) if el < 0.05 and ea < 110.0)
    if hook_frames >= 3: return 'hook'
    return 'jab' if side == 'lead' else 'cross'

def update_punch_detect(kp, sc, now):
    global _prev_v_lead, _prev_v_rear, _prev_lead_wr, _prev_rear_wr
    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()
    if sc[JAB_WR] < VIS_MIN or sc[CROSS_WR] < VIS_MIN: return
    jx,jy   = kp[JAB_WR][0],   kp[JAB_WR][1]
    cx_,cy_ = kp[CROSS_WR][0], kp[CROSS_WR][1]
    _trail_lead.append((jx,jy)); _trail_rear.append((cx_,cy_))
    sw = sw_px(kp)
    v_lead = (math.sqrt((jx-_prev_lead_wr[0])**2+(jy-_prev_lead_wr[1])**2)/sw if _prev_lead_wr else 0.0)
    v_rear = (math.sqrt((cx_-_prev_rear_wr[0])**2+(cy_-_prev_rear_wr[1])**2)/sw if _prev_rear_wr else 0.0)
    jel = (kp[JAB_EL][1]-kp[JAB_SH][1])/sw   if sc[JAB_EL]>VIS_MIN else 1.0
    rel = (kp[CROSS_EL][1]-kp[CROSS_SH][1])/sw if sc[CROSS_EL]>VIS_MIN else 1.0
    jea = (angle3pt(kp[JAB_SH][0],kp[JAB_SH][1],kp[JAB_EL][0],kp[JAB_EL][1],kp[JAB_WR][0],kp[JAB_WR][1]) if sc[JAB_EL]>VIS_MIN else 180.0)
    rea = (angle3pt(kp[CROSS_SH][0],kp[CROSS_SH][1],kp[CROSS_EL][0],kp[CROSS_EL][1],kp[CROSS_WR][0],kp[CROSS_WR][1]) if sc[CROSS_EL]>VIS_MIN else 180.0)
    _v_buf_lead.append(v_lead);  _v_buf_rear.append(v_rear)
    _el_buf_lead.append(jel);    _el_buf_rear.append(rel)
    _ea_buf_lead.append(jea);    _ea_buf_rear.append(rea)
    _wy_buf_lead.append(jy);     _wy_buf_rear.append(cy_)
    peak_lead = max(_v_buf_lead) if _v_buf_lead else 0.0
    peak_rear = max(_v_buf_rear) if _v_buf_rear else 0.0
    if (_prev_v_lead > VEL_START and v_lead <= VEL_START and
            peak_lead > peak_rear * DOM_RATIO and now - _punch_cd['lead'] > PUNCH_CD):
        pt = classify_from_buf('lead', sw, _el_buf_lead, _ea_buf_lead, _wy_buf_lead)
        _last_pt['lead'] = pt; _punch_cd['lead'] = now
        _pending.append([EXT_DELAY, pt, 'lead'])
    if (_prev_v_rear > VEL_START and v_rear <= VEL_START and
            peak_rear > peak_lead * DOM_RATIO and now - _punch_cd['rear'] > PUNCH_CD):
        pt = classify_from_buf('rear', sw, _el_buf_rear, _ea_buf_rear, _wy_buf_rear)
        _last_pt['rear'] = pt; _punch_cd['rear'] = now
        _pending.append([EXT_DELAY, pt, 'rear'])
    _prev_v_lead = v_lead; _prev_v_rear = v_rear
    _prev_lead_wr = (jx,jy); _prev_rear_wr = (cx_,cy_)

def analyse_punch(punch_type, side):
    global _report
    buf = list(_lm_buf)
    if len(buf) < 3 or punch_type not in PUNCH_DNA: return
    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()
    wr_i = JAB_WR if side=='lead' else CROSS_WR
    sh_i = JAB_SH if side=='lead' else CROSS_SH
    el_i = JAB_EL if side=='lead' else CROSS_EL
    best_kp, best_d = buf[-1][0], 0.0
    for kp,sc in buf:
        if sc[wr_i]<VIS_MIN or sc[sh_i]<VIS_MIN: continue
        d = math.sqrt((kp[wr_i][0]-kp[sh_i][0])**2+(kp[wr_i][1]-kp[sh_i][1])**2)
        if d > best_d: best_d=d; best_kp=kp
    kp = best_kp; sw = sw_px(kp)
    sh_cx = (kp[KP_L_SH][0]+kp[KP_R_SH][0])/2
    hi_ok = kp[KP_L_HI][0]>0 and kp[KP_R_HI][0]>0
    lean  = (sh_cx-(kp[KP_L_HI][0]+kp[KP_R_HI][0])/2)/sw if hi_ok else 0.0
    _report = {
        'type': punch_type,
        'arm_ext': math.sqrt((kp[wr_i][0]-kp[sh_i][0])**2+(kp[wr_i][1]-kp[sh_i][1])**2)/sw,
        'el_ang': angle3pt(kp[sh_i][0],kp[sh_i][1],kp[el_i][0],kp[el_i][1],kp[wr_i][0],kp[wr_i][1]),
        'lean': lean, 'time': time.time(),
    }
    _count[punch_type] += 1

# ══════════════════════════════════════════════════════
# 자세 점수
# ══════════════════════════════════════════════════════
def calc_posture(kp, sc, sw):
    issues = []
    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()
    sh_y = (kp[KP_L_SH][1]+kp[KP_R_SH][1])/2
    l_ydiff = (kp[JAB_WR][1]  - kp[JAB_SH][1])  / sw
    r_ydiff = (kp[CROSS_WR][1]- kp[CROSS_SH][1]) / sw
    ref_l = POSE_DNA['guard_l_ydiff']; ref_r = POSE_DNA['guard_r_ydiff']
    tol_g = 0.25
    def _partial(err, mx):
        if abs(err) <= tol_g: return mx
        return max(0, int(mx * max(0.0, 1.0-(abs(err)-tol_g)/tol_g)))
    l_err = l_ydiff - ref_l; r_err = r_ydiff - ref_r
    l_score = _partial(l_err, 18); r_score = _partial(r_err, 17)
    guard_score = l_score + r_score
    if l_score < 18:
        issues.append((0,'잽 가드 올려' if l_err>0 else '잽 가드 내려', 18-l_score))
    if r_score < 17:
        issues.append((0,'크로스 가드 올려' if r_err>0 else '크로스 가드 내려', 17-r_score))
    hi_ok = sc[KP_L_HI]>VIS_MIN and sc[KP_R_HI]>VIS_MIN
    lean_score = 0; lean_msg = '엉덩이 미감지'
    if hi_ok:
        sh_cx = (kp[KP_L_SH][0]+kp[KP_R_SH][0])/2
        hi_cx = (kp[KP_L_HI][0]+kp[KP_R_HI][0])/2
        lean  = (sh_cx - hi_cx) / sw
        lean_err = lean - POSE_DNA.get('lean_forward', 0.05)
        tol_lean = 0.12
        def _lean_p(err, mx):
            if abs(err) <= tol_lean: return mx
            return max(0, int(mx * max(0.0, 1.0-(abs(err)-tol_lean)/tol_lean)))
        lean_score = _lean_p(lean_err, 25)
        if lean_score < 25:
            lean_msg = '앞으로 기울여' if lean_err < 0 else '상체 세워'
            issues.append((1, lean_msg, 25-lean_score))
        else:
            lean_msg = f'기울기 좋아! ({lean:+.2f})'
    head_y   = (kp[KP_NOSE][1]-sh_y)/sw
    head_err = head_y - POSE_DNA['head_y_ratio']
    head_score = 20 if abs(head_err)<0.18 else 10 if abs(head_err)<0.30 else 0
    head_msg = '머리 좋아!' if abs(head_err)<0.18 else ('고개 들어!' if head_err>0 else '턱 당겨!')
    if abs(head_err) >= 0.18:
        issues.append((2, head_msg, 20-head_score))
    l_el_y = (kp[JAB_EL][1]-kp[JAB_SH][1])/sw
    r_el_y = (kp[CROSS_EL][1]-kp[CROSS_SH][1])/sw
    el_ok_l = (l_el_y > l_ydiff) and (l_el_y < 0.60)
    el_ok_r = (r_el_y > r_ydiff) and (r_el_y < 0.60)
    elbow_score = (10 if el_ok_l else 0) + (10 if el_ok_r else 0)
    el_msg = '팔꿈치 좋아!' if el_ok_l and el_ok_r else ('잽 팔꿈치 조정' if not el_ok_l else '크로스 팔꿈치 조정')
    total = guard_score + lean_score + head_score + elbow_score
    grade = 'S' if total>=85 else 'A' if total>=70 else 'B' if total>=50 else 'C'
    items = [
        ('가드',     guard_score,  35, '올려!' if guard_score<35 else '좋아!'),
        ('린포워드', lean_score,   25, lean_msg),
        ('머리',     head_score,   20, head_msg),
        ('팔꿈치',   elbow_score,  20, el_msg),
    ]
    if issues:
        speak(min(issues, key=lambda x:x[0])[1])
    else:
        speak('자세가 좋아요')
    return total, grade, items

# ══════════════════════════════════════════════════════
# PIL 헬퍼
# ══════════════════════════════════════════════════════
def bgr2pil(img):
    return PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil2bgr(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_text_pil(img, text, pos, font, color_rgb):
    pil = bgr2pil(img)
    ImageDraw.Draw(pil).text(pos, text, font=font, fill=color_rgb)
    img[:] = pil2bgr(pil)

def rgb(bgr):
    return (bgr[2], bgr[1], bgr[0])

# ══════════════════════════════════════════════════════
# 카드 (둥근 사각형)
# ══════════════════════════════════════════════════════
def draw_card(img, x, y, w, h, color=C_CANVAS, border=C_BORDER, radius=12):
    overlay = img.copy()
    cv2.rectangle(overlay, (x+radius, y), (x+w-radius, y+h), color, -1)
    cv2.rectangle(overlay, (x, y+radius), (x+w, y+h-radius), color, -1)
    for cx, cy in [(x+radius,y+radius),(x+w-radius,y+radius),(x+radius,y+h-radius),(x+w-radius,y+h-radius)]:
        cv2.circle(overlay, (cx,cy), radius, color, -1)
    cv2.addWeighted(overlay, 1.0, img, 0.0, 0, img)
    # border
    cv2.rectangle(img, (x+radius,y), (x+w-radius,y), border, 1)
    cv2.rectangle(img, (x+radius,y+h), (x+w-radius,y+h), border, 1)
    cv2.rectangle(img, (x,y+radius), (x,y+h-radius), border, 1)
    cv2.rectangle(img, (x+w,y+radius), (x+w,y+h-radius), border, 1)

def draw_bar(img, x, y, w, h, ratio, color=C_ACCENT, bg=C_BORDER):
    cv2.rectangle(img, (x,y), (x+w,y+h), bg, -1)
    if ratio > 0:
        fw = max(0, int(w * min(ratio, 1.0)))
        cv2.rectangle(img, (x,y), (x+fw,y+h), color, -1)
    # rounded ends
    r = h // 2
    cv2.circle(img, (x+r, y+r), r, bg, -1)
    if ratio > 0 and fw > r:
        cv2.circle(img, (x+fw-r, y+r), r, color, -1)

# ══════════════════════════════════════════════════════
# 레이아웃 상수
# ══════════════════════════════════════════════════════
WIN_W, WIN_H = 1280, 720
CAM_W = 853        # 2/3
SB_X  = CAM_W + 1  # 사이드바 시작 X
SB_W  = WIN_W - SB_X  # ~426px
PAD   = 14

# ══════════════════════════════════════════════════════
# 스켈레톤
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
        cx,cy = int(ix),int(iy)
        cv2.circle(ov,(cx,cy),22,col,2); cv2.circle(ov,(cx,cy),5,col,-1)
        cv2.line(ov,(cx-13,cy),(cx+13,cy),col,1); cv2.line(ov,(cx,cy-13),(cx,cy+13),col,1)
    cv2.addWeighted(ov,0.4,cam,0.6,0,cam)

# ══════════════════════════════════════════════════════
# 사이드바 렌더러
# ══════════════════════════════════════════════════════
def render_sidebar(canvas, posture, now):
    x0 = SB_X + PAD
    cw = SB_W - PAD*2

    # ── 헤더 ─────────────────────────────────────────
    pil = bgr2pil(canvas)
    d   = ImageDraw.Draw(pil)
    d.text((x0, 12), "BOXING", font=F_DISPLAY, fill=rgb(C_TEXT))
    bbox = d.textbbox((0,0), "BOXING", font=F_DISPLAY)
    bx = x0 + (bbox[2]-bbox[0]) + 4
    d.text((bx, 12), ".", font=F_DISPLAY, fill=rgb(C_ACCENT))
    bx2 = bx + d.textlength(".", font=F_DISPLAY) + 2
    d.text((bx2, 12), "COACH", font=F_DISPLAY, fill=rgb(C_TEXT))
    canvas[:] = pil2bgr(pil)

    cy = 70

    # ── 코치 정보 카드 ───────────────────────────────
    draw_card(canvas, SB_X+PAD, cy, cw, 52)
    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((x0+10, cy+8),  "LIM KWANWOO", font=F_TITLE, fill=rgb(C_TEXT))
    d.text((x0+10, cy+34), f"{'우향 Orthodox →' if _facing_right else '← 좌향 Southpaw'}", font=F_SMALL, fill=rgb(C_ACCENT))
    canvas[:] = pil2bgr(pil)
    cy += 60

    # ── 펀치 카운트 카드 ─────────────────────────────
    draw_card(canvas, SB_X+PAD, cy, cw, 158)
    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((x0+10, cy+10), "PUNCH COUNT", font=F_LABEL, fill=rgb(C_DIM))
    canvas[:] = pil2bgr(pil)
    cy2 = cy + 36
    total_punches = sum(_count.values())
    labels = list(PUNCH_BGR.keys())
    for i, pt in enumerate(labels):
        col = PUNCH_BGR[pt]
        row = i // 2; col_i = i % 2
        bx_ = SB_X+PAD+10 + col_i*(cw//2)
        by_ = cy2 + row * 52
        draw_card(canvas, bx_-4, by_-4, cw//2-8, 44, color=(38,33,31), border=C_BORDER, radius=8)
        pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
        d.text((bx_+4,  by_+2),  pt.upper(), font=F_SMALL, fill=rgb(C_MUTED))
        d.text((bx_+4,  by_+18), f"{_count[pt]:03d}", font=F_TITLE, fill=rgb(col))
        canvas[:] = pil2bgr(pil)
    cy += 166

    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((x0+10, cy-6), f"총 {total_punches}회", font=F_SMALL, fill=rgb(C_DIM))
    canvas[:] = pil2bgr(pil)
    cy += 10

    # ── 자세 점수 카드 ───────────────────────────────
    if posture:
        total_s, grade, items = posture
        grade_col = C_GREEN if total_s>=85 else C_ACCENT if total_s>=70 else C_AMBER
        card_h = 36 + len(items)*42 + 10
        draw_card(canvas, SB_X+PAD, cy, cw, card_h)

        pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
        d.text((x0+10, cy+10), "POSTURE", font=F_LABEL, fill=rgb(C_DIM))
        score_txt = f"{total_s}/100"
        grade_txt = f"[{grade}]"
        d.text((x0+cw-90, cy+6),  score_txt, font=F_TITLE, fill=rgb(C_TEXT))
        d.text((x0+cw-30, cy+10), grade_txt, font=F_LABEL, fill=rgb(grade_col))
        canvas[:] = pil2bgr(pil)

        iy = cy + 38
        for label, score, mx, msg in items:
            ratio = score/mx if mx else 0
            bar_col = C_GREEN if ratio > 0.85 else C_ACCENT if ratio > 0.5 else C_AMBER
            pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
            d.text((x0+10, iy), label, font=F_SMALL, fill=rgb(C_MUTED))
            d.text((x0+cw-50, iy), f"{score}/{mx}", font=F_SMALL, fill=rgb(C_TEXT))
            canvas[:] = pil2bgr(pil)
            draw_bar(canvas, x0+10, iy+18, cw-20, 6, ratio, bar_col)
            pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
            d.text((x0+10, iy+26), msg, font=F_SMALL, fill=rgb(bar_col if ratio<0.85 else C_DIM))
            canvas[:] = pil2bgr(pil)
            iy += 42
        cy += card_h + 8

    # ── 펀치 리포트 ──────────────────────────────────
    if _report:
        age = now - _report['time']
        if age < REPORT_DUR:
            alpha = max(0.0, 1.0-age/REPORT_DUR)
            pt   = _report['type']
            col  = PUNCH_BGR.get(pt,(0,200,200))
            dna  = PUNCH_DNA.get(pt,{})
            rh   = 100
            if cy + rh < WIN_H - 50:
                draw_card(canvas, SB_X+PAD, cy, cw, rh)
                pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
                d.text((x0+10, cy+8), f"[ {pt.upper()} ]", font=F_TITLE, fill=rgb(col))
                metrics = [
                    ('팔 뻗음',  _report['arm_ext'], dna.get('arm_extension_avg',0), TOL_PUNCH['arm_extension']),
                    ('팔꿈치°',  _report['el_ang'],  dna.get('elbow_angle_avg',0),   TOL_PUNCH['elbow_angle']),
                ]
                for mi,(name,val,ref,tol) in enumerate(metrics):
                    ok = abs(val-ref)<=tol
                    icon = '✓' if ok else ('▲' if val<ref else '▼')
                    fc = C_GREEN if ok else C_AMBER
                    d.text((x0+10,  cy+36+mi*26), name, font=F_SMALL, fill=rgb(C_MUTED))
                    d.text((x0+100, cy+36+mi*26), f"{val:.2f}/{ref:.2f} {icon}", font=F_SMALL, fill=rgb(fc))
                canvas[:] = pil2bgr(pil)
                cy += rh + 8

    # ── 단축키 힌트 ──────────────────────────────────
    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    hints = [("R", "초기화"), ("D", "방향전환"), ("G", "가이드"), ("V", f"음성{'ON' if _voice_on else 'OFF'}"), ("Q", "종료")]
    hx = x0
    for key, label in hints:
        d.text((hx, WIN_H-28), key, font=F_SMALL, fill=rgb(C_ACCENT))
        tw = int(d.textlength(key, font=F_SMALL))
        d.text((hx+tw+3, WIN_H-28), label, font=F_SMALL, fill=rgb(C_DIM))
        hx += tw + int(d.textlength(label+" ", font=F_SMALL)) + 10
    canvas[:] = pil2bgr(pil)

# ══════════════════════════════════════════════════════
# 카메라 영역 펀치 플래시
# ══════════════════════════════════════════════════════
def draw_punch_flash(cam, now):
    if _flash and now < _flash[1]:
        col  = PUNCH_BGR.get(_flash[0], (0,200,200))
        age  = now - (_flash[1] - 0.18)
        alpha = max(0.0, 1.0 - age/0.18)
        ov = cam.copy()
        cv2.rectangle(ov, (0,0), (cam.shape[1]-1, cam.shape[0]-1), col, 8)
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

    # 타이틀
    d.text((WIN_W//2-200, WIN_H//2-160), "BOXING", font=F_DISPLAY, fill=rgb(C_TEXT))
    d.text((WIN_W//2-200+d.textlength("BOXING", font=F_DISPLAY)+4, WIN_H//2-160), ".", font=F_DISPLAY, fill=rgb(C_ACCENT))
    d.text((WIN_W//2-50, WIN_H//2-100), "COACH", font=F_TITLE, fill=rgb(C_MUTED))
    d.text((WIN_W//2-170, WIN_H//2-50), "LIM KWANWOO — 측면 카메라 모드", font=F_MEDIUM, fill=rgb(C_MUTED))

    canvas[:] = pil2bgr(pil)

    # 프로그레스 바
    if pose_ok:
        prog = min(stable/READY_FRAMES, 1.0)
        bx, by, bw_, bh_ = WIN_W//2-180, WIN_H//2+20, 360, 8
        draw_bar(canvas, bx, by, bw_, bh_, prog, C_ACCENT)
        pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
        d.text((bx, by-24), f"자세 인식 중… {int(prog*100)}%", font=F_MEDIUM, fill=rgb(C_GREEN))
        canvas[:] = pil2bgr(pil)
    else:
        pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
        d.text((WIN_W//2-130, WIN_H//2+20), "전신이 보이도록 서주세요", font=F_MEDIUM, fill=rgb(C_DIM))
        canvas[:] = pil2bgr(pil)

    # 힌트
    tips = ["측면 카메라 | 오르토독스 스탠스", "D키 방향전환  G가이드  V음성  R초기화  Q종료"]
    pil = bgr2pil(canvas); d = ImageDraw.Draw(pil)
    for i,t in enumerate(tips):
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

    # 카메라는 원본 해상도, 캔버스는 1280x720
    fh, fw = frame.shape[:2]
    cam_frame = cv2.resize(frame, (CAM_W, WIN_H))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    kps, scs  = pose_model(rgb_frame)

    pose_ok = len(kps) > 0
    kp = kps[0] if pose_ok else None
    sc = scs[0] if pose_ok else None

    # 키포인트 좌표를 cam_frame 스케일로 변환
    if pose_ok:
        pose_ok = all(sc[i] > VIS_MIN for i in NEEDED_KP)
        if pose_ok:
            scale_x = CAM_W / fw
            scale_y = WIN_H / fh
            kp_disp = kp.copy().astype(float)
            kp_disp[:,0] *= scale_x
            kp_disp[:,1] *= scale_y
        else:
            kp_disp = None
    else:
        kp_disp = None

    sw = sw_px(kp_disp) if kp_disp is not None else None

    # ── 캔버스 구성 ──────────────────────────────────
    canvas = np.full((WIN_H, WIN_W, 3), C_INK, dtype=np.uint8)

    if not _pose_ready:
        if pose_ok: _pose_stable = min(_pose_stable+1, READY_FRAMES)
        else:       _pose_stable = max(_pose_stable-2, 0)
        draw_ready_screen(canvas, pose_ok, _pose_stable)
        if _pose_stable >= READY_FRAMES: _pose_ready = True
        cv2.imshow('BOXING COACH', canvas)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
        continue

    # ── 메인 게임 로직 ───────────────────────────────
    if pose_ok and kp_disp is not None:
        _lm_buf.append((kp, sc))
        for entry in list(_pending):
            entry[0] -= 1
            if entry[0] <= 0:
                _pending.remove(entry)
                analyse_punch(entry[1], entry[2])
                _flash = (entry[1], now+0.18)
        update_punch_detect(kp, sc, now)
        _posture_data = calc_posture(kp_disp, sc, sw)
    else:
        _pose_stable = max(_pose_stable-2, 0)
        if _pose_stable < READY_FRAMES//2: _pose_ready = False

    # ── 카메라 영역 그리기 ───────────────────────────
    canvas[:, :CAM_W] = cam_frame
    if kp_disp is not None:
        draw_skeleton(canvas[:, :CAM_W], kp_disp, sc)
        draw_trail(canvas[:, :CAM_W])
        if _show_ghost: draw_ghost(canvas[:, :CAM_W], kp_disp, sw)
    draw_punch_flash(canvas[:, :CAM_W], now)

    # 구분선
    cv2.line(canvas, (CAM_W, 0), (CAM_W, WIN_H), C_BORDER, 1)

    # ── 사이드바 ─────────────────────────────────────
    render_sidebar(canvas, _posture_data, now)

    cv2.imshow('BOXING COACH', canvas)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27): break
    elif key == ord('r'):
        for t in _count: _count[t] = 0
        _report = None
        _posture_data = None
    elif key == ord('g'): _show_ghost = not _show_ghost
    elif key == ord('v'): _voice_on = not _voice_on
    elif key == ord('d'):
        _facing_right = not _facing_right
        _prev_lead_wr = None; _prev_rear_wr = None
        _v_buf_lead.clear(); _v_buf_rear.clear()
    elif key == ord('s'):
        _snap_count += 1
        fn = os.path.join(BASE_DIR, f'coach_snap_{_snap_count:03d}.png')
        cv2.imwrite(fn, canvas)
        print(f'스냅샷: {fn}')

cap.release()
cv2.destroyAllWindows()
