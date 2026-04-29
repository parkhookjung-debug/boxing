"""
LIM coach 4.py — 측면 카메라 전용 4방향 펀치 코치 + 음성 피드백
─────────────────────────────────────────────────────────────────
RTMPose (RTMO-s) | 측면 카메라 | 오르토독스 스탠스

● 잽 / 크로스 / 훅 / 어퍼컷 자동 감지 + 카운트
● LIM DNA 기준 (측면 데이터 1,2,5) 타격 폼 피드백
● 음성 코치 — 5초마다 가장 중요한 교정 사항
● 측면 특화: 상체 전방 기울기(린 포워드) 점수 포함

데이터 준비 순서
  1. LIM data extraction.py
  2. LIM master average.py          → LIM_DNA.csv
  3. LIM punch extraction side.py   → LIM_punch_DNA_side.csv

단축키
  Q / ESC   종료
  R         카운터 초기화
  G         가이드 토글
  V         음성 ON/OFF
  D         방향 전환 (우향/좌향)
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

# ══════════════════════════════════════════════════════════════════
# TTS
# ══════════════════════════════════════════════════════════════════
_tts_ok = False; _tts_busy = False; _last_tts = 0.0; TTS_INTERVAL = 5.0
try:
    import pyttsx3 as _p3; _tts_ok = True; print("[TTS] pyttsx3 로드 완료")
except ImportError:
    print("[TTS] 비활성 — pip install pyttsx3")

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
        except Exception as ex: print(f"[TTS] {ex}")
        finally: _tts_busy = False
    threading.Thread(target=_run, daemon=True).start()

# ══════════════════════════════════════════════════════════════════
# RTMPose 모델
# ══════════════════════════════════════════════════════════════════
RTMO_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
    'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip'
)
print("RTMPose (RTMO-s) 로드 중...")
pose_model = RTMO(RTMO_URL, backend='onnxruntime', device='cpu')
print("모델 준비 완료")

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
VIS_MIN  = 0.30
NEEDED_KP = [KP_L_SH, KP_R_SH, KP_L_WR, KP_R_WR, KP_L_EL, KP_R_EL, KP_NOSE]

# ── 방향 설정 (D키로 토글) ────────────────────────────────────────
# 우향(facing_right=True): 오른손(KP_R_WR)이 앞손(잽), 왼손이 뒷손(크로스)
# 좌향(facing_right=False): 반대
_facing_right = True

def get_arm_indices():
    """현재 방향에 따라 (JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL) 반환"""
    if _facing_right:
        return KP_R_WR, KP_R_SH, KP_R_EL, KP_L_WR, KP_L_SH, KP_L_EL
    return KP_L_WR, KP_L_SH, KP_L_EL, KP_R_WR, KP_R_SH, KP_R_EL

# ══════════════════════════════════════════════════════════════════
# DNA 로드
# ══════════════════════════════════════════════════════════════════
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
print(f"[DNA] {os.path.basename(_dna_src)} — {list(PUNCH_DNA.keys())}")

TOL_PUNCH = {'arm_extension':0.12,'elbow_angle':28.0,'lean_forward':0.10}

# ══════════════════════════════════════════════════════════════════
# 폰트
# ══════════════════════════════════════════════════════════════════
def _font(sz):
    for p in ["C:/Windows/Fonts/malgun.ttf","C:/Windows/Fonts/gulim.ttc"]:
        try: return ImageFont.truetype(p,sz)
        except: pass
    return ImageFont.load_default()

F_SM=_font(16); F_MD=_font(24); F_LG=_font(38); F_XL=_font(60)

def put_kr(img,text,pos,col,font=None):
    font=font or F_MD
    pil=PILImage.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(pos,text,font=font,fill=(col[2],col[1],col[0]))
    img[:]=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)

# ══════════════════════════════════════════════════════════════════
# 기하 헬퍼
# ══════════════════════════════════════════════════════════════════
def sw_px(kp):
    dx=kp[KP_R_SH][0]-kp[KP_L_SH][0]; dy=kp[KP_R_SH][1]-kp[KP_L_SH][1]
    return math.sqrt(dx*dx+dy*dy)+1e-6

def angle3pt(ax,ay,bx,by,cx,cy):
    bax,bay=ax-bx,ay-by; bcx,bcy=cx-bx,cy-by
    dot=bax*bcx+bay*bcy
    mag=math.sqrt(bax**2+bay**2)*math.sqrt(bcx**2+bcy**2)+1e-9
    return math.degrees(math.acos(max(-1,min(1,dot/mag))))

def draw_skeleton(frame, kp, sc):
    for a,b in COCO_CONN:
        if sc[a]>VIS_MIN and sc[b]>VIS_MIN:
            cv2.line(frame,(int(kp[a][0]),int(kp[a][1])),
                     (int(kp[b][0]),int(kp[b][1])),(80,80,80),2)
    for i in range(17):
        if sc[i]>VIS_MIN:
            cv2.circle(frame,(int(kp[i][0]),int(kp[i][1])),4,(0,200,255),-1)

# ══════════════════════════════════════════════════════════════════
# 상태 변수
# ══════════════════════════════════════════════════════════════════
PUNCH_COLS = {
    'jab':(0,220,255),'cross':(255,160,40),
    'hook':(200,50,255),'uppercut':(100,255,50),
}
_count      = {t:0 for t in PUNCH_COLS}
_trail_lead = deque(maxlen=22)
_trail_rear = deque(maxlen=22)
_last_pt    = {'lead':'jab','rear':'cross'}

_lm_buf  = deque(maxlen=30)
_pending = []
EXT_DELAY = 8

_BUF_SZ      = 12
_v_buf_lead  = deque(maxlen=_BUF_SZ)
_v_buf_rear  = deque(maxlen=_BUF_SZ)
_el_buf_lead = deque(maxlen=_BUF_SZ)
_el_buf_rear = deque(maxlen=_BUF_SZ)
_ea_buf_lead = deque(maxlen=_BUF_SZ)
_ea_buf_rear = deque(maxlen=_BUF_SZ)
_wy_buf_lead = deque(maxlen=_BUF_SZ)
_wy_buf_rear = deque(maxlen=_BUF_SZ)

_prev_v_lead = 0.0; _prev_v_rear = 0.0
_punch_cd    = {'lead':0.0,'rear':0.0}

PUNCH_CD  = 0.55
VEL_START = 0.15
DOM_RATIO = 1.35

HOOK_ELBOW_THRESH = 0.05
HOOK_ELBOW_ANGLE  = 110.0
HOOK_MIN_FRAMES   = 3
UPPER_RISE_THRESH = 0.20
UPPER_ELBOW_ANGLE = 120.0
UPPER_ELBOW_LOW   = 0.15

_prev_lead_wr  = None
_prev_rear_wr  = None
_dbg_vel       = {'lead':0.0,'rear':0.0}

_report    = None
REPORT_DUR = 4.0

READY_FRAMES = 20
_pose_stable = 0
_pose_ready  = False
_show_ghost  = True
_voice_on    = True
_snap_count  = 0

# ══════════════════════════════════════════════════════════════════
# 펀치 분류
# ══════════════════════════════════════════════════════════════════
def classify_from_buf(side, sw, el_buf, ea_buf, wy_buf):
    min_el = min(el_buf) if el_buf else 1.0
    min_ea = min(ea_buf) if ea_buf else 180.0

    wy_list = list(wy_buf)
    if wy_list:
        rise = (wy_list[0] - min(wy_list)) / (sw + 1e-6)
        if rise > UPPER_RISE_THRESH and min_ea < UPPER_ELBOW_ANGLE and min_el > UPPER_ELBOW_LOW:
            return 'uppercut'

    hook_frames = sum(1 for el,ea in zip(el_buf,ea_buf)
                      if el < HOOK_ELBOW_THRESH and ea < HOOK_ELBOW_ANGLE)
    if hook_frames >= HOOK_MIN_FRAMES:
        return 'hook'

    return 'jab' if side == 'lead' else 'cross'

# ══════════════════════════════════════════════════════════════════
# 펀치 감지 — falling-edge 기반
# ══════════════════════════════════════════════════════════════════
def update_punch_detect(kp, sc, now):
    global _prev_v_lead, _prev_v_rear, _prev_lead_wr, _prev_rear_wr

    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()

    if sc[JAB_WR] < VIS_MIN or sc[CROSS_WR] < VIS_MIN: return

    jx,jy   = kp[JAB_WR][0],   kp[JAB_WR][1]
    cx_,cy_ = kp[CROSS_WR][0], kp[CROSS_WR][1]

    _trail_lead.append((jx, jy))
    _trail_rear.append((cx_, cy_))

    sw = sw_px(kp)
    v_lead = (math.sqrt((jx-_prev_lead_wr[0])**2+(jy-_prev_lead_wr[1])**2)/sw
              if _prev_lead_wr else 0.0)
    v_rear = (math.sqrt((cx_-_prev_rear_wr[0])**2+(cy_-_prev_rear_wr[1])**2)/sw
              if _prev_rear_wr else 0.0)
    _dbg_vel['lead'] = v_lead; _dbg_vel['rear'] = v_rear

    jel = (kp[JAB_EL][1]-kp[JAB_SH][1])/sw   if sc[JAB_EL]>VIS_MIN   else 1.0
    rel = (kp[CROSS_EL][1]-kp[CROSS_SH][1])/sw if sc[CROSS_EL]>VIS_MIN else 1.0

    jea = (angle3pt(kp[JAB_SH][0],kp[JAB_SH][1], kp[JAB_EL][0],kp[JAB_EL][1],
                    kp[JAB_WR][0],kp[JAB_WR][1])
           if sc[JAB_EL]>VIS_MIN else 180.0)
    rea = (angle3pt(kp[CROSS_SH][0],kp[CROSS_SH][1], kp[CROSS_EL][0],kp[CROSS_EL][1],
                    kp[CROSS_WR][0],kp[CROSS_WR][1])
           if sc[CROSS_EL]>VIS_MIN else 180.0)

    _v_buf_lead.append(v_lead);  _v_buf_rear.append(v_rear)
    _el_buf_lead.append(jel);    _el_buf_rear.append(rel)
    _ea_buf_lead.append(jea);    _ea_buf_rear.append(rea)
    _wy_buf_lead.append(jy);     _wy_buf_rear.append(cy_)

    peak_lead = max(_v_buf_lead) if _v_buf_lead else 0.0
    peak_rear = max(_v_buf_rear) if _v_buf_rear else 0.0

    if (_prev_v_lead > VEL_START and v_lead <= VEL_START and
            peak_lead > peak_rear * DOM_RATIO and
            now - _punch_cd['lead'] > PUNCH_CD):
        pt = classify_from_buf('lead', sw, _el_buf_lead, _ea_buf_lead, _wy_buf_lead)
        _last_pt['lead'] = pt; _punch_cd['lead'] = now
        _pending.append([EXT_DELAY, pt, 'lead'])

    if (_prev_v_rear > VEL_START and v_rear <= VEL_START and
            peak_rear > peak_lead * DOM_RATIO and
            now - _punch_cd['rear'] > PUNCH_CD):
        pt = classify_from_buf('rear', sw, _el_buf_rear, _ea_buf_rear, _wy_buf_rear)
        _last_pt['rear'] = pt; _punch_cd['rear'] = now
        _pending.append([EXT_DELAY, pt, 'rear'])

    _prev_v_lead  = v_lead;  _prev_v_rear  = v_rear
    _prev_lead_wr = (jx,jy); _prev_rear_wr = (cx_,cy_)

# ══════════════════════════════════════════════════════════════════
# 폼 분석
# ══════════════════════════════════════════════════════════════════
def analyse_punch(punch_type, side):
    global _report
    buf = list(_lm_buf)
    if len(buf) < 3: return
    if punch_type not in PUNCH_DNA: return

    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()
    wr_i = JAB_WR if side=='lead' else CROSS_WR
    sh_i = JAB_SH if side=='lead' else CROSS_SH
    el_i = JAB_EL if side=='lead' else CROSS_EL

    best_kp, best_d = buf[-1][0], 0.0
    for kp,sc in buf:
        if sc[wr_i]<VIS_MIN or sc[sh_i]<VIS_MIN: continue
        d = math.sqrt((kp[wr_i][0]-kp[sh_i][0])**2+(kp[wr_i][1]-kp[sh_i][1])**2)
        if d > best_d: best_d=d; best_kp=kp

    kp  = best_kp
    sw  = sw_px(kp)
    sh_cx = (kp[KP_L_SH][0]+kp[KP_R_SH][0])/2
    hi_ok = kp[KP_L_HI][0]>0 and kp[KP_R_HI][0]>0
    lean  = (sh_cx-(kp[KP_L_HI][0]+kp[KP_R_HI][0])/2)/sw if hi_ok else 0.0

    _report = {
        'type'   : punch_type,
        'arm_ext': math.sqrt((kp[wr_i][0]-kp[sh_i][0])**2+(kp[wr_i][1]-kp[sh_i][1])**2)/sw,
        'el_ang' : angle3pt(kp[sh_i][0],kp[sh_i][1],kp[el_i][0],kp[el_i][1],
                            kp[wr_i][0],kp[wr_i][1]),
        'lean'   : lean,
        'time'   : time.time(),
    }
    _count[punch_type] += 1

# ══════════════════════════════════════════════════════════════════
# 자세 점수 — 측면 특화
# ══════════════════════════════════════════════════════════════════
def calc_posture(kp, sc, sw):
    issues = []
    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()
    sh_y = (kp[KP_L_SH][1]+kp[KP_R_SH][1])/2

    # ── 가드 높이 (35점) ─────────────────────────────────────────
    l_ydiff = (kp[JAB_WR][1]  - kp[JAB_SH][1])  / sw
    r_ydiff = (kp[CROSS_WR][1]- kp[CROSS_SH][1]) / sw
    ref_l = POSE_DNA['guard_l_ydiff']; ref_r = POSE_DNA['guard_r_ydiff']
    tol_g = 0.25

    l_err = l_ydiff - ref_l; r_err = r_ydiff - ref_r

    def _partial(err, mx):
        if abs(err) <= tol_g: return mx
        return max(0, int(mx * max(0.0, 1.0 - (abs(err)-tol_g)/tol_g)))

    l_score = _partial(l_err, 18); r_score = _partial(r_err, 17)
    guard_score = l_score + r_score

    if l_score < 18:
        issues.append((0, '잽 가드 올려' if l_err>0 else '잽 가드 내려',
                       '잽손 올려!' if l_err>0 else '잽손 내려!', 18-l_score, (0,60,255)))
    if r_score < 17:
        issues.append((0, '크로스 가드 올려' if r_err>0 else '크로스 가드 내려',
                       '크로스손 올려!' if r_err>0 else '크로스손 내려!', 17-r_score, (0,60,255)))

    l_ok = l_score==18; r_ok = r_score==17
    if l_ok and r_ok:       guard_msg='Guard 좋아!'; guard_col=(0,220,100)
    elif not l_ok and not r_ok: guard_msg='양손 가드 조정'; guard_col=(0,60,255)
    elif not l_ok:  guard_msg=('잽손 가드 내려감' if l_err>0 else '잽손 너무 높음'); guard_col=(0,60,255)
    else:           guard_msg=('크로스손 가드 내려감' if r_err>0 else '크로스손 너무 높음'); guard_col=(0,60,255)

    # ── 린 포워드 (25점) — 측면 카메라 핵심 지표 ─────────────────
    hi_ok = sc[KP_L_HI]>VIS_MIN and sc[KP_R_HI]>VIS_MIN
    lean_score = 0; lean_msg = '엉덩이 미감지'; lean_col = (120,120,120)
    if hi_ok:
        sh_cx = (kp[KP_L_SH][0]+kp[KP_R_SH][0])/2
        hi_cx = (kp[KP_L_HI][0]+kp[KP_R_HI][0])/2
        lean  = (sh_cx - hi_cx) / sw
        ref_l_fwd = POSE_DNA.get('lean_forward', 0.05)
        lean_err  = lean - ref_l_fwd
        tol_lean  = 0.12

        def _lean_partial(err, mx):
            if abs(err) <= tol_lean: return mx
            return max(0, int(mx * max(0.0, 1.0 - (abs(err)-tol_lean)/tol_lean)))

        lean_score = _lean_partial(lean_err, 25)
        if lean_score == 25:
            lean_msg = f'상체 기울기 좋아! ({lean:+.2f})'; lean_col=(0,220,100)
        elif lean_err < 0:
            lean_msg = f'앞으로 더 기울여 ({lean:+.2f})'; lean_col=(0,165,255)
            issues.append((1,'앞으로 기울여',lean_msg,25-lean_score,(0,165,255)))
        else:
            lean_msg = f'상체 너무 앞으로 ({lean:+.2f})'; lean_col=(0,165,255)
            issues.append((1,'상체 세워',lean_msg,25-lean_score,(0,165,255)))

    # ── 머리 자세 (20점) ─────────────────────────────────────────
    head_y   = (kp[KP_NOSE][1]-sh_y)/sw
    head_err = head_y - POSE_DNA['head_y_ratio']
    head_score = 20 if abs(head_err)<0.18 else 10 if abs(head_err)<0.30 else 0
    if abs(head_err) < 0.18:
        head_msg='머리 자세 좋아!'; head_col=(0,220,100)
    elif head_err > 0:
        head_msg='고개 들어!'; head_col=(0,165,255)
        issues.append((2,'고개 드세요','고개 들어!',20-head_score,(0,165,255)))
    else:
        head_msg='턱 당겨!'; head_col=(0,165,255)
        issues.append((2,'턱 당기세요','턱 당겨!',20-head_score,(0,165,255)))

    # ── 팔꿈치 (20점) ────────────────────────────────────────────
    l_el_y = (kp[JAB_EL][1]  - kp[JAB_SH][1])  / sw
    r_el_y = (kp[CROSS_EL][1]- kp[CROSS_SH][1]) / sw
    el_ok_l = (l_el_y > l_ydiff) and (l_el_y < 0.60)
    el_ok_r = (r_el_y > r_ydiff) and (r_el_y < 0.60)
    elbow_score = (10 if el_ok_l else 0) + (10 if el_ok_r else 0)
    if el_ok_l and el_ok_r:
        el_msg='팔꿈치 자세 좋아!'; el_col=(0,220,100)
    elif not el_ok_l:
        if l_el_y >= 0.60: el_msg='잽 팔꿈치 올려!'; issues.append((3,'잽 팔꿈치 올려','팔꿈치 올려!',10,(0,165,255)))
        else:              el_msg='잽 팔꿈치 내려!'; issues.append((3,'잽 팔꿈치 내려','팔꿈치 내려!',10,(0,165,255)))
        el_col=(0,165,255)
    else:
        if r_el_y >= 0.60: el_msg='크로스 팔꿈치 올려!'; issues.append((3,'크로스 팔꿈치 올려','팔꿈치 올려!',10,(0,165,255)))
        else:              el_msg='크로스 팔꿈치 내려!'; issues.append((3,'크로스 팔꿈치 내려','팔꿈치 내려!',10,(0,165,255)))
        el_col=(0,165,255)

    total = guard_score + lean_score + head_score + elbow_score
    if total >= 85:   grade,gcol = 'S',(0,255,120)
    elif total >= 70: grade,gcol = 'A',(0,200,255)
    elif total >= 50: grade,gcol = 'B',(0,165,255)
    else:             grade,gcol = 'C',(0,60,255)

    items = [
        (guard_score,  35, '가드',    guard_msg, guard_col),
        (lean_score,   25, '린포워드', lean_msg,  lean_col),
        (head_score,   20, '머리',    head_msg,  head_col),
        (elbow_score,  20, '팔꿈치',  el_msg,    el_col),
    ]
    if issues:
        speak(min(issues, key=lambda x:x[0])[1])
    else:
        speak('자세가 좋아요')

    return total, grade, gcol, items

# ══════════════════════════════════════════════════════════════════
# 그리기
# ══════════════════════════════════════════════════════════════════
def draw_trail(frame):
    for trail, side in [(_trail_lead,'lead'),(_trail_rear,'rear')]:
        col = PUNCH_COLS.get(_last_pt[side],(0,200,200))
        pts = list(trail); n = len(pts)
        if n < 2: continue
        for i in range(1, n):
            a = i/n; c = tuple(int(x*a) for x in col)
            cv2.line(frame,(int(pts[i-1][0]),int(pts[i-1][1])),
                     (int(pts[i][0]),int(pts[i][1])),c,max(1,int(4*a)))
        cv2.circle(frame,(int(pts[-1][0]),int(pts[-1][1])),6,col,-1)

def draw_counter(frame, w):
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(w,80),(8,8,20),-1)
    cv2.addWeighted(ov,0.7,frame,0.3,0,frame)
    pil  = PILImage.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    x = 10
    for pt, col_bgr in PUNCH_COLS.items():
        r,g,b = col_bgr[2],col_bgr[1],col_bgr[0]
        draw.text((x,6), f"{pt[:2].upper()} {_count[pt]:03d}", font=F_LG, fill=(r,g,b))
        x += w//4
    frame[:]=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)

def draw_score_panel(frame, total, grade, gcol, items):
    h = frame.shape[0]
    px,py = 10, h-290
    ov = frame.copy()
    cv2.rectangle(ov,(px-5,py-5),(px+255,h-5),(15,15,25),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)

    y = py
    for score,mx,_,_,col in items:
        bw=160
        cv2.rectangle(frame,(px,y),(px+bw,y+13),(50,50,60),-1)
        cv2.rectangle(frame,(px,y),(px+int(bw*score/mx),y+13),col,-1)
        y += 35
    cv2.putText(frame,f"TOTAL: {total}/100  [{grade}]",
                (px,y+16),cv2.FONT_HERSHEY_SIMPLEX,0.72,gcol,2)

    pil  = PILImage.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    y = py
    for score,mx,label,_,_ in items:
        draw.text((px,y-20), f'{label}: {score}/{mx}', font=F_SM, fill=(200,200,200))
        y += 35
    my = h-5-len(items)*26
    for i,(_,_,_,msg,col) in enumerate(items):
        b,g,r = col
        draw.text((px,my+i*26), msg, font=F_SM, fill=(r,g,b))
    frame[:]=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)

def draw_report(frame, w, h, now):
    if _report is None: return
    age = now - _report['time']
    if age > REPORT_DUR: return
    alpha = max(0.0,1.0-age/REPORT_DUR)
    pt    = _report['type']
    col   = PUNCH_COLS.get(pt,(0,200,200))
    dna   = PUNCH_DNA.get(pt,{})

    rx,ry = w-310,100; rw,rh = 290,180
    ov=frame.copy(); cv2.rectangle(ov,(rx,ry),(rx+rw,ry+rh),(8,8,25),-1)
    cv2.addWeighted(ov,alpha*0.8,frame,1-alpha*0.8,0,frame)

    pil  = PILImage.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).convert('RGBA')
    tmp  = PILImage.new('RGBA',pil.size,(0,0,0,0))
    draw = ImageDraw.Draw(tmp)
    def txt(t,x,y,c,f=F_MD):
        b,g,r=c; draw.text((x,y),t,font=f,fill=(r,g,b,int(alpha*240)))
    txt(f'[ {pt.upper()} FORM ]',rx+10,ry+6,col)
    metrics = [
        ('팔 뻗음',  _report['arm_ext'],dna.get('arm_extension_avg',0),TOL_PUNCH['arm_extension'],'{:.2f}',''),
        ('팔꿈치 각',_report['el_ang'], dna.get('elbow_angle_avg',0),  TOL_PUNCH['elbow_angle'],  '{:.0f}','°'),
        ('상체 균형',_report['lean'],   dna.get('lean_forward_avg',0), TOL_PUNCH['lean_forward'], '{:+.2f}',''),
    ]
    for i,(name,val,ref,tol,fmt,unit) in enumerate(metrics):
        y = ry+42+i*44
        ok = abs(val-ref)<=tol; bar_col=(0,200,80) if ok else (0,80,255)
        icon = '✓' if ok else ('▲' if val<ref else '▼')
        txt(name,rx+10,y,(180,180,180),F_SM)
        bx,by2,bw2,bh2 = rx+10,y+20,180,10
        cv2.rectangle(frame,(bx,by2),(bx+bw2,by2+bh2),(40,40,60),-1)
        ratio=min(1.0,val/max(ref,0.001)) if ref>0 else 0.5
        cv2.rectangle(frame,(bx,by2),(bx+int(bw2*ratio),by2+bh2),bar_col,-1)
        ref_x=bx+int(bw2*min(1.0,ref/max(ref,0.001)))
        cv2.line(frame,(ref_x,by2-2),(ref_x,by2+bh2+2),(200,200,200),2)
        txt(f'{fmt.format(val)+unit} / {fmt.format(ref)+unit} {icon}',
            rx+200,y+14,(0,200,80) if ok else (80,140,255),F_SM)
    merged=PILImage.alpha_composite(pil,tmp).convert('RGB')
    frame[:]=cv2.cvtColor(np.array(merged),cv2.COLOR_RGB2BGR)

def draw_ghost(frame, kp, sw):
    JAB_WR, JAB_SH, JAB_EL, CROSS_WR, CROSS_SH, CROSS_EL = get_arm_indices()
    targets = [
        (kp[JAB_SH][0],   kp[JAB_SH][1]  +POSE_DNA['guard_l_ydiff']*sw, (0,220,255),'JAB'),
        (kp[CROSS_SH][0], kp[CROSS_SH][1] +POSE_DNA['guard_r_ydiff']*sw, (255,160,40),'CROSS'),
    ]
    ov=frame.copy()
    for ix,iy,col,_ in targets:
        cx,cy=int(ix),int(iy)
        cv2.circle(ov,(cx,cy),22,col,2); cv2.circle(ov,(cx,cy),5,col,-1)
        cv2.line(ov,(cx-13,cy),(cx+13,cy),col,1); cv2.line(ov,(cx,cy-13),(cx,cy+13),col,1)
    cv2.addWeighted(ov,0.4,frame,0.6,0,frame)

def draw_vel_debug(frame, w):
    cd_l = max(0.0, PUNCH_CD-(time.time()-_punch_cd['lead']))
    cd_r = max(0.0, PUNCH_CD-(time.time()-_punch_cd['rear']))
    dir_lbl = '우향→' if _facing_right else '←좌향'
    lines = [
        (f"LEAD  v={_dbg_vel['lead']:.3f}  cd={cd_l:.2f}  → {_last_pt['lead']}",
         PUNCH_COLS[_last_pt['lead']] if cd_l<0.1 else (120,120,120)),
        (f"REAR  v={_dbg_vel['rear']:.3f}  cd={cd_r:.2f}  → {_last_pt['rear']}",
         PUNCH_COLS[_last_pt['rear']] if cd_r<0.1 else (120,120,120)),
        (f"방향:{dir_lbl}  vel={VEL_START}  TTS={'ON' if _voice_on else 'OFF'}",
         (80,80,80)),
    ]
    for i,(t,col) in enumerate(lines):
        cv2.putText(frame,t,(w-460,110+i*22),cv2.FONT_HERSHEY_SIMPLEX,0.50,col,1)

def draw_ready(frame, pose_ok, stable, w, h):
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(w,h),(8,8,25),-1)
    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
    put_kr(frame,'LIM 코치 4 — 측면 카메라 펀치 코치',(w//2-220,55),(0,200,255),F_LG)
    put_kr(frame,'카메라 옆에 서서 오르토독스 자세로 시작',(w//2-210,115),(160,255,160),F_SM)
    if pose_ok:
        prog=min(stable/READY_FRAMES,1.0)
        bx,by,bw_,bh_=w//2-150,h//2,300,20
        cv2.rectangle(frame,(bx,by),(bx+bw_,by+bh_),(50,50,50),-1)
        cv2.rectangle(frame,(bx,by),(bx+int(bw_*prog),by+bh_),(0,200,80),-1)
        put_kr(frame,f'자세 인식 중… {int(prog*100)}%',(bx,by-28),(200,255,200),F_SM)
    else:
        put_kr(frame,'전신이 보이도록 서주세요',(w//2-140,h//2),(150,150,255),F_SM)
    tips = [
        '• 측면 카메라 | 오르토독스 스탠스 (우향이 기본)',
        '• D키로 방향 전환 (우향/좌향)',
        '• R=초기화  G=가이드  V=음성  S=스냅샷  Q=종료',
    ]
    for i,t in enumerate(tips):
        put_kr(frame,t,(40,h-110+i*30),(120,120,120),F_SM)

# ══════════════════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════════════════
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

_flash = None

while cap.isOpened():
    ret,frame=cap.read()
    if not ret: break
    h,w=frame.shape[:2]
    now=time.time()

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    kps,scs=pose_model(rgb)
    disp=frame.copy()   # 측면 카메라는 미러 없음

    pose_ok=len(kps)>0
    kp=kps[0] if pose_ok else None
    sc=scs[0] if pose_ok else None
    if pose_ok:
        pose_ok=all(sc[i]>VIS_MIN for i in NEEDED_KP)

    sw=None
    if pose_ok:
        sw=sw_px(kp)
        draw_skeleton(disp,kp,sc)

    # ── 준비 화면 ────────────────────────────────────────────────
    if not _pose_ready:
        if pose_ok: _pose_stable=min(_pose_stable+1,READY_FRAMES)
        else:       _pose_stable=max(_pose_stable-2,0)
        draw_ready(disp,pose_ok,_pose_stable,w,h)
        if _pose_stable>=READY_FRAMES: _pose_ready=True
        cv2.imshow('LIM Coach 4 — 측면',disp)
        if cv2.waitKey(1)&0xFF in (ord('q'),27): break
        continue

    # ── 메인 ────────────────────────────────────────────────────
    if pose_ok:
        _lm_buf.append((kp,sc))

        for entry in list(_pending):
            entry[0]-=1
            if entry[0]<=0:
                _pending.remove(entry)
                analyse_punch(entry[1],entry[2])
                _flash=(entry[1],now+0.18)

        update_punch_detect(kp,sc,now)
        draw_trail(disp)

        if _show_ghost: draw_ghost(disp,kp,sw)

        total,grade,gcol,items = calc_posture(kp,sc,sw)
        draw_score_panel(disp,total,grade,gcol,items)
    else:
        _pose_stable=max(_pose_stable-2,0)
        if _pose_stable<READY_FRAMES//2: _pose_ready=False

    if _flash and now<_flash[1]:
        col=PUNCH_COLS.get(_flash[0],(0,200,200))
        cv2.rectangle(disp,(0,0),(w-1,h-1),col,6)

    draw_counter(disp,w)
    draw_report(disp,w,h,now)
    draw_vel_debug(disp,w)

    v_col=(0,200,80) if _voice_on else (0,60,180)
    put_kr(disp,f'음성 {"ON" if _voice_on else "OFF"}  (V키)',(w-180,h-30),v_col,F_SM)
    dir_col=(0,220,255) if _facing_right else (255,160,40)
    put_kr(disp,f'{"우향 →" if _facing_right else "← 좌향"}  (D키)',(w-180,h-56),dir_col,F_SM)
    put_kr(disp,f'DNA: {os.path.basename(_dna_src)}',(10,h-30),(80,180,80),F_SM)

    cv2.imshow('LIM Coach 4 — 측면',disp)
    key=cv2.waitKey(1)&0xFF
    if key in (ord('q'),27): break
    elif key==ord('r'):
        for t in _count: _count[t]=0
        _report=None
    elif key==ord('g'): _show_ghost=not _show_ghost
    elif key==ord('v'): _voice_on=not _voice_on
    elif key==ord('d'):
        _facing_right=not _facing_right
        _prev_lead_wr=None; _prev_rear_wr=None
        _v_buf_lead.clear(); _v_buf_rear.clear()
        print(f"[방향] {'우향 →' if _facing_right else '← 좌향'}")
    elif key==ord('s'):
        _snap_count+=1
        fn=os.path.join(BASE_DIR,f'coach4_snap_{_snap_count:03d}.png')
        cv2.imwrite(fn,disp); print(f'스냅샷: {fn}')

cap.release()
cv2.destroyAllWindows()
