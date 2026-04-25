"""
boxing game 3.py — 방어 & 카운터 복싱 게임
────────────────────────────────────────────
RTMPose (RTMO-s) 기반. 정면 카메라.

규칙
  AI가 공격 방향을 알려주면 막거나 피하고 카운터를 날려라!
  ● 팔로 막기 (팔을 코 위로 높이 올리기) → 막은 팔로 카운터 = AI -1
  ● 머리로 피하기 (고개를 왼쪽/오른쪽으로) → 반대 팔로 카운터 = AI -1
  ● 카운터는 선택 — 막기만 해도 OK

난이도 (키보드 1~4)
  1  EASY    : AI 체력 3  / 목숨 무한
  2  NORMAL  : AI 체력 5  / 목숨 5
  3  HARD    : AI 체력 7  / 목숨 3
  4  EXTREME : AI 체력 10 / 목숨 1

단축키
  1~4  난이도 선택
  R    재시작
  Q    종료
"""

import cv2, numpy as np, math, os, time, random
from collections import deque
from PIL import ImageFont, ImageDraw, Image as PILImage

try:
    from rtmlib import RTMO
except ImportError:
    raise SystemExit("pip install rtmlib onnxruntime")

# ══════════════════════════════════════════════════════════════════
# 모델 (LIM coach 2 와 공유)
# ══════════════════════════════════════════════════════════════════
RTMO_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
    'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip'
)
print("RTMPose 로드 중...")
pose_model = RTMO(RTMO_URL, backend='onnxruntime', device='cpu')
print("준비 완료")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
NEEDED  = [KP_L_SH,KP_R_SH,KP_L_WR,KP_R_WR,KP_L_EL,KP_R_EL,KP_NOSE]


# ══════════════════════════════════════════════════════════════════
# 난이도
# ══════════════════════════════════════════════════════════════════
DIFF = {
    'EASY':    {'ai_hp':3,  'p_lives':99, 'col':(0,255,120),  'label':'EASY'},
    'NORMAL':  {'ai_hp':5,  'p_lives':5,  'col':(0,200,255),  'label':'NORMAL'},
    'HARD':    {'ai_hp':7,  'p_lives':3,  'col':(0,130,255),  'label':'HARD'},
    'EXTREME': {'ai_hp':10, 'p_lives':1,  'col':(0,50,255),   'label':'EXTREME'},
}

# ══════════════════════════════════════════════════════════════════
# 게임 상수
# ══════════════════════════════════════════════════════════════════
TOTAL_ROUNDS  = 10
# 단일 공격 타이밍 범위 (초)
WARN_SINGLE   = (1.4, 2.0)
DEF_SINGLE    = (1.1, 1.5)
# 콤보 타이밍 범위 (빠름)
WARN_COMBO    = (0.6, 1.0)
DEF_COMBO     = (0.7, 1.0)

COUNTER_DUR   = 1.5    # 카운터 입력 시간
RESULT_DUR    = 0.8    # 결과 표시 시간

# 블록: 손목이 코보다 위에 있어야 함 (코 Y - threshold * sw)
BLOCK_NOSE_THRESH = 0.10   # 코보다 sw*10% 이상 위
# 슬립: 코가 어깨 중심에서 sw*thresh 이상 벗어남
SLIP_THRESH    = 0.28
# 카운터 펀치 속도 (sw-normalized)
PUNCH_VEL      = 0.35
PUNCH_DOM      = 1.5
COUNTER_DELAY  = 0.40   # 카운터 페이즈 시작 후 감지 대기 (오탐 방지)

# 공격 → 유효 방어 → 카운터 팔 매핑
# 'LEFT' 공격 = 플레이어 왼쪽으로 AI 주먹이 날아옴
VALID_DEF = {
    'LEFT':  {'BLOCK_L':'LEFT',  'SLIP_R':'RIGHT'},
    'RIGHT': {'BLOCK_R':'RIGHT', 'SLIP_L':'LEFT'},
}

# ══════════════════════════════════════════════════════════════════
# 게임 상태 변수
# ══════════════════════════════════════════════════════════════════
_gstate      = 'DIFF_SELECT'  # DIFF_SELECT / COUNTDOWN / PLAYING / WIN / LOSE
_sub         = 'WARN'          # WARN / DEFEND / COUNTER / RESULT
_diff        = 'NORMAL'
_ai_hp       = 5
_p_lives     = 5
_round_num   = 0
_combo       = []
_combo_idx   = 0
_defended    = False
_counter_arm = None
_countered   = False
_result_ok   = False
_phase_start = 0.0
_cntdn       = 3
_score       = 0               # 총 카운터 횟수

_prev_kp_m   = None
_warn_dur    = 1.8
_defend_dur  = 1.4

# ══════════════════════════════════════════════════════════════════
# 폰트
# ══════════════════════════════════════════════════════════════════
def _font(sz):
    for p in ["C:/Windows/Fonts/malgun.ttf","C:/Windows/Fonts/gulim.ttc"]:
        try: return ImageFont.truetype(p, sz)
        except: pass
    return ImageFont.load_default()

F_SM=_font(18); F_MD=_font(28); F_LG=_font(46); F_XL=_font(80)

def put_kr(img, text, pos, col, font=None):
    font = font or F_MD
    pil  = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(pos, text, font=font, fill=(col[2],col[1],col[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ══════════════════════════════════════════════════════════════════
# 기하 헬퍼
# ══════════════════════════════════════════════════════════════════
def spatial_arms(kp_m):
    """미러 디스플레이 기준 오른쪽/왼쪽 팔 인덱스.
    RTMO가 해부학적(좌/우)이든 위치적(이미지 좌/우)이든 x 비교로 해결."""
    if kp_m[KP_L_WR][0] > kp_m[KP_R_WR][0]:
        return {'r':[KP_L_SH,KP_L_EL,KP_L_WR], 'l':[KP_R_SH,KP_R_EL,KP_R_WR]}
    return {'r':[KP_R_SH,KP_R_EL,KP_R_WR], 'l':[KP_L_SH,KP_L_EL,KP_L_WR]}

def sw_m(kp_m):
    dx = kp_m[KP_R_SH][0]-kp_m[KP_L_SH][0]
    dy = kp_m[KP_R_SH][1]-kp_m[KP_L_SH][1]
    return math.sqrt(dx*dx+dy*dy)+1e-6

def mirror_kp(kp, w):
    m = kp.copy(); m[:,0] = w-kp[:,0]; return m

def draw_skeleton(frame, kp_m, sc):
    for a,b in COCO_CONN:
        if sc[a]>VIS_MIN and sc[b]>VIS_MIN:
            cv2.line(frame,(int(kp_m[a][0]),int(kp_m[a][1])),
                     (int(kp_m[b][0]),int(kp_m[b][1])),(60,60,60),2)
    for i in range(17):
        if sc[i]>VIS_MIN:
            cv2.circle(frame,(int(kp_m[i][0]),int(kp_m[i][1])),4,(0,180,220),-1)

# ══════════════════════════════════════════════════════════════════
# 방어 / 카운터 감지
# ══════════════════════════════════════════════════════════════════
def detect_block(kp_m, sw):
    """손목이 코보다 BLOCK_NOSE_THRESH*sw 만큼 위에 있으면 블록 (공간적 좌우)"""
    nose_y = kp_m[KP_NOSE][1]
    thr    = nose_y - BLOCK_NOSE_THRESH * sw
    arms   = spatial_arms(kp_m)
    r_up   = kp_m[arms['r'][2]][1] < thr
    l_up   = kp_m[arms['l'][2]][1] < thr
    if r_up and not l_up: return 'RIGHT'
    if l_up and not r_up: return 'LEFT'
    return None

def detect_slip(kp_m, sw):
    """코가 어깨 중심에서 SLIP_THRESH*sw 이상 벗어나면 슬립"""
    sh_cx  = (kp_m[KP_L_SH][0]+kp_m[KP_R_SH][0])/2
    dev    = (kp_m[KP_NOSE][0]-sh_cx)/sw
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

def detect_punch(kp_m, sw):
    """카운터 펀치 방향 감지 (RIGHT/LEFT/None) - 공간적 좌우"""
    global _prev_kp_m
    if _prev_kp_m is None:
        _prev_kp_m = kp_m.copy(); return None
    arms  = spatial_arms(kp_m)
    r_wr  = arms['r'][2]; l_wr = arms['l'][2]
    dr = math.sqrt((kp_m[r_wr][0]-_prev_kp_m[r_wr][0])**2+
                   (kp_m[r_wr][1]-_prev_kp_m[r_wr][1])**2)/sw
    dl = math.sqrt((kp_m[l_wr][0]-_prev_kp_m[l_wr][0])**2+
                   (kp_m[l_wr][1]-_prev_kp_m[l_wr][1])**2)/sw
    _prev_kp_m = kp_m.copy()
    if dr > PUNCH_VEL and dr > dl*PUNCH_DOM: return 'RIGHT'
    if dl > PUNCH_VEL and dl > dr*PUNCH_DOM: return 'LEFT'
    return None

# ══════════════════════════════════════════════════════════════════
# 게임 로직
# ══════════════════════════════════════════════════════════════════
def gen_combo():
    return [random.choice(['LEFT','RIGHT']) for _ in range(random.randint(1,3))]

def start_attack():
    global _sub,_phase_start,_defended,_counter_arm,_countered,_result_ok,_prev_kp_m
    _sub='WARN'; _phase_start=time.time()
    _defended=False; _counter_arm=None; _countered=False; _result_ok=False
    _prev_kp_m=None

def start_round_combo():
    """라운드마다 콤보 생성 + 콤보 크기에 따라 타이밍 결정 후 시작"""
    global _combo,_combo_idx,_warn_dur,_defend_dur
    _combo=gen_combo(); _combo_idx=0
    if len(_combo)==1:
        _warn_dur  = random.uniform(*WARN_SINGLE)
        _defend_dur = random.uniform(*DEF_SINGLE)
    else:
        _warn_dur  = random.uniform(*WARN_COMBO)
        _defend_dur = random.uniform(*DEF_COMBO)
    start_attack()

def start_game(diff_key):
    global _gstate,_diff,_ai_hp,_p_lives,_round_num,_cntdn,_score
    _diff=diff_key
    _ai_hp   = DIFF[diff_key]['ai_hp']
    _p_lives = DIFF[diff_key]['p_lives']
    _round_num=0; _score=0
    _gstate='COUNTDOWN'; _cntdn=3
    global _phase_start; _phase_start=time.time()

def advance():
    global _round_num,_gstate
    _round_num+=1
    if _round_num>=TOTAL_ROUNDS:
        _gstate='WIN'; return
    start_round_combo()

# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════
def draw_overlay(frame, alpha, col=(0,0,0)):
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(frame.shape[1],frame.shape[0]),col,-1)
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)

def timer_bar(frame, w, h, elapsed, total, col):
    ratio = max(0.0, 1.0-elapsed/total)
    cv2.rectangle(frame,(0,h-10),(w,h),(30,30,30),-1)
    cv2.rectangle(frame,(0,h-10),(int(w*ratio),h),col,-1)

def draw_hud(frame, w, h):
    cfg=DIFF[_diff]
    # 라운드 / 콤보
    cv2.putText(frame,f"Round {_round_num+1}/{TOTAL_ROUNDS}  Combo {_combo_idx+1}/{len(_combo)}",
                (10,32),cv2.FONT_HERSHEY_SIMPLEX,0.75,(180,180,180),2)
    # AI 체력
    ai_txt='AI: '+'★'*_ai_hp+'☆'*(cfg['ai_hp']-_ai_hp)
    put_kr(frame,ai_txt,(w//2-130,6),(0,220,255),F_MD)
    # 플레이어 목숨
    if _p_lives==99: lives='♥ ∞'
    else:            lives='♥ '+'|'*_p_lives
    put_kr(frame,lives,(w-200,6),(0,200,100),F_MD)
    # 점수
    cv2.putText(frame,f"Score:{_score}",(10,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(160,160,160),1)

def draw_attack_arrow(frame, w, h, attack):
    cx,cy=w//2,h//2-30
    if attack=='LEFT':
        tip=(cx-140,cy)
        pts=np.array([[tip[0],cy],[cx-60,cy-55],[cx-60,cy-18],
                      [cx+120,cy-18],[cx+120,cy+18],[cx-60,cy+18],[cx-60,cy+55]],np.int32)
        lbl='← 왼쪽 공격!'; lcol=(0,60,255)
    else:
        tip=(cx+140,cy)
        pts=np.array([[tip[0],cy],[cx+60,cy-55],[cx+60,cy-18],
                      [cx-120,cy-18],[cx-120,cy+18],[cx+60,cy+18],[cx+60,cy+55]],np.int32)
        lbl='오른쪽 공격! →'; lcol=(0,60,255)
    ov=frame.copy(); cv2.fillPoly(ov,[pts],lcol)
    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    cv2.polylines(frame,[pts],True,(0,100,255),2)
    put_kr(frame,lbl,(w//2-160,h//2+80),(0,120,255),F_LG)

def draw_defend_phase(frame, w, h, attack, elapsed):
    draw_overlay(frame,0.25,(0,10,30))
    draw_attack_arrow(frame,w,h,attack)
    if attack=='LEFT':
        h1,h2='← 왼팔 높이 올려서 막기','→ 오른쪽으로 고개 피하기'
    else:
        h1,h2='오른팔 높이 올려서 막기 →','← 왼쪽으로 고개 피하기'
    put_kr(frame,'[ 방어하세요! ]',(w//2-110,h//2-130),(0,220,255),F_MD)
    put_kr(frame,h1,(w//2-200,h//2+130),(0,200,255),F_SM)
    put_kr(frame,h2,(w//2-200,h//2+158),(180,180,180),F_SM)
    timer_bar(frame,w,h,elapsed,_defend_dur,(0,200,255))

def draw_counter_phase(frame, w, h, counter_arm, elapsed):
    draw_overlay(frame,0.20,(0,30,0))
    arm_kr='오른손' if counter_arm=='RIGHT' else '왼손'
    put_kr(frame,'카운터!',(w//2-80,h//2-100),(0,255,100),F_XL)
    put_kr(frame,f'{arm_kr}으로 펀치!',(w//2-130,h//2+10),(0,220,100),F_LG)
    put_kr(frame,'(안 쳐도 OK)',(w//2-90,h//2+80),(120,120,120),F_SM)
    timer_bar(frame,w,h,elapsed,COUNTER_DUR,(0,200,80))

def draw_result_flash(frame, w, h, ok):
    if ok:
        draw_overlay(frame,0.18,(0,30,0))
        put_kr(frame,'성공! ✓',(w//2-80,h//2-50),(0,255,100),F_XL)
    else:
        draw_overlay(frame,0.30,(30,0,0))
        put_kr(frame,'실패! ✗',(w//2-80,h//2-50),(0,60,255),F_XL)

def draw_diff_select(frame, w, h):
    draw_overlay(frame,0.75)
    put_kr(frame,'BOXING DEFENSE GAME',(w//2-250,50),(0,220,255),F_LG)
    put_kr(frame,'난이도를 선택하세요  (1 ~ 4 키)',(w//2-200,120),(180,180,180),F_MD)
    rows=[('1  EASY',    'AI 체력 3 / 목숨 무한',DIFF['EASY']['col']),
          ('2  NORMAL',  'AI 체력 5 / 목숨 5',   DIFF['NORMAL']['col']),
          ('3  HARD',    'AI 체력 7 / 목숨 3',   DIFF['HARD']['col']),
          ('4  EXTREME', 'AI 체력 10 / 목숨 1',  DIFF['EXTREME']['col'])]
    for i,(lbl,desc,col) in enumerate(rows):
        y=190+i*95
        cv2.rectangle(frame,(w//2-220,y),(w//2+220,y+78),col,2)
        put_kr(frame,lbl,(w//2-200,y+6),col,F_MD)
        put_kr(frame,desc,(w//2-200,y+44),(140,140,140),F_SM)

def draw_countdown(frame, w, h):
    draw_overlay(frame,0.65)
    put_kr(frame,DIFF[_diff]['label'],(w//2-80,h//2-200),DIFF[_diff]['col'],F_LG)
    put_kr(frame,'준비!',(w//2-50,h//2-120),(180,180,180),F_MD)
    n=str(_cntdn) if _cntdn>0 else 'GO!'
    col=(0,220,255) if _cntdn>0 else (0,255,100)
    put_kr(frame,n,(w//2-50,h//2-40),col,F_XL)

def draw_win(frame, w, h):
    draw_overlay(frame,0.80,(0,15,0))
    put_kr(frame,'🏆 승리!',(w//2-120,h//2-130),(0,255,120),F_XL)
    put_kr(frame,f'카운터 {_score}번',(w//2-100,h//2+10),(0,200,100),F_LG)
    put_kr(frame,f'AI 남은 체력: {_ai_hp}',(w//2-130,h//2+80),(0,180,100),F_MD)
    put_kr(frame,'R = 재시작  /  Q = 종료',(w//2-180,h//2+160),(140,140,140),F_MD)

def draw_lose(frame, w, h):
    draw_overlay(frame,0.80,(20,0,0))
    put_kr(frame,'패배',(w//2-70,h//2-130),(0,60,255),F_XL)
    put_kr(frame,f'카운터 {_score}번',(w//2-100,h//2+10),(100,100,255),F_LG)
    put_kr(frame,'R = 재시작  /  Q = 종료',(w//2-180,h//2+120),(140,140,140),F_MD)

def highlight_arm(frame, kp_m, sc, arm, col):
    """방어/카운터한 팔 강조 (공간적 좌우)"""
    idxs = spatial_arms(kp_m)['r' if arm=='RIGHT' else 'l']
    for i in range(len(idxs)-1):
        a,b=idxs[i],idxs[i+1]
        if sc[a]>VIS_MIN and sc[b]>VIS_MIN:
            cv2.line(frame,(int(kp_m[a][0]),int(kp_m[a][1])),
                     (int(kp_m[b][0]),int(kp_m[b][1])),col,6)
    if sc[idxs[2]]>VIS_MIN:
        cv2.circle(frame,(int(kp_m[idxs[2]][0]),int(kp_m[idxs[2]][1])),12,col,-1)

def draw_center_line(frame, kp_m, sw, h):
    """어깨 중심 수직 점선 + 코 위치 표시 (슬립 기준선)"""
    sh_cx = int((kp_m[KP_L_SH][0]+kp_m[KP_R_SH][0])/2)
    for y in range(0, h, 24):
        cv2.line(frame,(sh_cx,y),(sh_cx,min(y+12,h)),(90,90,90),1)
    nx, ny = int(kp_m[KP_NOSE][0]), int(kp_m[KP_NOSE][1])
    dev = (nx - sh_cx) / sw
    col = (0,200,255) if abs(dev)>=SLIP_THRESH else (80,220,80)
    cv2.circle(frame,(nx,ny),10,col,2)
    if abs(dev) > 0.10:
        cv2.arrowedLine(frame,(sh_cx,ny),(nx,ny),col,2,tipLength=0.3)

# ══════════════════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════════════════
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

while cap.isOpened():
    ret,frame=cap.read()
    if not ret: break
    h,w=frame.shape[:2]
    now=time.time()

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    kps,scs=pose_model(rgb)
    disp=cv2.flip(frame,1)

    pose_ok=len(kps)>0
    kp=kps[0] if pose_ok else None
    sc=scs[0] if pose_ok else None
    if pose_ok:
        pose_ok=all(sc[i]>VIS_MIN for i in NEEDED)

    kp_m=None
    if pose_ok:
        kp_m=mirror_kp(kp,w)
        sw=sw_m(kp_m)
        draw_skeleton(disp,kp_m,sc)

    # ── 상태 머신 ────────────────────────────────────────────────
    if _gstate=='DIFF_SELECT':
        draw_diff_select(disp,w,h)

    elif _gstate=='COUNTDOWN':
        draw_countdown(disp,w,h)
        elapsed=now-_phase_start
        new_n=3-int(elapsed)
        if new_n!=_cntdn and new_n>=0: _cntdn=new_n
        if elapsed>=4.0:
            _gstate='PLAYING'; start_round_combo()

    elif _gstate=='PLAYING':
        attack=_combo[_combo_idx]
        elapsed=now-_phase_start

        draw_hud(disp,w,h)

        if kp_m is not None:
            draw_center_line(disp,kp_m,sw,h)

        # ── WARN: 공격 예고 ──────────────────────────────────────
        if _sub=='WARN':
            draw_overlay(disp,0.20,(20,0,0))
            draw_attack_arrow(disp,w,h,attack)
            timer_bar(disp,w,h,elapsed,_warn_dur,(0,60,255))
            if elapsed>=_warn_dur:
                _sub='DEFEND'; _phase_start=now

        # ── DEFEND: 방어 입력 ────────────────────────────────────
        elif _sub=='DEFEND':
            draw_defend_phase(disp,w,h,attack,elapsed)

            if kp_m is not None and not _defended:
                defense=get_defense(kp_m,sw)
                if defense and defense in VALID_DEF.get(attack,{}):
                    _defended=True
                    _counter_arm=VALID_DEF[attack][defense]
                    arm_kw='RIGHT' if 'R' in defense else 'LEFT'
                    highlight_arm(disp,kp_m,sc,arm_kw,(0,255,100))
                    if _combo_idx < len(_combo)-1:
                        # 콤보 중간: 결과화면 없이 즉시 다음 공격
                        _combo_idx+=1
                        start_attack()
                    else:
                        # 마지막 공격: 카운터로
                        _sub='COUNTER'; _phase_start=now; _prev_kp_m=None

            if elapsed>=_defend_dur and _sub=='DEFEND':
                if _diff!='EASY': _p_lives-=1
                _result_ok=False
                _sub='RESULT'; _phase_start=now
                if _p_lives<=0 and _diff!='EASY': _gstate='LOSE'

        # ── COUNTER: 카운터 기회 ─────────────────────────────────
        elif _sub=='COUNTER':
            draw_counter_phase(disp,w,h,_counter_arm,elapsed)

            if kp_m is not None:
                if elapsed < COUNTER_DELAY:
                    _prev_kp_m = kp_m.copy()   # 대기 중 베이스라인 갱신만
                elif not _countered:
                    punch=detect_punch(kp_m,sw)
                    if punch==_counter_arm:
                        _countered=True; _result_ok=True
                        _score+=1
                        _ai_hp=max(0,_ai_hp-1)
                        highlight_arm(disp,kp_m,sc,_counter_arm,(0,255,200))
                        _sub='RESULT'; _phase_start=now
                        if _ai_hp<=0: _gstate='WIN'
                else:
                    detect_punch(kp_m,sw)   # _prev_kp_m 계속 갱신

            if elapsed>=COUNTER_DUR and _sub=='COUNTER':
                _result_ok=True
                _sub='RESULT'; _phase_start=now

        # ── RESULT: 결과 표시 ────────────────────────────────────
        elif _sub=='RESULT':
            draw_result_flash(disp,w,h,_result_ok)
            if now-_phase_start>=RESULT_DUR:
                if _gstate=='PLAYING': advance()

    elif _gstate=='WIN':
        if kp_m is not None: draw_skeleton(disp,kp_m,sc)
        draw_win(disp,w,h)

    elif _gstate=='LOSE':
        draw_lose(disp,w,h)

    cv2.imshow('Boxing Defense Game',disp)
    key=cv2.waitKey(1)&0xFF
    if key in (ord('q'),27): break
    elif key==ord('r'):      _gstate='DIFF_SELECT'
    elif _gstate=='DIFF_SELECT':
        if   key==ord('1'): start_game('EASY')
        elif key==ord('2'): start_game('NORMAL')
        elif key==ord('3'): start_game('HARD')
        elif key==ord('4'): start_game('EXTREME')

cap.release()
cv2.destroyAllWindows()
