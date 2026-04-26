"""
boxing game 3.py — 방어 & 카운터 복싱 게임
────────────────────────────────────────────
RTMPose (RTMO-s) 기반. 정면 카메라.

규칙
  AI가 공격 방향을 알려주면 막거나 피하고 카운터를 날려라!
  ● 팔로 막기 (팔을 코 위로 높이 올리기)  → 막은 팔로 카운터
  ● 머리로 피하기 (고개를 왼쪽/오른쪽으로) → 반대 팔로 카운터
  ● 콤보는 전부 막은 뒤 마지막에만 카운터 기회 제공
  ● 카운터 성공 = PERFECT / 카운터 없이 방어만 = 성공

난이도 (키보드 1~4)
  1  EASY    : AI 체력 3  / HP 무한 (연습용)
  2  NORMAL  : AI 체력 5  / HP 100 (5회 실패 허용)
  3  HARD    : AI 체력 7  / HP 100 (3회 실패 허용) + 패턴 AI
  4  EXTREME : AI 체력 10 / HP 100 (1회 실패 허용) + 극한 패턴

스피드 모드
  10라운드 클리어 후 자동 진입 — 2배속부터 시작, 막을수록 빨라짐
  방어 실패하면 스피드 라운드 기록과 함께 종료

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
# 모델
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
# 난이도 — HP 시스템
# ══════════════════════════════════════════════════════════════════
DIFF = {
    'EASY':    {'ai_hp':3,  'p_hp':100, 'p_dmg':0,   'col':(0,255,120),  'label':'EASY'},
    'NORMAL':  {'ai_hp':5,  'p_hp':100, 'p_dmg':20,  'col':(0,200,255),  'label':'NORMAL'},
    'HARD':    {'ai_hp':7,  'p_hp':100, 'p_dmg':34,  'col':(0,130,255),  'label':'HARD'},
    'EXTREME': {'ai_hp':10, 'p_hp':100, 'p_dmg':100, 'col':(0,50,255),   'label':'EXTREME'},
}

# ══════════════════════════════════════════════════════════════════
# 패턴 AI — 난이도별 공격 패턴
# ══════════════════════════════════════════════════════════════════
AI_PATTERNS = {
    'EASY':    None,  # 완전 랜덤
    'NORMAL':  [
        ['LEFT','RIGHT'], ['RIGHT','LEFT'],
        ['LEFT'], ['RIGHT'],
        ['LEFT','RIGHT','LEFT'],
    ],
    'HARD':    [
        ['LEFT','LEFT'], ['RIGHT','RIGHT'],
        ['LEFT','RIGHT','LEFT'], ['RIGHT','LEFT','RIGHT'],
        ['LEFT','RIGHT','RIGHT'], ['RIGHT','LEFT','LEFT'],
    ],
    'EXTREME': [
        ['LEFT','LEFT','LEFT'], ['RIGHT','RIGHT','RIGHT'],
        ['LEFT','RIGHT','LEFT'], ['RIGHT','LEFT','RIGHT'],
        ['LEFT','LEFT','RIGHT'], ['RIGHT','RIGHT','LEFT'],
        ['RIGHT','LEFT','LEFT'], ['LEFT','RIGHT','RIGHT'],
    ],
}

# ══════════════════════════════════════════════════════════════════
# 타이밍 상수
# ══════════════════════════════════════════════════════════════════
TOTAL_ROUNDS    = 10
WARN_SINGLE     = (1.4, 2.0)   # 단일 공격 예고
DEF_SINGLE      = (1.1, 1.5)   # 단일 공격 방어 시간
WARN_COMBO      = (0.6, 1.0)   # 콤보 예고
DEF_COMBO       = (0.7, 1.0)   # 콤보 방어 시간
COUNTER_DUR     = 1.5
RESULT_DUR      = 0.8
SPEED_ALERT_DUR = 2.5

# ══════════════════════════════════════════════════════════════════
# 감지 상수
# ══════════════════════════════════════════════════════════════════
BLOCK_NOSE_THRESH  = 0.10
SLIP_THRESH        = 0.28
GUARD_DROP_THRESH  = 0.55  # 어깨에서 sw*0.55 이상 낮으면 가드 내려간 것
PUNCH_VEL      = 0.18    # 프레임간 속도 (낮게 — extension이 주 필터)
PUNCH_EXTEND   = 0.38    # 베이스라인 대비 최소 이동거리 (sw 배수)
PUNCH_DOM      = 1.4     # 한 팔이 다른 팔보다 이만큼 더 빨라야
COUNTER_DELAY  = 0.45    # 블로킹 자세 충분히 풀릴 시간

VALID_DEF = {
    'LEFT':  {'BLOCK_L':'LEFT',  'SLIP_R':'RIGHT'},
    'RIGHT': {'BLOCK_R':'RIGHT', 'SLIP_L':'LEFT'},
}

# ══════════════════════════════════════════════════════════════════
# 게임 상태 변수
# ══════════════════════════════════════════════════════════════════
_gstate      = 'DIFF_SELECT'
_sub         = 'WARN'
_diff        = 'NORMAL'
_ai_hp       = 5
_ai_hp_max   = 5
_p_hp        = 100
_p_hp_max    = 100
_p_dmg       = 20
_round_num   = 0
_combo       = []
_combo_idx   = 0
_defended    = False
_counter_arm = None
_countered   = False
_result_ok   = False
_phase_start = 0.0
_cntdn       = 3
_score       = 0

_prev_kp_m          = None
_vel_buf_r          = deque(maxlen=3)   # 오른팔 속도 버퍼 (3프레임)
_vel_buf_l          = deque(maxlen=3)   # 왼팔 속도 버퍼
_punch_base_r       = None              # 카운터 딜레이 끝 기준 오른손목 위치
_punch_base_l       = None              # 카운터 딜레이 끝 기준 왼손목 위치
_warn_dur           = 1.8
_defend_dur         = 1.4
_defend_phase_start = 0.0

# 스피드 모드
_speed_mode  = False
_speed_mult  = 1.0
_speed_round = 0

# 반응속도 통계
_react_times = []

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
    """어깨 x좌표 기준 좌우 판단 — 어깨는 펀치/블록 중에도 안정적"""
    if kp_m[KP_L_SH][0] > kp_m[KP_R_SH][0]:
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
def check_guard(kp_m, sw):
    """양 손목이 어깨에서 GUARD_DROP_THRESH*sw 이내에 있으면 가드 OK"""
    sh_y  = (kp_m[KP_L_SH][1]+kp_m[KP_R_SH][1])/2
    thr   = sh_y + GUARD_DROP_THRESH*sw
    arms  = spatial_arms(kp_m)
    return (kp_m[arms['r'][2]][1] < thr) and (kp_m[arms['l'][2]][1] < thr)

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
    sh_cx = (kp_m[KP_L_SH][0]+kp_m[KP_R_SH][0])/2
    dev   = (kp_m[KP_NOSE][0]-sh_cx)/sw
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

def set_punch_baseline(kp_m):
    """카운터 딜레이 중 매 프레임 갱신 — 딜레이 끝 시점의 안정 위치를 기준으로 삼음"""
    global _punch_base_r, _punch_base_l
    arms = spatial_arms(kp_m)
    _punch_base_r = kp_m[arms['r'][2]][:2].copy()
    _punch_base_l = kp_m[arms['l'][2]][:2].copy()

def detect_punch(kp_m, sw):
    """속도(프레임간) AND 이동거리(베이스라인 대비) 둘 다 충족해야 펀치 인정"""
    global _prev_kp_m, _vel_buf_r, _vel_buf_l
    if _prev_kp_m is None or _punch_base_r is None:
        _prev_kp_m = kp_m.copy(); return None
    arms = spatial_arms(kp_m)
    r_wr = arms['r'][2]; l_wr = arms['l'][2]
    vr = math.sqrt((kp_m[r_wr][0]-_prev_kp_m[r_wr][0])**2+
                   (kp_m[r_wr][1]-_prev_kp_m[r_wr][1])**2)/sw
    vl = math.sqrt((kp_m[l_wr][0]-_prev_kp_m[l_wr][0])**2+
                   (kp_m[l_wr][1]-_prev_kp_m[l_wr][1])**2)/sw
    _prev_kp_m = kp_m.copy()
    _vel_buf_r.append(vr); _vel_buf_l.append(vl)
    er = math.sqrt((kp_m[r_wr][0]-_punch_base_r[0])**2+
                   (kp_m[r_wr][1]-_punch_base_r[1])**2)/sw
    el = math.sqrt((kp_m[l_wr][0]-_punch_base_l[0])**2+
                   (kp_m[l_wr][1]-_punch_base_l[1])**2)/sw
    pr = max(_vel_buf_r); pl = max(_vel_buf_l)
    if pr > PUNCH_VEL and er > PUNCH_EXTEND and pr > pl*PUNCH_DOM: return 'RIGHT'
    if pl > PUNCH_VEL and el > PUNCH_EXTEND and pl > pr*PUNCH_DOM: return 'LEFT'
    return None

# ══════════════════════════════════════════════════════════════════
# 게임 로직
# ══════════════════════════════════════════════════════════════════
def gen_combo():
    patterns = AI_PATTERNS.get(_diff)
    if patterns is None or random.random() < 0.3:
        return [random.choice(['LEFT','RIGHT']) for _ in range(random.randint(1,3))]
    return list(random.choice(patterns))

def start_attack():
    global _sub,_phase_start,_defended,_counter_arm,_countered,_result_ok,_prev_kp_m
    global _punch_base_r, _punch_base_l
    _sub='WARN'; _phase_start=time.time()
    _defended=False; _counter_arm=None; _countered=False; _result_ok=False
    _prev_kp_m=None; _vel_buf_r.clear(); _vel_buf_l.clear()
    _punch_base_r=None; _punch_base_l=None

def start_round_combo():
    global _combo,_combo_idx,_warn_dur,_defend_dur
    _combo=gen_combo(); _combo_idx=0
    mult = _speed_mult
    if len(_combo)==1:
        _warn_dur   = random.uniform(*WARN_SINGLE) / mult
        _defend_dur = random.uniform(*DEF_SINGLE)  / mult
    else:
        _warn_dur   = random.uniform(*WARN_COMBO) / mult
        _defend_dur = random.uniform(*DEF_COMBO)  / mult
    start_attack()

def start_game(diff_key):
    global _gstate,_diff,_ai_hp,_ai_hp_max,_p_hp,_p_hp_max,_p_dmg
    global _round_num,_cntdn,_score,_speed_mode,_speed_mult,_speed_round,_react_times
    _diff    = diff_key
    cfg      = DIFF[diff_key]
    _ai_hp   = _ai_hp_max = cfg['ai_hp']
    _p_hp    = _p_hp_max  = cfg['p_hp']
    _p_dmg   = cfg['p_dmg']
    _round_num=0; _score=0
    _speed_mode=False; _speed_mult=1.0; _speed_round=0
    _react_times=[]
    _gstate='COUNTDOWN'; _cntdn=3
    global _phase_start; _phase_start=time.time()

def advance():
    global _round_num,_gstate,_speed_mode,_speed_mult,_speed_round,_phase_start
    _round_num+=1
    if _round_num>=TOTAL_ROUNDS and not _speed_mode:
        _speed_mode=True; _speed_mult=2.0; _speed_round=0
        _gstate='SPEED_ALERT'; _phase_start=time.time()
        return
    if _speed_mode:
        _speed_round+=1
        _speed_mult=2.0+_speed_round*0.15
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
    cfg = DIFF[_diff]
    # 라운드 표시 (스피드 모드는 별도)
    if _speed_mode:
        spd_lbl = f'SPEED x{_speed_mult:.1f}  #{_speed_round+1}'
        cv2.putText(frame,f"{spd_lbl}  Combo {_combo_idx+1}/{len(_combo)}",
                    (10,32),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,180,255),2)
        # 스피드 모드 테두리 깜빡임
        if int(time.time()*3)%2:
            cv2.rectangle(frame,(3,3),(w-3,h-3),(0,80,255),3)
    else:
        cv2.putText(frame,f"Round {_round_num+1}/{TOTAL_ROUNDS}  Combo {_combo_idx+1}/{len(_combo)}",
                    (10,32),cv2.FONT_HERSHEY_SIMPLEX,0.75,(180,180,180),2)
    # AI 체력
    ai_txt='AI: '+'★'*_ai_hp+'☆'*(cfg['ai_hp']-_ai_hp)
    put_kr(frame,ai_txt,(w//2-130,6),(0,220,255),F_MD)
    # 플레이어 HP 바
    bw,bh=200,18; bx,by=w-bw-10,10
    ratio=max(0.0,_p_hp/_p_hp_max)
    bar_col=(0,200,100) if ratio>0.5 else (0,180,255) if ratio>0.25 else (0,60,255)
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(40,40,40),-1)
    cv2.rectangle(frame,(bx,by),(bx+int(bw*ratio),by+bh),bar_col,-1)
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(120,120,120),1)
    hp_lbl = 'HP ∞' if _p_dmg==0 else f'HP {_p_hp}'
    put_kr(frame,hp_lbl,(bx-65,by),(180,180,180),F_SM)
    # 점수 + 평균 반응속도
    cv2.putText(frame,f"Score:{_score}",(10,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(160,160,160),1)
    if _react_times:
        avg_ms=int(sum(_react_times)/len(_react_times)*1000)
        cv2.putText(frame,f"Avg react:{avg_ms}ms",(10,h-44),cv2.FONT_HERSHEY_SIMPLEX,0.55,(120,180,120),1)

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

def draw_result_flash(frame, w, h, ok, perfect=False):
    if ok and perfect:
        draw_overlay(frame,0.15,(0,20,10))
        for i in range(3):
            cv2.rectangle(frame,(3+i*4,3+i*4),(w-3-i*4,h-3-i*4),(0,220,180),2)
        put_kr(frame,'PERFECT!',(w//2-110,h//2-60),(0,255,200),F_XL)
    elif ok:
        draw_overlay(frame,0.18,(0,30,0))
        put_kr(frame,'성공! ✓',(w//2-80,h//2-50),(0,255,100),F_XL)
    else:
        draw_overlay(frame,0.30,(30,0,0))
        put_kr(frame,'실패! ✗',(w//2-80,h//2-50),(0,60,255),F_XL)

def draw_diff_select(frame, w, h):
    draw_overlay(frame,0.75)
    put_kr(frame,'BOXING DEFENSE GAME',(w//2-250,50),(0,220,255),F_LG)
    put_kr(frame,'난이도를 선택하세요  (1 ~ 4 키)',(w//2-200,120),(180,180,180),F_MD)
    rows=[
        ('1  EASY',    'AI 체력 3  / HP 무한 (연습)',    DIFF['EASY']['col']),
        ('2  NORMAL',  'AI 체력 5  / HP 100 (5회 허용)', DIFF['NORMAL']['col']),
        ('3  HARD',    'AI 체력 7  / HP 100 (3회 허용)', DIFF['HARD']['col']),
        ('4  EXTREME', 'AI 체력 10 / HP 100 (1회 허용)', DIFF['EXTREME']['col']),
    ]
    for i,(lbl,desc,col) in enumerate(rows):
        y=190+i*95
        cv2.rectangle(frame,(w//2-220,y),(w//2+220,y+78),col,2)
        put_kr(frame,lbl,(w//2-200,y+6),col,F_MD)
        put_kr(frame,desc,(w//2-200,y+44),(140,140,140),F_SM)
    put_kr(frame,'10라운드 클리어 → 스피드 서바이벌 모드 자동 진입',
           (w//2-300,h-50),(80,180,80),F_SM)

def draw_countdown(frame, w, h):
    draw_overlay(frame,0.65)
    put_kr(frame,DIFF[_diff]['label'],(w//2-80,h//2-200),DIFF[_diff]['col'],F_LG)
    put_kr(frame,'준비!',(w//2-50,h//2-120),(180,180,180),F_MD)
    n=str(_cntdn) if _cntdn>0 else 'GO!'
    col=(0,220,255) if _cntdn>0 else (0,255,100)
    put_kr(frame,n,(w//2-50,h//2-40),col,F_XL)

def draw_speed_alert(frame, w, h):
    draw_overlay(frame,0.80,(20,0,30))
    put_kr(frame,'SPEED MODE!',(w//2-200,h//2-140),(0,180,255),F_XL)
    put_kr(frame,'2배속으로 버텨라!',(w//2-170,h//2+0),(200,200,200),F_LG)
    put_kr(frame,'방어 실패 = 게임 오버',(w//2-160,h//2+70),(120,120,120),F_MD)
    for i in range(3):
        cv2.rectangle(frame,(3+i*4,3+i*4),(w-3-i*4,h-3-i*4),(0,80,255),2)

def _stats_lines(frame, w, cy):
    """WIN/LOSE 화면 공통 통계 출력"""
    if _react_times:
        avg_ms=int(sum(_react_times)/len(_react_times)*1000)
        best_ms=int(min(_react_times)*1000)
        put_kr(frame,f'평균 반응속도: {avg_ms}ms  (최고: {best_ms}ms)',
               (w//2-210,cy),(160,200,160),F_SM)

def draw_win(frame, w, h):
    draw_overlay(frame,0.80,(0,15,0))
    put_kr(frame,'승리!',(w//2-100,h//2-180),(0,255,120),F_XL)
    put_kr(frame,f'카운터 {_score}번',(w//2-100,h//2-80),(0,200,100),F_LG)
    if _speed_mode and _speed_round>0:
        put_kr(frame,f'스피드 {_speed_round}라운드 생존',
               (w//2-170,h//2+0),(0,200,255),F_MD)
    _stats_lines(frame,w,h//2+55)
    put_kr(frame,'R = 재시작  /  Q = 종료',(w//2-180,h//2+140),(140,140,140),F_MD)

def draw_lose(frame, w, h):
    draw_overlay(frame,0.80,(20,0,0))
    if _speed_mode:
        title=f'스피드 {_speed_round}라운드 생존'
        tcol=(0,150,255)
    else:
        title='패배'; tcol=(0,60,255)
    put_kr(frame,title,(w//2-180,h//2-180),tcol,F_XL)
    put_kr(frame,f'카운터 {_score}번',(w//2-100,h//2-80),(100,100,255),F_LG)
    _stats_lines(frame,w,h//2+0)
    put_kr(frame,'R = 재시작  /  Q = 종료',(w//2-180,h//2+100),(140,140,140),F_MD)

def highlight_arm(frame, kp_m, sc, arm, col):
    idxs=spatial_arms(kp_m)['r' if arm=='RIGHT' else 'l']
    for i in range(len(idxs)-1):
        a,b=idxs[i],idxs[i+1]
        if sc[a]>VIS_MIN and sc[b]>VIS_MIN:
            cv2.line(frame,(int(kp_m[a][0]),int(kp_m[a][1])),
                     (int(kp_m[b][0]),int(kp_m[b][1])),col,6)
    if sc[idxs[2]]>VIS_MIN:
        cv2.circle(frame,(int(kp_m[idxs[2]][0]),int(kp_m[idxs[2]][1])),12,col,-1)

def draw_center_line(frame, kp_m, sw, h):
    sh_cx=int((kp_m[KP_L_SH][0]+kp_m[KP_R_SH][0])/2)
    for y in range(0,h,24):
        cv2.line(frame,(sh_cx,y),(sh_cx,min(y+12,h)),(90,90,90),1)
    nx,ny=int(kp_m[KP_NOSE][0]),int(kp_m[KP_NOSE][1])
    dev=(nx-sh_cx)/sw
    col=(0,200,255) if abs(dev)>=SLIP_THRESH else (80,220,80)
    cv2.circle(frame,(nx,ny),10,col,2)
    if abs(dev)>0.10:
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

    elif _gstate=='SPEED_ALERT':
        draw_speed_alert(disp,w,h)
        if now-_phase_start>=SPEED_ALERT_DUR:
            _gstate='PLAYING'; start_round_combo()

    elif _gstate=='PLAYING':
        attack=_combo[_combo_idx]
        elapsed=now-_phase_start

        draw_hud(disp,w,h)
        if kp_m is not None:
            draw_center_line(disp,kp_m,sw,h)

        # ── 가드 경고 (WARN·DEFEND 공통) ────────────────────────
        if kp_m is not None and _sub in ('WARN','DEFEND'):
            if not check_guard(kp_m, sw):
                put_kr(disp,'★ 가드 올려! ★',(w//2-140,h-75),(0,60,255),F_MD)

        # ── WARN: 공격 예고 ──────────────────────────────────────
        if _sub=='WARN':
            draw_overlay(disp,0.20,(20,0,0))
            draw_attack_arrow(disp,w,h,attack)
            timer_bar(disp,w,h,elapsed,_warn_dur,(0,60,255))
            if elapsed>=_warn_dur:
                _sub='DEFEND'; _phase_start=now
                _defend_phase_start=now   # 반응속도 측정 시작

        # ── DEFEND: 방어 입력 ────────────────────────────────────
        elif _sub=='DEFEND':
            draw_defend_phase(disp,w,h,attack,elapsed)

            if kp_m is not None and not _defended:
                defense=get_defense(kp_m,sw)
                if defense and defense in VALID_DEF.get(attack,{}):
                    _react_times.append(now-_defend_phase_start)  # 반응속도 기록
                    _defended=True
                    _counter_arm=VALID_DEF[attack][defense]
                    arm_kw='RIGHT' if 'R' in defense else 'LEFT'
                    highlight_arm(disp,kp_m,sc,arm_kw,(0,255,100))
                    if _combo_idx<len(_combo)-1:
                        _combo_idx+=1; start_attack()
                    else:
                        _sub='COUNTER'; _phase_start=now; _prev_kp_m=None

            if elapsed>=_defend_dur and _sub=='DEFEND':
                _p_hp=max(0,_p_hp-_p_dmg)
                _result_ok=False
                _sub='RESULT'; _phase_start=now
                if _p_hp<=0: _gstate='LOSE'

        # ── COUNTER: 카운터 기회 ─────────────────────────────────
        elif _sub=='COUNTER':
            draw_counter_phase(disp,w,h,_counter_arm,elapsed)

            if kp_m is not None:
                if elapsed<COUNTER_DELAY:
                    _prev_kp_m=kp_m.copy()
                    set_punch_baseline(kp_m)  # 딜레이 끝까지 갱신 → 안정 기준점 확보
                elif not _countered:
                    punch=detect_punch(kp_m,sw)
                    if punch==_counter_arm:
                        _countered=True; _result_ok=True
                        _score+=1; _ai_hp=max(0,_ai_hp-1)
                        highlight_arm(disp,kp_m,sc,_counter_arm,(0,255,200))
                        _sub='RESULT'; _phase_start=now
                        if _ai_hp<=0: _gstate='WIN'
                else:
                    detect_punch(kp_m,sw)

            if elapsed>=COUNTER_DUR and _sub=='COUNTER':
                _result_ok=True
                _sub='RESULT'; _phase_start=now

        # ── RESULT: 결과 표시 ────────────────────────────────────
        elif _sub=='RESULT':
            draw_result_flash(disp,w,h,_result_ok,perfect=(_result_ok and _countered))
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
    elif key==ord('r'):       _gstate='DIFF_SELECT'
    elif _gstate=='DIFF_SELECT':
        if   key==ord('1'): start_game('EASY')
        elif key==ord('2'): start_game('NORMAL')
        elif key==ord('3'): start_game('HARD')
        elif key==ord('4'): start_game('EXTREME')

cap.release()
cv2.destroyAllWindows()
