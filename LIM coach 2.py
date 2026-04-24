"""
LIM coach 2.py — 잽 / 크로스 전용 타격 코치
─────────────────────────────────────────────
RTMPose (RTMO-s) 기반. 정면 카메라, 오르토독스 스탠스.
오른손=잽(JAB), 왼손=크로스(CROSS).

설치
  pip install rtmlib onnxruntime

기능
  ● 잽 / 크로스 자동 감지 및 횟수 카운트
  ● 타격 직후 LIM DNA 비교 form 리포트
  ● 손목 궤적 잔상 (잽=청색, 크로스=주황)
  ● 기준 자세 가이드라인 (Ghost Guide)
  ● Garcia 스타일 자세 점수 패널

단축키
  Q / ESC   종료
  R         카운터 초기화
  G         가이드 토글
  S         스냅샷 저장
"""

import cv2
import numpy as np
import math, os, csv, time
from collections import deque
from PIL import ImageFont, ImageDraw, Image as PILImage

try:
    from rtmlib import RTMO
    RTM_AVAILABLE = True
except ImportError:
    RTM_AVAILABLE = False

if not RTM_AVAILABLE:
    raise SystemExit(
        "[오류] rtmlib가 없습니다.\n"
        "pip install rtmlib onnxruntime  을 실행하고 재시작하세요."
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════
# 모델 초기화 (첫 실행 시 ONNX 자동 다운로드 ~30MB)
# ══════════════════════════════════════════════════════════════════
RTMO_S_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
    'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip'
)
print("RTMPose (RTMO-s) 로드 중... (첫 실행 시 자동 다운로드)")
pose_model = RTMO(RTMO_S_URL, backend='onnxruntime', device='cpu')
print("모델 준비 완료")

# ══════════════════════════════════════════════════════════════════
# COCO 17 keypoint 인덱스
# ══════════════════════════════════════════════════════════════════
KP_NOSE = 0
KP_L_SH = 5;  KP_R_SH = 6
KP_L_EL = 7;  KP_R_EL = 8
KP_L_WR = 9;  KP_R_WR = 10
KP_L_HI = 11; KP_R_HI = 12
KP_L_KN = 13; KP_R_KN = 14
KP_L_AN = 15; KP_R_AN = 16

COCO_CONNECTIONS = [
    (KP_L_SH, KP_R_SH),
    (KP_L_SH, KP_L_EL), (KP_L_EL, KP_L_WR),
    (KP_R_SH, KP_R_EL), (KP_R_EL, KP_R_WR),
    (KP_L_SH, KP_L_HI), (KP_R_SH, KP_R_HI),
    (KP_L_HI, KP_R_HI),
    (KP_L_HI, KP_L_KN), (KP_L_KN, KP_L_AN),
    (KP_R_HI, KP_R_KN), (KP_R_KN, KP_R_AN),
    (KP_NOSE, KP_L_SH), (KP_NOSE, KP_R_SH),
]
NEEDED_KP = [KP_L_SH, KP_R_SH, KP_L_WR, KP_R_WR,
             KP_L_EL, KP_R_EL, KP_L_HI, KP_R_HI,
             KP_L_AN, KP_R_AN, KP_NOSE]

VIS_MIN = 0.30

# 스탠스: orthodox = 오른손(잽), 왼손(크로스)
# 정면 원본 프레임 기준: 사람 오른손이 이미지 왼쪽 → RTMO가 KP_L로 레이블
STANCE = 'orthodox'
JAB_WR,   JAB_SH,   JAB_EL   = KP_L_WR, KP_L_SH, KP_L_EL
CROSS_WR, CROSS_SH, CROSS_EL = KP_R_WR, KP_R_SH, KP_R_EL

# ══════════════════════════════════════════════════════════════════
# DNA 로드
# ══════════════════════════════════════════════════════════════════
def load_punch_dna(path):
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ptype = row['punch_type']
            result[ptype] = {k: float(v) for k, v in row.items()
                             if k not in ('punch_type', 'count')}
    return result

PUNCH_DNA = load_punch_dna(os.path.join(BASE_DIR, 'LIM_punch_DNA.csv'))
DNA_OK    = 'jab' in PUNCH_DNA and 'cross' in PUNCH_DNA
print(f"[DNA] {'잽/크로스 DNA 로드 완료' if DNA_OK else '경고: LIM_punch_DNA.csv 없음'}")

_POSE_DEFAULTS = {
    'guard_l_ydiff': -0.12, 'guard_r_ydiff': -0.08,
    'stance_3d_ratio': 1.50, 'lean_forward': -0.05,
    'head_y_ratio': -0.80,
}

def load_pose_dna(path):
    if not os.path.exists(path):
        return dict(_POSE_DEFAULTS)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            merged = dict(_POSE_DEFAULTS)
            for k, v in row.items():
                if k.endswith('_std'): continue
                try:
                    if k in merged: merged[k] = float(v)
                except ValueError: pass
            return merged
    return dict(_POSE_DEFAULTS)

POSE_DNA = load_pose_dna(os.path.join(BASE_DIR, 'LIM_DNA.csv'))
TOL = {'arm_extension': 0.10, 'elbow_angle': 25.0, 'lean_forward': 0.10}

# ══════════════════════════════════════════════════════════════════
# 폰트
# ══════════════════════════════════════════════════════════════════
def _load_font(size):
    for p in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"]:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()

F_SM = _load_font(16); F_MD = _load_font(24)
F_LG = _load_font(38); F_XL = _load_font(64)

def put_kr(img, text, pos, color, font=None):
    font = font or F_MD
    pil  = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(pos, text, font=font,
                             fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ══════════════════════════════════════════════════════════════════
# 기하 헬퍼 (픽셀 좌표)
# ══════════════════════════════════════════════════════════════════
def sw_px(kp):
    """어깨폭 (픽셀)"""
    dx = kp[KP_R_SH][0] - kp[KP_L_SH][0]
    dy = kp[KP_R_SH][1] - kp[KP_L_SH][1]
    return math.sqrt(dx*dx + dy*dy) + 1e-6

def angle3pt(ax, ay, bx, by, cx, cy):
    bax, bay = ax-bx, ay-by
    bcx, bcy = cx-bx, cy-by
    dot = bax*bcx + bay*bcy
    mag = math.sqrt(bax**2+bay**2) * math.sqrt(bcx**2+bcy**2) + 1e-9
    return math.degrees(math.acos(max(-1, min(1, dot/mag))))

def mirror_kp(kp, w):
    """x좌표 미러링 (디스플레이용)"""
    m = kp.copy()
    m[:, 0] = w - kp[:, 0]
    return m

def draw_skeleton(frame, kp_m, sc):
    for a, b in COCO_CONNECTIONS:
        if sc[a] > VIS_MIN and sc[b] > VIS_MIN:
            cv2.line(frame,
                     (int(kp_m[a][0]), int(kp_m[a][1])),
                     (int(kp_m[b][0]), int(kp_m[b][1])),
                     (80,80,80), 2)
    for i in range(len(kp_m)):
        if sc[i] > VIS_MIN:
            cv2.circle(frame, (int(kp_m[i][0]), int(kp_m[i][1])), 4, (0,200,255), -1)

# ══════════════════════════════════════════════════════════════════
# 상태
# ══════════════════════════════════════════════════════════════════
_trail_jab   = deque(maxlen=22)
_trail_cross = deque(maxlen=22)
_lm_buf      = deque(maxlen=30)   # (kp, sc) 버퍼

_count       = {'jab': 0, 'cross': 0}
_show_ghost  = True
_snap_count  = 0

_punch_state = {'lead': 'IDLE', 'rear': 'IDLE'}
_punch_cd    = {'lead': 0.0, 'rear': 0.0}
PUNCH_CD  = 0.6
VEL_START = 0.18   # sw-normalized 속도 임계값 (낮출수록 민감)
VEL_END   = 0.07
DOM_RATIO = 1.5    # 자기 손이 상대 손보다 이 배율 이상 빠를 때만 펀치 인정

_prev_jab_wr   = None
_prev_cross_wr = None

# 디버그용 실시간 속도 표시
_dbg_jab_vel   = 0.0
_dbg_cross_vel = 0.0

_report    = None
REPORT_DUR = 4.0

READY_FRAMES = 20
_pose_stable = 0
_pose_ready  = False

_ankle_hist = deque(maxlen=30)

# ══════════════════════════════════════════════════════════════════
# 자세 점수
# ══════════════════════════════════════════════════════════════════
def calc_posture_score(kp):
    sw = sw_px(kp)
    l_wr_y = kp[KP_L_WR][1]; r_wr_y = kp[KP_R_WR][1]
    l_sh_y = kp[KP_L_SH][1]; r_sh_y = kp[KP_R_SH][1]
    sh_cy  = (l_sh_y + r_sh_y) / 2
    l_an_x = kp[KP_L_AN][0]; r_an_x = kp[KP_R_AN][0]
    l_an_y = kp[KP_L_AN][1]; r_an_y = kp[KP_R_AN][1]
    nose_y = kp[KP_NOSE][1]

    # ① 가드 (35점)
    l_yd = (l_wr_y - l_sh_y) / sw
    r_yd = (r_wr_y - r_sh_y) / sw
    l_err = abs(l_yd - POSE_DNA['guard_l_ydiff'])
    r_err = abs(r_yd - POSE_DNA['guard_r_ydiff'])
    tol_g = 0.18
    l_pts = 17 if l_err<tol_g*0.5 else 12 if l_err<tol_g else 5 if l_err<tol_g*1.8 else 0
    r_pts = 18 if r_err<tol_g*0.5 else 12 if r_err<tol_g else 5 if r_err<tol_g*1.8 else 0
    guard_score = l_pts + r_pts
    if l_err<tol_g*0.5 and r_err<tol_g*0.5:
        guard_msg, guard_col = 'Guard 완벽!', (0,255,100)
    elif l_err<tol_g and r_err<tol_g:
        guard_msg, guard_col = 'Guard OK', (0,210,100)
    elif l_err >= r_err:
        guard_msg, guard_col = ('왼손 올려!' if l_yd>POSE_DNA['guard_l_ydiff'] else '왼손 내려!'), (0,100,255)
    else:
        guard_msg, guard_col = ('오른손 올려!' if r_yd>POSE_DNA['guard_r_ydiff'] else '오른손 내려!'), (0,100,255)

    # ② 스탠스 (25점)
    ankle_w = math.sqrt((r_an_x-l_an_x)**2 + (r_an_y-l_an_y)**2)
    ankle_ratio = ankle_w / sw
    ref_st  = POSE_DNA['stance_3d_ratio']
    st_err  = abs(ankle_ratio - ref_st)
    stance_score = 25 if st_err<0.20 else 15 if st_err<0.35 else 5 if st_err<0.55 else 0
    if st_err < 0.20:
        stance_msg, stance_col = f'Stance 완벽! ({ankle_ratio:.2f}x)', (0,255,100)
    elif ankle_ratio < ref_st:
        stance_msg, stance_col = f'발 더 벌려 ({ankle_ratio:.2f}x)', (0,165,255)
    else:
        stance_msg, stance_col = f'발 간격 좁혀 ({ankle_ratio:.2f}x)', (0,165,255)

    # ③ 머리 (20점)
    head_y = (nose_y - sh_cy) / sw
    head_err = abs(head_y - POSE_DNA['head_y_ratio'])
    head_score = 20 if head_err<0.15 else 10 if head_err<0.25 else 0
    if head_err < 0.15:
        head_msg, head_col = '머리 자세 좋아!', (0,255,100)
    elif head_y > POSE_DNA['head_y_ratio']:
        head_msg, head_col = '고개 들어!', (0,165,255)
    else:
        head_msg, head_col = '턱 당겨!', (0,165,255)

    # ④ 풋워크 (20점)
    avg_an = (l_an_y + r_an_y) / 2 / sw
    _ankle_hist.append(avg_an)
    bounce = (max(_ankle_hist) - min(_ankle_hist)) if len(_ankle_hist) >= 10 else 0
    bounce_score = 20 if bounce>0.06 else 10 if bounce>0.03 else 0
    if bounce > 0.06:
        bounce_msg, bounce_col = f'풋워크 활발! ({bounce:.3f})', (0,255,100)
    elif bounce > 0.03:
        bounce_msg, bounce_col = f'조금 더 움직여 ({bounce:.3f})', (0,210,100)
    else:
        bounce_msg, bounce_col = f'발을 움직여! ({bounce:.3f})', (0,80,255)

    total = guard_score + stance_score + head_score + bounce_score
    if total >= 85:   grade, g_col = 'S', (0,255,120)
    elif total >= 70: grade, g_col = 'A', (0,200,255)
    elif total >= 50: grade, g_col = 'B', (0,165,255)
    else:             grade, g_col = 'C', (0,60,255)

    return {
        'total': total, 'grade': grade, 'g_col': g_col,
        'items': [
            (guard_score,  35, '[1] Guard',    guard_msg,  guard_col),
            (stance_score, 25, '[2] Stance',   stance_msg, stance_col),
            (head_score,   20, '[3] Head',     head_msg,   head_col),
            (bounce_score, 20, '[4] Footwork', bounce_msg, bounce_col),
        ]
    }

def draw_score_panel(frame, scores):
    h = frame.shape[0]
    px2, py = 10, h - 310
    ov = frame.copy()
    cv2.rectangle(ov, (px2-5, py-5), (px2+255, h-5), (15,15,25), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    def bar(x, y, val, mx, col, label):
        bw = 160
        cv2.rectangle(frame, (x,y), (x+bw, y+13), (50,50,60), -1)
        cv2.rectangle(frame, (x,y), (x+int(bw*val/mx), y+13), col, -1)
        cv2.putText(frame, f'{label}: {val}/{mx}', (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)

    y = py
    for score, mx, label, _, col in scores['items']:
        bar(px2, y, score, mx, col, label)
        y += 35
    cv2.putText(frame, f"TOTAL: {scores['total']}/100  [{scores['grade']}]",
                (px2, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.75, scores['g_col'], 2)

    pil  = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    mx2  = h - 5 - len(scores['items']) * 26
    for i, (_, _, _, msg, col) in enumerate(scores['items']):
        b, g, r = col
        draw.text((px2, mx2+i*26), msg, font=F_SM, fill=(r,g,b))
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ══════════════════════════════════════════════════════════════════
# Motion Trail (미러 픽셀 좌표)
# ══════════════════════════════════════════════════════════════════
def draw_trail(frame):
    for trail, base in [(_trail_cross,(255,160,40)), (_trail_jab,(0,220,255))]:
        pts = list(trail)
        n   = len(pts)
        if n < 2: continue
        for i in range(1, n):
            a   = i / n
            col = tuple(int(c*a) for c in base)
            cv2.line(frame,
                     (int(pts[i-1][0]), int(pts[i-1][1])),
                     (int(pts[i][0]),   int(pts[i][1])),
                     col, max(1, int(4*a)))
        cv2.circle(frame, (int(pts[-1][0]), int(pts[-1][1])), 6, base, -1)

# ══════════════════════════════════════════════════════════════════
# Ghost Guide (미러 키포인트 기준)
# ══════════════════════════════════════════════════════════════════
def draw_ghost(frame, kp_m, sw):
    # JAB_SH=KP_L_SH(5), CROSS_SH=KP_R_SH(6) — 미러 디스플레이에서 각각 올바른 위치
    jab_sh   = kp_m[JAB_SH]
    cross_sh = kp_m[CROSS_SH]
    jab_gy   = jab_sh[1]   + POSE_DNA.get('guard_r_ydiff', -0.10) * sw
    cross_gy = cross_sh[1] + POSE_DNA.get('guard_l_ydiff', -0.10) * sw
    targets = [
        (jab_sh[0],   jab_gy,   (0,220,255),  'JAB 가드'),
        (cross_sh[0], cross_gy, (255,160,40), 'CROSS 가드'),
    ]
    ov = frame.copy()
    for ix, iy, col, _ in targets:
        cx2, cy2 = int(ix), int(iy)
        cv2.circle(ov,(cx2,cy2),24,col,2)
        cv2.circle(ov,(cx2,cy2),5,col,-1)
        cv2.line(ov,(cx2-14,cy2),(cx2+14,cy2),col,1)
        cv2.line(ov,(cx2,cy2-14),(cx2,cy2+14),col,1)
    cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)

# ══════════════════════════════════════════════════════════════════
# 펀치 form 분석
# ══════════════════════════════════════════════════════════════════
def analyse_punch(punch_type):
    global _report
    buf = list(_lm_buf)
    if len(buf) < 3 or not DNA_OK: return

    wr_i, sh_i, el_i = (JAB_WR, JAB_SH, JAB_EL) if punch_type == 'jab' \
                       else (CROSS_WR, CROSS_SH, CROSS_EL)

    best_kp, best_dist = buf[-1][0], 0.0
    for kp, sc in buf:
        if sc[wr_i] < VIS_MIN or sc[sh_i] < VIS_MIN: continue
        dx = kp[wr_i][0] - kp[sh_i][0]
        dy = kp[wr_i][1] - kp[sh_i][1]
        d  = math.sqrt(dx*dx + dy*dy)
        if d > best_dist:
            best_dist = d; best_kp = kp

    kp  = best_kp
    sw  = sw_px(kp)
    wr_x, wr_y = kp[wr_i][0], kp[wr_i][1]
    sh_x, sh_y = kp[sh_i][0], kp[sh_i][1]
    el_x, el_y = kp[el_i][0], kp[el_i][1]
    sh_cx = (kp[KP_L_SH][0] + kp[KP_R_SH][0]) / 2
    hi_cx = (kp[KP_L_HI][0] + kp[KP_R_HI][0]) / 2

    _report = {
        'type'   : punch_type,
        'arm_ext': math.sqrt((wr_x-sh_x)**2 + (wr_y-sh_y)**2) / sw,
        'el_ang' : angle3pt(sh_x, sh_y, el_x, el_y, wr_x, wr_y),
        'lean'   : (sh_cx - hi_cx) / sw,
        'time'   : time.time(),
    }
    _count[punch_type] += 1

# ══════════════════════════════════════════════════════════════════
# 펀치 감지 (상태머신)
# ══════════════════════════════════════════════════════════════════
_pending  = []
EXT_DELAY = 8

def update_punch_detect(kp, sc, now, w):
    global _prev_jab_wr, _prev_cross_wr, _dbg_jab_vel, _dbg_cross_vel

    if sc[JAB_WR] < VIS_MIN or sc[CROSS_WR] < VIS_MIN:
        return

    jx, jy   = kp[JAB_WR][0],   kp[JAB_WR][1]
    cx_, cy_ = kp[CROSS_WR][0], kp[CROSS_WR][1]

    _trail_jab.append((w - jx, jy))
    _trail_cross.append((w - cx_, cy_))

    sw = sw_px(kp)
    d_jab   = (math.sqrt((jx-_prev_jab_wr[0])**2   + (jy-_prev_jab_wr[1])**2)   / sw
               if _prev_jab_wr   else 0.0)
    d_cross = (math.sqrt((cx_-_prev_cross_wr[0])**2 + (cy_-_prev_cross_wr[1])**2) / sw
               if _prev_cross_wr else 0.0)

    _dbg_jab_vel   = d_jab
    _dbg_cross_vel = d_cross

    def check(side, d_self, d_other, punch_type):
        if d_self > VEL_START and d_self > d_other * DOM_RATIO and _punch_state[side] == 'IDLE':
            if now - _punch_cd[side] > PUNCH_CD:
                _punch_state[side] = 'PUNCHING'
        elif d_self < VEL_END and _punch_state[side] == 'PUNCHING':
            _punch_state[side] = 'IDLE'
            _punch_cd[side]    = now
            _pending.append([EXT_DELAY, punch_type])

    check('lead', d_jab,   d_cross, 'jab')
    check('rear',  d_cross, d_jab,  'cross')

    _prev_jab_wr   = (jx, jy)
    _prev_cross_wr = (cx_, cy_)

# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════
def draw_counter(frame, w):
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,90), (8,8,20), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
    pil  = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((30, 12),      f"JAB   {_count['jab']:03d}",   font=F_XL, fill=(0,220,255))
    draw.text((w//2+30, 12), f"CROSS {_count['cross']:03d}", font=F_XL, fill=(255,180,50))
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.line(frame, (w//2, 5), (w//2, 85), (60,60,80), 2)


def draw_report(frame, w, h, now):
    if _report is None: return
    age = now - _report['time']
    if age > REPORT_DUR: return

    alpha   = max(0.0, 1.0 - age / REPORT_DUR)
    pt      = _report['type']
    label   = 'JAB' if pt == 'jab' else 'CROSS'
    col_hdr = (0,220,255) if pt == 'jab' else (255,180,50)
    dna     = PUNCH_DNA.get(pt, {})

    rx, ry = w - 310, 110
    rw, rh = 290, 175
    ov = frame.copy()
    cv2.rectangle(ov, (rx,ry), (rx+rw,ry+rh), (8,8,25), -1)
    cv2.addWeighted(ov, alpha*0.8, frame, 1-alpha*0.8, 0, frame)

    pil  = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    tmp  = PILImage.new('RGBA', pil.size, (0,0,0,0))
    draw = ImageDraw.Draw(tmp)

    def txt(text, x, y, col, font=F_MD):
        b, g, r = col
        draw.text((x,y), text, font=font, fill=(r,g,b,int(alpha*240)))

    txt(f'[ {label} FORM ]', rx+10, ry+6, col_hdr, F_MD)
    metrics = [
        ('팔 뻗음',    _report['arm_ext'], dna.get('arm_extension_avg',0), TOL['arm_extension'], '{:.2f}', ''),
        ('팔꿈치 각도', _report['el_ang'],  dna.get('elbow_angle_avg',0),  TOL['elbow_angle'],   '{:.0f}', '°'),
        ('상체 균형',  _report['lean'],    dna.get('lean_forward_avg',0),  TOL['lean_forward'],  '{:+.2f}',''),
    ]
    for i, (name, val, ref, tol, fmt, unit) in enumerate(metrics):
        y = ry + 42 + i * 44
        ok      = abs(val - ref) <= tol
        bar_col = (0,200,80) if ok else (0,80,255)
        icon    = '✓' if ok else ('▲' if val < ref else '▼')
        txt(name, rx+10, y, (180,180,180), F_SM)
        bx, by2, bw2, bh = rx+10, y+20, 180, 10
        cv2.rectangle(frame,(bx,by2),(bx+bw2,by2+bh),(40,40,60),-1)
        ratio = min(1.0, val/max(ref,0.001)) if ref > 0 else 0.5
        cv2.rectangle(frame,(bx,by2),(bx+int(bw2*ratio),by2+bh),bar_col,-1)
        ref_x = bx + int(bw2 * min(1.0, ref/max(ref,0.001)))
        cv2.line(frame,(ref_x,by2-2),(ref_x,by2+bh+2),(200,200,200),2)
        txt(f'{fmt.format(val)+unit} / {fmt.format(ref)+unit} {icon}',
            rx+200, y+14, (0,200,80) if ok else (80,140,255), F_SM)

    merged = PILImage.alpha_composite(pil, tmp).convert('RGB')
    frame[:] = cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)


def draw_vel_debug(frame, w):
    """우측 상단 — 실시간 손 속도 & 상태 표시 (튜닝용)"""
    jab_st   = _punch_state['lead']
    cross_st = _punch_state['rear']
    lines = [
        (f"JAB   vel={_dbg_jab_vel:.3f}  [{jab_st}]",
         (0,220,255) if jab_st == 'PUNCHING' else (160,160,160)),
        (f"CROSS vel={_dbg_cross_vel:.3f}  [{cross_st}]",
         (255,160,40) if cross_st == 'PUNCHING' else (160,160,160)),
        (f"threshold={VEL_START:.2f}  ratio={DOM_RATIO:.1f}x",
         (100,100,100)),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (w - 400, 110 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)


def draw_punch_flash(frame, punch_type, w, h):
    col = (0,220,255) if punch_type == 'jab' else (255,160,40)
    cv2.rectangle(frame, (0,0), (w-1,h-1), col, 6)


def draw_ready(frame, pose_ok, stable, w, h):
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(w,h),(8,8,25),-1)
    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
    put_kr(frame,'LIM 코치 2 — 잽 / 크로스',(w//2-160,60),(0,200,255),F_LG)
    put_kr(frame,'정면 카메라를 사용하세요 (오르토독스)',(w//2-170,130),(160,160,160),F_SM)
    if pose_ok:
        prog = min(stable/READY_FRAMES, 1.0)
        bx, by, bw2, bh = w//2-150, h//2, 300, 20
        cv2.rectangle(frame,(bx,by),(bx+bw2,by+bh),(50,50,50),-1)
        cv2.rectangle(frame,(bx,by),(bx+int(bw2*prog),by+bh),(0,200,80),-1)
        put_kr(frame, f'자세 인식 중… {int(prog*100)}%', (bx,by-28), (200,255,200), F_SM)
    else:
        put_kr(frame,'전신이 보이도록 서주세요',(w//2-160,h//2),(150,150,255),F_SM)
    tips = ['• 카메라 정면을 향해 서주세요 (전신)',
            '• 가드 자세로 시작 (오른손=잽, 왼손=크로스)',
            '• 단축키: R=초기화  G=가이드  S=스냅샷  Q=종료']
    for i, t in enumerate(tips):
        put_kr(frame, t, (40, h-110+i*30), (120,120,120), F_SM)

# ══════════════════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

_flash = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    now  = time.time()

    # RTMPose 추론 — 원본(unflipped) 프레임으로 정확한 해부학적 좌우 인식
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    kps, scs = pose_model(rgb)   # (N,17,2) pixel, (N,17) score

    # 디스플레이용 미러 프레임 (거울 효과)
    disp = cv2.flip(frame, 1)

    pose_ok = len(kps) > 0
    kp = kps[0] if pose_ok else None
    sc = scs[0] if pose_ok else None

    if pose_ok:
        pose_ok = all(sc[i] > VIS_MIN for i in NEEDED_KP)

    if pose_ok:
        kp_m = mirror_kp(kp, w)  # 디스플레이용 미러 키포인트

    # ── 준비 화면 ────────────────────────────────────────────────
    if not _pose_ready:
        if pose_ok: _pose_stable = min(_pose_stable+1, READY_FRAMES)
        else:        _pose_stable = max(_pose_stable-2, 0)
        if pose_ok:
            draw_skeleton(disp, kp_m, sc)
        draw_ready(disp, pose_ok, _pose_stable, w, h)
        if _pose_stable >= READY_FRAMES: _pose_ready = True
        cv2.imshow('LIM Coach 2', disp)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
        continue

    # ── 메인 ────────────────────────────────────────────────────
    if pose_ok:
        _lm_buf.append((kp, sc))

        for entry in list(_pending):
            entry[0] -= 1
            if entry[0] <= 0:
                _pending.remove(entry)
                analyse_punch(entry[1])
                _flash = (entry[1], now + 0.18)

        sw = sw_px(kp)
        draw_skeleton(disp, kp_m, sc)
        if _show_ghost:
            draw_ghost(disp, kp_m, sw)
        update_punch_detect(kp, sc, now, w)
        draw_trail(disp)
        scores = calc_posture_score(kp)
        draw_score_panel(disp, scores)
    else:
        _pose_stable = max(_pose_stable-2, 0)
        if _pose_stable < READY_FRAMES//2: _pose_ready = False

    if _flash and now < _flash[1]:
        draw_punch_flash(disp, _flash[0], w, h)

    draw_counter(disp, w)
    draw_report(disp, w, h, now)
    draw_vel_debug(disp, w)

    if _show_ghost:
        put_kr(disp, 'G — 가이드 ON', (10, h-30), (180,80,255), F_SM)
    put_kr(disp, 'DNA ✓' if DNA_OK else 'DNA 없음', (w-120, h-30),
           (0,200,80) if DNA_OK else (0,80,255), F_SM)
    put_kr(disp, 'RTMPose', (w-120, h-55), (100,200,100), F_SM)

    cv2.imshow('LIM Coach 2', disp)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27): break
    elif key == ord('r'):
        _count['jab'] = 0; _count['cross'] = 0
        _report = None
    elif key == ord('g'):
        _show_ghost = not _show_ghost
    elif key == ord('s'):
        _snap_count += 1
        fn = os.path.join(BASE_DIR, f'coach2_snap_{_snap_count:03d}.png')
        cv2.imwrite(fn, disp)
        print(f'스냅샷 저장: {fn}')

cap.release()
cv2.destroyAllWindows()
