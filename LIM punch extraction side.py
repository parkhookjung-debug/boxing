"""
LIM punch extraction side.py
LIM1.mp4 / LIM2.mp4 / LIM5.mp4 (측면 뷰) 에서 잽/크로스/훅/어퍼컷 자동 감지
→ LIM_punch_DNA_side.csv 저장

실행 순서
  1. LIM data extraction.py  (전체 데이터 추출)
  2. 이 파일

측면 뷰 분류 기준
  - 어퍼컷 : 손목이 위로 이동 (dy < 0)
  - 훅     : 팔꿈치가 어깨 높이 근처
  - 잽     : 앞발 쪽 손목(lead hand)이 빠르게 이동
  - 크로스 : 뒷발 쪽 손목(rear hand)이 빠르게 이동
"""

import csv, math, os
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = ['LIM_full_data1.csv', 'LIM_full_data2.csv', 'LIM_full_data5.csv']
OUT_PATH   = os.path.join(BASE_DIR, 'LIM_punch_DNA_side.csv')

# ── 파라미터 ──────────────────────────────────────────────────────
VEL_THRESH        = 0.022   # 측면 뷰 기준 속도 임계
MIN_PEAK_GAP      = 12
WINDOW_BEFORE     = 8
WINDOW_AFTER      = 6
HOOK_ELBOW_THRESH = 0.10    # (elbow_y - sh_y)/sw 이 값보다 작으면 훅 후보


# ══════════════════════════════════════════════════════════════════
# CSV 읽기
# ══════════════════════════════════════════════════════════════════
def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def g(row, name, axis):
    return row[f'{name}_{axis}']


# ══════════════════════════════════════════════════════════════════
# 기하 헬퍼 (측면 뷰용)
# ══════════════════════════════════════════════════════════════════
def sw_side(row):
    """측면 뷰 어깨폭: 3D(X²+Z²) 거리 — 측면에서도 정확"""
    dx = g(row,'right_shoulder','x') - g(row,'left_shoulder','x')
    dz = g(row,'right_shoulder','z') - g(row,'left_shoulder','z')
    return math.sqrt(dx*dx + dz*dz) + 1e-6

def dist2d(row, a, b):
    dx = g(row,a,'x') - g(row,b,'x')
    dy = g(row,a,'y') - g(row,b,'y')
    return math.sqrt(dx*dx + dy*dy)

def angle3(ax,ay, bx,by, cx,cy):
    bax,bay = ax-bx, ay-by
    bcx,bcy = cx-bx, cy-by
    dot = bax*bcx + bay*bcy
    mag = math.sqrt(bax**2+bay**2)*math.sqrt(bcx**2+bcy**2)+1e-9
    return math.degrees(math.acos(max(-1,min(1,dot/mag))))

def lead_wrist(row):
    """측면 뷰: x좌표가 더 앞(작은 쪽)인 손목이 앞손(lead=잽)"""
    lx = g(row, 'left_wrist', 'x')
    rx = g(row, 'right_wrist', 'x')
    return ('left_wrist', 'right_wrist') if lx < rx else ('right_wrist', 'left_wrist')


# ══════════════════════════════════════════════════════════════════
# 속도 & 피크 감지
# ══════════════════════════════════════════════════════════════════
def compute_velocities(rows):
    vel_l, vel_r = [0.0], [0.0]
    for i in range(1, len(rows)):
        dlx = g(rows[i],'left_wrist','x')  - g(rows[i-1],'left_wrist','x')
        dly = g(rows[i],'left_wrist','y')  - g(rows[i-1],'left_wrist','y')
        drx = g(rows[i],'right_wrist','x') - g(rows[i-1],'right_wrist','x')
        dry = g(rows[i],'right_wrist','y') - g(rows[i-1],'right_wrist','y')
        vel_l.append(math.sqrt(dlx**2+dly**2))
        vel_r.append(math.sqrt(drx**2+dry**2))
    return vel_l, vel_r

def find_peaks(vels, thresh, min_gap):
    peaks = []
    for i in range(2, len(vels)-2):
        if (vels[i] >= thresh and
                vels[i] >= vels[i-1] and vels[i] >= vels[i+1] and
                vels[i] >= vels[i-2] and vels[i] >= vels[i+2]):
            if not peaks or i - peaks[-1] >= min_gap:
                peaks.append(i)
    return peaks


# ══════════════════════════════════════════════════════════════════
# 측면 뷰 펀치 분류
# ══════════════════════════════════════════════════════════════════
def classify_punch_side(rows, peak_idx, moving_wrist):
    """
    측면 뷰 기준 분류
    • 어퍼컷 : 손목이 위로 이동 (dy < 0)
    • 훅     : 팔꿈치가 어깨 높이 근처
    • 잽     : lead hand(앞손)가 빠르게 이동
    • 크로스 : rear hand(뒷손)가 빠르게 이동
    """
    start = max(0, peak_idx - WINDOW_BEFORE)
    end   = min(len(rows)-1, peak_idx + WINDOW_AFTER)
    row_p = rows[peak_idx]
    sw    = sw_side(row_p)
    sh_cy = (g(row_p,'left_shoulder','y') + g(row_p,'right_shoulder','y')) / 2

    elbow_name  = moving_wrist.replace('wrist', 'elbow')
    elbow_y_rel = (g(row_p, elbow_name, 'y') - sh_cy) / sw

    wy_s = g(rows[start], moving_wrist, 'y')
    wy_e = g(rows[end],   moving_wrist, 'y')
    wx_s = g(rows[start], moving_wrist, 'x')
    wx_e = g(rows[end],   moving_wrist, 'x')
    dy   = wy_s - wy_e   # 위로 = 양수
    dx   = abs(wx_e - wx_s)

    # 1) 어퍼컷: 위쪽 이동이 지배적
    if dy > dx * 1.1 and dy > 0.025:
        return 'uppercut', moving_wrist

    # 2) 훅: 팔꿈치가 어깨 높이
    if elbow_y_rel < HOOK_ELBOW_THRESH:
        return 'hook', moving_wrist

    # 3) 잽/크로스: 측면 뷰에서 앞손/뒷손 구분
    front_w, _ = lead_wrist(row_p)
    if moving_wrist == front_w:
        return 'jab', moving_wrist
    else:
        return 'cross', moving_wrist


# ══════════════════════════════════════════════════════════════════
# 메트릭 추출
# ══════════════════════════════════════════════════════════════════
def find_extension_peak(rows, peak_idx, wrist_name):
    shoulder_name = wrist_name.replace('wrist', 'shoulder')
    end = min(len(rows)-1, peak_idx + WINDOW_AFTER*2)
    best_idx  = peak_idx
    best_dist = dist2d(rows[peak_idx], shoulder_name, wrist_name)
    for i in range(peak_idx+1, end+1):
        d = dist2d(rows[i], shoulder_name, wrist_name)
        if d > best_dist:
            best_dist = d; best_idx = i
    return best_idx

def extract_metrics(rows, peak_idx, wrist_name, punch_type):
    ext_idx = find_extension_peak(rows, peak_idx, wrist_name)
    row   = rows[ext_idx]
    sw    = sw_side(row)
    sh_cy = (g(row,'left_shoulder','y') + g(row,'right_shoulder','y')) / 2
    sh_cx = (g(row,'left_shoulder','x') + g(row,'right_shoulder','x')) / 2
    hi_cx = (g(row,'left_hip','x')      + g(row,'right_hip','x'))      / 2

    elbow_name    = wrist_name.replace('wrist', 'elbow')
    shoulder_name = wrist_name.replace('wrist', 'shoulder')

    arm_ext   = dist2d(row, shoulder_name, wrist_name) / sw
    elbow_h   = (g(row, elbow_name, 'y') - sh_cy) / sw
    wrist_h   = (g(row, wrist_name, 'y') - sh_cy) / sw
    lean      = (sh_cx - hi_cx) / sw
    elbow_ang = angle3(
        g(row,shoulder_name,'x'), g(row,shoulder_name,'y'),
        g(row,elbow_name,'x'),    g(row,elbow_name,'y'),
        g(row,wrist_name,'x'),    g(row,wrist_name,'y'),
    )
    dip = 0.0
    if punch_type == 'uppercut':
        hip_ys = [g(rows[max(0,peak_idx-i)],'left_hip','y') for i in range(WINDOW_BEFORE)]
        dip = (max(hip_ys)-min(hip_ys)) / sw

    return {
        'arm_extension' : round(arm_ext,   4),
        'elbow_height'  : round(elbow_h,   4),
        'wrist_height'  : round(wrist_h,   4),
        'lean_forward'  : round(lean,      4),
        'elbow_angle'   : round(elbow_ang, 2),
        'dip'           : round(dip,       4),
    }


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
all_punches = {'jab':[],'cross':[],'hook':[],'uppercut':[]}

for fname in DATA_FILES:
    fpath = os.path.join(BASE_DIR, fname)
    if not os.path.exists(fpath):
        print(f"[건너뜀] {fname} 없음 — LIM data extraction.py 먼저 실행하세요")
        continue

    rows = read_csv(fpath)
    vel_l, vel_r = compute_velocities(rows)
    peaks_l = find_peaks(vel_l, VEL_THRESH, MIN_PEAK_GAP)
    peaks_r = find_peaks(vel_r, VEL_THRESH, MIN_PEAK_GAP)

    counts = {'jab':0,'cross':0,'hook':0,'uppercut':0}
    used   = set()

    for peak_idx, wrist in ([(p,'left_wrist') for p in peaks_l] +
                             [(p,'right_wrist') for p in peaks_r]):
        if peak_idx in used: continue
        used.add(peak_idx)
        pt, w_name = classify_punch_side(rows, peak_idx, wrist)
        m = extract_metrics(rows, peak_idx, w_name, pt)
        m['source'] = fname; m['frame'] = int(rows[peak_idx]['frame_number'])
        all_punches[pt].append(m)
        counts[pt] += 1

    print(f"{fname}: 잽={counts['jab']}  크로스={counts['cross']}  "
          f"훅={counts['hook']}  어퍼={counts['uppercut']}")

# ── 평균 & 저장 ────────────────────────────────────────────────────
KEYS = ['arm_extension','elbow_height','wrist_height','lean_forward','elbow_angle','dip']
header_out = ['punch_type','count'] + [f'{k}_avg' for k in KEYS] + [f'{k}_std' for k in KEYS]
rows_out   = []

print("\n── 측면 뷰 펀치 DNA ──────────────────────────────────────")
for ptype, samples in all_punches.items():
    if not samples:
        print(f"  {ptype:10s}: 감지 없음")
        continue
    avg = {k: round(float(np.mean([s[k] for s in samples])),4) for k in KEYS}
    std = {k: round(float(np.std([s[k]  for s in samples])),4) for k in KEYS}
    row = [ptype, len(samples)] + [avg[k] for k in KEYS] + [std[k] for k in KEYS]
    rows_out.append(row)
    print(f"  {ptype:10s} (n={len(samples):3d}): 팔뻗음={avg['arm_extension']:.3f}  "
          f"팔꿈치={avg['elbow_angle']:.1f}°  높이={avg['elbow_height']:+.3f}")

if rows_out:
    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(header_out)
        csv.writer(f).writerows(rows_out)
    print(f"\n저장 완료 → {OUT_PATH}")
else:
    print("\n[경고] 감지된 펀치 없음 — LIM data extraction.py 를 먼저 실행하세요")
