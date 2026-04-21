"""
LIM punch extraction.py
LIM_full_data CSVs에서 잽/크로스/훅/어퍼컷을 자동 감지하고
펀치 타입별 평균 메트릭을 LIM_punch_DNA.csv로 저장합니다.

분류 로직 (측면 뷰 기준)
  잽    : 앞손(X 작은 손)이 빠르게 앞으로 → 팔꿈치가 어깨보다 낮음
  크로스 : 뒷손(X 큰 손)이 빠르게 앞으로 → 팔꿈치가 어깨보다 낮음
  훅    : 어느 손이든 팔꿈치가 어깨 높이에 있고 호 궤적
  어퍼컷 : 손목이 빠르게 위로 이동 (dy 음수)
"""

import csv
import math
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = ['LIM_full_data1.csv', 'LIM_full_data2.csv', 'LIM_full_data3.csv']
OUT_PATH   = os.path.join(BASE_DIR, 'LIM_punch_DNA.csv')

# ── 파라미터 ──────────────────────────────────────────────────────
VEL_THRESH    = 0.025   # 펀치 감지 속도 임계값 (normalized/frame)
MIN_PEAK_GAP  = 12      # 연속 펀치 최소 간격 (프레임)
WINDOW_BEFORE = 8       # 피크 전 분석 윈도우
WINDOW_AFTER  = 4       # 피크 후 분석 윈도우

# 팔꿈치 높이 기준: 어깨 대비 이 값보다 높으면 훅 후보
HOOK_ELBOW_THRESH = -0.05  # (elbow_y - shoulder_y) / sw


# ══════════════════════════════════════════════════════════════════
# CSV 읽기
# ══════════════════════════════════════════════════════════════════
def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def g(row, name, axis):
    return row[f'{name}_{axis}']


# ══════════════════════════════════════════════════════════════════
# 기하 헬퍼
# ══════════════════════════════════════════════════════════════════
def sw(row):
    """3D 어깨폭 (측면 뷰 정확도용)"""
    dx = g(row,'right_shoulder','x') - g(row,'left_shoulder','x')
    dz = g(row,'right_shoulder','z') - g(row,'left_shoulder','z')
    return math.sqrt(dx*dx + dz*dz) + 1e-6

def dist2d(row, a_name, b_name):
    dx = g(row,a_name,'x') - g(row,b_name,'x')
    dy = g(row,a_name,'y') - g(row,b_name,'y')
    return math.sqrt(dx*dx + dy*dy)

def angle3(ax,ay, bx,by, cx,cy):
    """B 꼭짓점 기준 각도"""
    bax, bay = ax-bx, ay-by
    bcx, bcy = cx-bx, cy-by
    dot = bax*bcx + bay*bcy
    mag = math.sqrt(bax**2+bay**2) * math.sqrt(bcx**2+bcy**2) + 1e-9
    return math.degrees(math.acos(max(-1, min(1, dot/mag))))


# ══════════════════════════════════════════════════════════════════
# 피크 감지
# ══════════════════════════════════════════════════════════════════
def compute_velocities(rows):
    """두 손목 각각의 프레임별 속도 반환"""
    vel_l, vel_r = [0.0], [0.0]
    for i in range(1, len(rows)):
        dlx = g(rows[i],'left_wrist','x')  - g(rows[i-1],'left_wrist','x')
        dly = g(rows[i],'left_wrist','y')  - g(rows[i-1],'left_wrist','y')
        drx = g(rows[i],'right_wrist','x') - g(rows[i-1],'right_wrist','x')
        dry = g(rows[i],'right_wrist','y') - g(rows[i-1],'right_wrist','y')
        vel_l.append(math.sqrt(dlx**2 + dly**2))
        vel_r.append(math.sqrt(drx**2 + dry**2))
    return vel_l, vel_r

def find_peaks(velocities, threshold, min_gap):
    peaks = []
    for i in range(2, len(velocities)-2):
        if (velocities[i] >= threshold and
                velocities[i] >= velocities[i-1] and
                velocities[i] >= velocities[i+1] and
                velocities[i] >= velocities[i-2] and
                velocities[i] >= velocities[i+2]):
            if not peaks or i - peaks[-1] >= min_gap:
                peaks.append(i)
    return peaks


# ══════════════════════════════════════════════════════════════════
# 펀치 분류
# ══════════════════════════════════════════════════════════════════
def classify_punch(rows, peak_idx, moving_wrist):
    """
    moving_wrist: 'left_wrist' or 'right_wrist' (빠르게 움직인 손)
    반환: punch_type, wrist_name
    """
    start = max(0, peak_idx - WINDOW_BEFORE)
    end   = min(len(rows)-1, peak_idx + WINDOW_AFTER)

    # ── 앞손/뒷손 판별 (측면 X축) ─────────────────────────────────
    lx = g(rows[peak_idx], 'left_wrist', 'x')
    rx = g(rows[peak_idx], 'right_wrist', 'x')
    lead_wrist = 'left_wrist'  if lx < rx else 'right_wrist'
    rear_wrist = 'right_wrist' if lx < rx else 'left_wrist'

    lead_elbow = lead_wrist.replace('wrist', 'elbow')
    rear_elbow = rear_wrist.replace('wrist', 'elbow')

    # ── 이동 방향 계산 ─────────────────────────────────────────────
    wx_s = g(rows[start], moving_wrist, 'x')
    wy_s = g(rows[start], moving_wrist, 'y')
    wx_e = g(rows[end],   moving_wrist, 'x')
    wy_e = g(rows[end],   moving_wrist, 'y')
    dx = wx_e - wx_s
    dy = wy_s - wy_e   # 위로 = 양수 (image y가 아래로 증가)

    # ── 팔꿈치 높이 (훅 판별 핵심) ────────────────────────────────
    sw_val = sw(rows[peak_idx])
    elbow_name = moving_wrist.replace('wrist', 'elbow')
    sh_cy = (g(rows[peak_idx],'left_shoulder','y') +
             g(rows[peak_idx],'right_shoulder','y')) / 2
    elbow_y_rel = (g(rows[peak_idx], elbow_name, 'y') - sh_cy) / sw_val
    # elbow_y_rel < 0 = 팔꿈치가 어깨 위, > 0 = 아래

    # ── 분류 ──────────────────────────────────────────────────────
    # 1) 훅: 팔꿈치가 어깨 높이 근처 + 전진 성분 적음
    if elbow_y_rel < HOOK_ELBOW_THRESH and abs(dx) < 0.15:
        return 'hook', moving_wrist

    # 2) 어퍼컷: 손목이 빠르게 위로
    if dy > abs(dx) * 1.2 and dy > 0.04:
        return 'uppercut', moving_wrist

    # 3) 잽/크로스: 앞으로 직선
    if moving_wrist == lead_wrist:
        return 'jab', moving_wrist
    else:
        return 'cross', moving_wrist


# ══════════════════════════════════════════════════════════════════
# 메트릭 추출 (피크 시점 기준)
# ══════════════════════════════════════════════════════════════════
def extract_metrics(rows, peak_idx, wrist_name, punch_type):
    row   = rows[peak_idx]
    sw_v  = sw(row)
    sh_cy = (g(row,'left_shoulder','y') + g(row,'right_shoulder','y')) / 2
    sh_cx = (g(row,'left_shoulder','x') + g(row,'right_shoulder','x')) / 2
    hi_cx = (g(row,'left_hip','x')      + g(row,'right_hip','x'))      / 2

    elbow_name    = wrist_name.replace('wrist', 'elbow')
    shoulder_name = wrist_name.replace('wrist', 'shoulder')

    # 팔 뻗음: shoulder → wrist 거리 / sw
    arm_ext = dist2d(row, shoulder_name, wrist_name) / sw_v

    # 팔꿈치 높이: (elbow_y - sh_cy) / sw  (음수 = 위)
    elbow_h = (g(row, elbow_name, 'y') - sh_cy) / sw_v

    # 손목 높이
    wrist_h = (g(row, wrist_name, 'y') - sh_cy) / sw_v

    # 상체 기울기
    lean = (sh_cx - hi_cx) / sw_v

    # 팔꿈치 각도 (shoulder-elbow-wrist)
    elbow_ang = angle3(
        g(row, shoulder_name, 'x'), g(row, shoulder_name, 'y'),
        g(row, elbow_name,    'x'), g(row, elbow_name,    'y'),
        g(row, wrist_name,    'x'), g(row, wrist_name,    'y'),
    )

    # 어퍼컷 전 dip: peak 전 WINDOW_BEFORE 동안 hip 최대 y 변화
    dip = 0.0
    if punch_type == 'uppercut':
        hip_ys = [g(rows[max(0,peak_idx-i)], 'left_hip', 'y') for i in range(WINDOW_BEFORE)]
        dip = (max(hip_ys) - min(hip_ys)) / sw_v

    return {
        'arm_extension' : round(arm_ext,   4),
        'elbow_height'  : round(elbow_h,   4),
        'wrist_height'  : round(wrist_h,   4),
        'lean_forward'  : round(lean,      4),
        'elbow_angle'   : round(elbow_ang, 2),
        'dip'           : round(dip,       4),
    }


# ══════════════════════════════════════════════════════════════════
# 메인: 모든 CSV 처리
# ══════════════════════════════════════════════════════════════════
all_punches = {'jab': [], 'cross': [], 'hook': [], 'uppercut': []}
counts_per_file = {}

for fname in DATA_FILES:
    fpath = os.path.join(BASE_DIR, fname)
    if not os.path.exists(fpath):
        print(f"[건너뜀] {fname} 없음")
        continue

    rows  = read_csv(fpath)
    vel_l, vel_r = compute_velocities(rows)

    peaks_l = find_peaks(vel_l, VEL_THRESH, MIN_PEAK_GAP)
    peaks_r = find_peaks(vel_r, VEL_THRESH, MIN_PEAK_GAP)

    file_counts = {'jab': 0, 'cross': 0, 'hook': 0, 'uppercut': 0}
    used_peaks  = set()

    for peak_idx, wrist in (
        [(p, 'left_wrist')  for p in peaks_l] +
        [(p, 'right_wrist') for p in peaks_r]
    ):
        if peak_idx in used_peaks:
            continue
        used_peaks.add(peak_idx)

        punch_type, w_name = classify_punch(rows, peak_idx, wrist)
        metrics = extract_metrics(rows, peak_idx, w_name, punch_type)
        metrics['source'] = fname
        metrics['frame']  = int(rows[peak_idx]['frame_number'])
        all_punches[punch_type].append(metrics)
        file_counts[punch_type] += 1

    counts_per_file[fname] = file_counts
    total = sum(file_counts.values())
    print(f"{fname}: 총 {total}개 감지  "
          f"잽={file_counts['jab']}  크로스={file_counts['cross']}  "
          f"훅={file_counts['hook']}  어퍼={file_counts['uppercut']}")

# ══════════════════════════════════════════════════════════════════
# 평균 계산 & DNA 저장
# ══════════════════════════════════════════════════════════════════
METRIC_KEYS = ['arm_extension','elbow_height','wrist_height',
               'lean_forward','elbow_angle','dip']

punch_dna = {}
print("\n── 펀치 타입별 평균 ──────────────────────────────────")
for ptype, samples in all_punches.items():
    if not samples:
        print(f"  {ptype:10s}: 감지 없음")
        continue
    avg = {k: round(float(np.mean([s[k] for s in samples])), 4)
           for k in METRIC_KEYS}
    std = {k: round(float(np.std([s[k]  for s in samples])), 4)
           for k in METRIC_KEYS}
    punch_dna[ptype] = {'avg': avg, 'std': std, 'count': len(samples)}
    print(f"  {ptype:10s} (n={len(samples):3d}): "
          f"팔뻗음={avg['arm_extension']:.3f}  "
          f"팔꿈치높이={avg['elbow_height']:+.3f}  "
          f"기울기={avg['lean_forward']:+.3f}  "
          f"팔꿈치각도={avg['elbow_angle']:.1f}°")

# 플랫 CSV 저장 (punch_type + metric_avg + metric_std)
rows_out = []
header_out = ['punch_type', 'count']
for k in METRIC_KEYS:
    header_out += [f'{k}_avg', f'{k}_std']

for ptype, data in punch_dna.items():
    row = [ptype, data['count']]
    for k in METRIC_KEYS:
        row += [data['avg'][k], data['std'][k]]
    rows_out.append(row)

with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header_out)
    writer.writerows(rows_out)

print(f"\n저장 완료 → {OUT_PATH}")

# ══════════════════════════════════════════════════════════════════
# 시각화 (matplotlib 있을 때)
# ══════════════════════════════════════════════════════════════════
if HAS_PLT and all_punches:
    COLORS = {'jab':'#00ccff','cross':'#ff8800','hook':'#ff3366','uppercut':'#88ff00'}
    metrics_to_plot = ['arm_extension','elbow_height','elbow_angle']
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5))
    fig.suptitle('LIM Punch DNA — 타입별 메트릭 비교', fontsize=13)

    for ax, metric in zip(axes, metrics_to_plot):
        for ptype, data in punch_dna.items():
            ax.bar(ptype, data['avg'][metric],
                   yerr=data['std'][metric],
                   color=COLORS[ptype], alpha=0.85,
                   capsize=5, label=ptype)
        ax.set_title(metric)
        ax.set_ylabel('normalized value')
        ax.axhline(0, color='white', linewidth=0.5, linestyle='--')
        ax.set_facecolor('#111122')
        fig.patch.set_facecolor('#0a0a1a')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    plt.tight_layout()
    plt.show()
