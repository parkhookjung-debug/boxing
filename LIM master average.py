"""
LIM master average.py
────────────────────
LIM_full_data1~3.csv 를 읽어 LIM의 복싱 자세 DNA를 계산합니다.
측면 촬영 기준 — 어깨폭은 3D(X²+Z²) 거리로 계산합니다.

출력: LIM_DNA.csv  (한 행짜리 프로필)

측정 항목
─────────
정규화 기준
  sw_3d = sqrt((r_sh_x - l_sh_x)² + (r_sh_z - l_sh_z)²)  ← 측면도 정확

가드
  guard_l_ydiff   : (왼 손목 Y - 왼 어깨 Y) / sw   (음수 = 손목이 어깨보다 위)
  guard_r_ydiff   : (오른 손목 Y - 오른 어깨 Y) / sw
  guard_l_zdiff   : (왼 손목 Z - 왼 어깨 Z) / sw   (음수 = 손목이 앞으로 나옴)
  guard_r_zdiff   : (오른 손목 Z - 오른 어깨 Z) / sw
  guard_l_elbow_y : (왼 팔꿈치 Y - 왼 어깨 Y) / sw
  guard_r_elbow_y : (오른 팔꿈치 Y - 오른 어깨 Y) / sw

스텝 / 스탠스
  stance_3d_ratio  : 3D 발목 간격 / sw  (어느 각도서든 정확한 발 간격)
  stance_step_x    : |앞발 X - 뒷발 X| / sw  (측면 앞뒤 간격)
  knee_bend_l      : 왼쪽 무릎 각도 (도) — 굽힘 판단
  knee_bend_r      : 오른쪽 무릎 각도 (도)

자세
  lean_forward     : (어깨 X - 엉덩이 X) / sw  (상체 전방 기울기, 측면 핵심)
  head_y_ratio     : (코 Y - 어깨 중간 Y) / sw
  head_fwd_z       : (코 Z - 어깨 중간 Z) / sw  (코가 어깨보다 앞으로 나온 정도)
  shoulder_tilt    : (오른 어깨 Y - 왼 어깨 Y) / sw
"""

import csv
import os
import math
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LIM3.mp4 = 정면 촬영 → 측면 기준 DNA에서 제외
CSV_FILES = [
    os.path.join(BASE_DIR, "LIM_full_data1.csv"),
    os.path.join(BASE_DIR, "LIM_full_data2.csv"),
]
OUTPUT_DNA = os.path.join(BASE_DIR, "LIM_DNA.csv")

# ── 데이터 수집 버킷 ──────────────────────────────────────────────
buckets = {
    # 가드
    'guard_l_ydiff'  : [],
    'guard_r_ydiff'  : [],
    'guard_l_zdiff'  : [],
    'guard_r_zdiff'  : [],
    'guard_l_elbow_y': [],
    'guard_r_elbow_y': [],
    # 스탠스
    'stance_3d_ratio': [],
    'stance_step_x'  : [],
    'knee_bend_l'    : [],
    'knee_bend_r'    : [],
    # 가드 위치 (측면: 손목 X - 어깨 X, 앞으로 나온 정도)
    'guard_l_xfwd'   : [],   # 왼 손목 X - 어깨 중간 X  (앞으로 나온 거리)
    'guard_r_xfwd'   : [],   # 오른 손목 X - 어깨 중간 X
    'guard_lr_xdiff' : [],   # 앞손X - 뒷손X  (두 손의 앞뒤 간격)
    # 자세
    'lean_forward'   : [],
    'head_y_ratio'   : [],
    'head_fwd_z'     : [],
    'shoulder_tilt'  : [],
}

VISIBILITY_MIN = 0.45

def safe(row, name, axis):
    return float(row[f"{name}_{axis}"])

def vis(row, name):
    return float(row[f"{name}_v"])

def angle3(ax, ay, bx, by, cx, cy):
    """b 꼭짓점 각도 (도)"""
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag = math.sqrt(bax**2 + bay**2) * math.sqrt(bcx**2 + bcy**2) + 1e-9
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))

total_frames = 0
used_frames  = 0

for csv_path in CSV_FILES:
    if not os.path.exists(csv_path):
        print(f"[건너뜀] 파일 없음: {os.path.basename(csv_path)}")
        continue

    print(f"읽는 중: {os.path.basename(csv_path)} ...", end=' ', flush=True)
    count = 0

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_frames += 1

            # 핵심 관절 가시성
            needed = ['left_shoulder', 'right_shoulder',
                      'left_wrist', 'right_wrist',
                      'left_elbow', 'right_elbow',
                      'left_hip', 'right_hip',
                      'left_knee', 'right_knee',
                      'left_ankle', 'right_ankle', 'nose']
            if any(vis(row, n) < VISIBILITY_MIN for n in needed):
                continue

            # ── 기준 좌표 ────────────────────────────────────────
            l_sh_x = safe(row, 'left_shoulder',  'x')
            l_sh_y = safe(row, 'left_shoulder',  'y')
            l_sh_z = safe(row, 'left_shoulder',  'z')
            r_sh_x = safe(row, 'right_shoulder', 'x')
            r_sh_y = safe(row, 'right_shoulder', 'y')
            r_sh_z = safe(row, 'right_shoulder', 'z')
            sh_cx  = (l_sh_x + r_sh_x) / 2
            sh_cy  = (l_sh_y + r_sh_y) / 2
            sh_cz  = (l_sh_z + r_sh_z) / 2

            # 3D 어깨폭 (정면/측면 모두 정확)
            sw = math.sqrt((r_sh_x - l_sh_x)**2 + (r_sh_z - l_sh_z)**2) + 1e-6

            nose_x = safe(row, 'nose', 'x')
            nose_y = safe(row, 'nose', 'y')
            nose_z = safe(row, 'nose', 'z')

            l_wr_x = safe(row, 'left_wrist',  'x')
            l_wr_y = safe(row, 'left_wrist',  'y')
            l_wr_z = safe(row, 'left_wrist',  'z')
            r_wr_x = safe(row, 'right_wrist', 'x')
            r_wr_y = safe(row, 'right_wrist', 'y')
            r_wr_z = safe(row, 'right_wrist', 'z')

            l_el_x = safe(row, 'left_elbow',  'x')
            l_el_y = safe(row, 'left_elbow',  'y')
            r_el_x = safe(row, 'right_elbow', 'x')
            r_el_y = safe(row, 'right_elbow', 'y')

            l_hi_x = safe(row, 'left_hip',  'x')
            l_hi_y = safe(row, 'left_hip',  'y')
            r_hi_x = safe(row, 'right_hip', 'x')
            r_hi_y = safe(row, 'right_hip', 'y')
            hi_cx  = (l_hi_x + r_hi_x) / 2

            l_kn_x = safe(row, 'left_knee',  'x')
            l_kn_y = safe(row, 'left_knee',  'y')
            r_kn_x = safe(row, 'right_knee', 'x')
            r_kn_y = safe(row, 'right_knee', 'y')

            l_an_x = safe(row, 'left_ankle',  'x')
            l_an_y = safe(row, 'left_ankle',  'y')
            l_an_z = safe(row, 'left_ankle',  'z')
            r_an_x = safe(row, 'right_ankle', 'x')
            r_an_y = safe(row, 'right_ankle', 'y')
            r_an_z = safe(row, 'right_ankle', 'z')

            # ── 가드 ─────────────────────────────────────────────
            buckets['guard_l_ydiff'].append((l_wr_y - l_sh_y) / sw)
            buckets['guard_r_ydiff'].append((r_wr_y - r_sh_y) / sw)
            buckets['guard_l_zdiff'].append((l_wr_z - l_sh_z) / sw)
            buckets['guard_r_zdiff'].append((r_wr_z - r_sh_z) / sw)
            buckets['guard_l_elbow_y'].append((l_el_y - l_sh_y) / sw)
            buckets['guard_r_elbow_y'].append((r_el_y - r_sh_y) / sw)
            # 가드 위치: 손목이 어깨 중간 X 기준으로 얼마나 앞에 있는지 (측면)
            buckets['guard_l_xfwd'].append((l_wr_x - sh_cx) / sw)
            buckets['guard_r_xfwd'].append((r_wr_x - sh_cx) / sw)
            # 앞손 vs 뒷손 X 간격 (앞손이 더 앞에 있어야 함)
            buckets['guard_lr_xdiff'].append(abs(l_wr_x - r_wr_x) / sw)

            # ── 스탠스 ───────────────────────────────────────────
            ankle_3d = math.sqrt((r_an_x - l_an_x)**2 + (r_an_z - l_an_z)**2)
            buckets['stance_3d_ratio'].append(ankle_3d / sw)
            buckets['stance_step_x'].append(abs(r_an_x - l_an_x) / sw)

            # 무릎 각도: 엉덩이-무릎-발목
            ang_l = angle3(l_hi_x, l_hi_y, l_kn_x, l_kn_y, l_an_x, l_an_y)
            ang_r = angle3(r_hi_x, r_hi_y, r_kn_x, r_kn_y, r_an_x, r_an_y)
            buckets['knee_bend_l'].append(ang_l)
            buckets['knee_bend_r'].append(ang_r)

            # ── 자세 ─────────────────────────────────────────────
            # 상체 전방 기울기: 어깨 X - 엉덩이 X (측면에서 핵심)
            buckets['lean_forward'].append((sh_cx - hi_cx) / sw)
            buckets['head_y_ratio'].append((nose_y - sh_cy) / sw)
            buckets['head_fwd_z'].append((nose_z - sh_cz) / sw)
            buckets['shoulder_tilt'].append((r_sh_y - l_sh_y) / sw)

            count += 1
            used_frames += 1

    print(f"{count}프레임 사용")

print(f"\n총 {total_frames}프레임 중 {used_frames}프레임 유효 사용")

# ── 평균 계산 ─────────────────────────────────────────────────────
dna = {}
print("\n┌─── LIM DNA 프로필 (측면 기준) ──────────────────────────┐")
for key, vals in buckets.items():
    if vals:
        avg = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        dna[key]          = avg
        dna[f"{key}_std"] = std
        unit = '°' if 'knee' in key else ''
        print(f"│  {key:<22} avg={avg:+7.3f}{unit}   std={std:.3f}")
    else:
        dna[key]          = 0.0
        dna[f"{key}_std"] = 0.0
        print(f"│  {key:<22} (데이터 없음)")
print("└──────────────────────────────────────────────────────────┘")

# ── CSV 저장 ─────────────────────────────────────────────────────
with open(OUTPUT_DNA, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(dna.keys()))
    writer.writeheader()
    writer.writerow(dna)

print(f"\nLIM DNA 저장 완료: {OUTPUT_DNA}")
print("다음 단계: LIM coach 1.py 를 실행하세요.")
