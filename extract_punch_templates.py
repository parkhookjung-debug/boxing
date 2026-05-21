"""
extract_punch_templates.py
─────────────────────────────────────────────────────────────
수동 라벨된 펀치 프레임에서 스켈레톤 템플릿 추출.

각 펀치 = 어깨 중심 기준으로 정규화된 9개 키포인트 좌표 벡터.
LIM coach 5.py 가 라이브에서 사용자의 포즈와 1-NN 매칭하여 분류.

출력: punch_templates.csv  (한 행당 하나의 라벨된 펀치)
"""

import csv, math, os
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONT_IDS = [1, 3, 5, 7]
SIDE_IDS  = [2, 4, 6, 8]
ALL_IDS   = sorted(FRONT_IDS + SIDE_IDS)

ACTIVE_WIN = 6  # 활성 손목 판별 윈도우 (프레임)
EXT_SEARCH = 6  # 라벨 프레임 부근에서 확장 피크 검색 범위 (+프레임)

# 캡처할 키포인트 — 어깨 중심 기준으로 (dx, dy) 정규화
KP_NAMES = [
    'left_wrist',  'right_wrist',
    'left_elbow',  'right_elbow',
    'left_shoulder','right_shoulder',
    'nose',
    'left_hip',    'right_hip',
]


def g(row, name, axis):
    return row[f'{name}_{axis}']

def sw_2d(row):
    dx = g(row,'right_shoulder','x') - g(row,'left_shoulder','x')
    dy = g(row,'right_shoulder','y') - g(row,'left_shoulder','y')
    return math.sqrt(dx*dx + dy*dy) + 1e-6

def sw_robust(row):
    """측면 뷰: 어깨가 겹쳐 sw가 0에 가까워질 때 torso로 보완."""
    sw_v = sw_2d(row)
    sh_y = (g(row,'left_shoulder','y') + g(row,'right_shoulder','y')) / 2
    hi_y = (g(row,'left_hip','y')      + g(row,'right_hip','y'))      / 2
    torso = abs(hi_y - sh_y) + 1e-6
    return max(sw_v, torso)


def detect_active_wrist(rows, peak_idx, window=ACTIVE_WIN):
    """라벨된 프레임 직전 N프레임에서 더 빨리 움직인 손목 반환."""
    start = max(1, peak_idx - window)
    dl = dr = 0.0
    for i in range(start, min(peak_idx + 1, len(rows))):
        dlx = g(rows[i],'left_wrist','x')  - g(rows[i-1],'left_wrist','x')
        dly = g(rows[i],'left_wrist','y')  - g(rows[i-1],'left_wrist','y')
        drx = g(rows[i],'right_wrist','x') - g(rows[i-1],'right_wrist','x')
        dry = g(rows[i],'right_wrist','y') - g(rows[i-1],'right_wrist','y')
        dl += math.sqrt(dlx*dlx + dly*dly)
        dr += math.sqrt(drx*drx + dry*dry)
    return 'left_wrist' if dl >= dr else 'right_wrist'


def find_extension_peak(rows, peak_idx, wrist_name):
    """라벨된 프레임 ± 윈도우에서 어깨-손목 거리가 최대인 시점."""
    sh_name = wrist_name.replace('wrist','shoulder')
    start = max(0, peak_idx - 2)
    end   = min(len(rows)-1, peak_idx + EXT_SEARCH)
    best_idx, best_d = peak_idx, 0.0
    for i in range(start, end + 1):
        dx = g(rows[i], wrist_name, 'x') - g(rows[i], sh_name, 'x')
        dy = g(rows[i], wrist_name, 'y') - g(rows[i], sh_name, 'y')
        d = math.sqrt(dx*dx + dy*dy)
        if d > best_d:
            best_d = d
            best_idx = i
    return best_idx


def load_full_data(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def load_labels(path):
    out = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out.append((int(row['frame_number']), row['punch_type']))
    return out


def extract_pose_vector(row, sw_v):
    """프레임에서 9 키포인트의 (dx, dy)를 어깨 중심 기준으로 정규화."""
    sh_cx = (g(row,'left_shoulder','x') + g(row,'right_shoulder','x')) / 2
    sh_cy = (g(row,'left_shoulder','y') + g(row,'right_shoulder','y')) / 2
    vec = []
    for name in KP_NAMES:
        dx = (g(row, name, 'x') - sh_cx) / sw_v
        dy = (g(row, name, 'y') - sh_cy) / sw_v
        vec.extend([round(dx, 4), round(dy, 4)])
    return vec


def main():
    print("━━━ 펀치 템플릿 추출 (스켈레톤 좌표) ━━━\n")
    templates = []

    for i in ALL_IDS:
        data_path  = os.path.join(BASE_DIR, f'LIM_full_data{i}.csv')
        label_path = os.path.join(BASE_DIR, f'LIM{i}_labels.csv')
        if not (os.path.exists(data_path) and os.path.exists(label_path)):
            print(f"  [건너뜀] LIM{i} 누락")
            continue

        rows   = load_full_data(data_path)
        labels = load_labels(label_path)
        view   = 'front' if i in FRONT_IDS else 'side'
        sw_fn  = sw_robust if view == 'side' else sw_2d

        per_type = defaultdict(int)
        for frame, pt in labels:
            if not (0 <= frame < len(rows)): continue

            # 펀치 타입별 활성 손목 강제 (orthodox 가정)
            # jab: 왼손(lead) / cross: 오른손(rear) / hook·uppercut: 속도로 판별
            if pt == 'jab':
                wrist = 'left_wrist'
            elif pt == 'cross':
                wrist = 'right_wrist'
            else:
                wrist = detect_active_wrist(rows, frame)

            side = 'lead' if wrist == 'left_wrist' else 'rear'

            # 확장 피크 시점의 포즈로 템플릿 추출 (라이브와 시점 일치)
            ext_idx = find_extension_peak(rows, frame, wrist)
            sw_v   = sw_fn(rows[ext_idx])
            vec    = extract_pose_vector(rows[ext_idx], sw_v)

            templates.append({
                'file_id': i, 'frame': ext_idx, 'view': view,
                'side': side, 'punch_type': pt, 'vec': vec,
            })
            per_type[pt] += 1
        dist = ', '.join(f'{k}:{v}' for k, v in sorted(per_type.items()))
        print(f"  LIM{i} ({view}): {len(labels)}개  ({dist})")

    # ── CSV 저장 ──
    out_path = os.path.join(BASE_DIR, 'punch_templates.csv')
    fieldnames = ['file_id', 'frame', 'view', 'side', 'punch_type']
    for name in KP_NAMES:
        fieldnames += [f'{name}_dx', f'{name}_dy']

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in templates:
            row = {'file_id': t['file_id'], 'frame': t['frame'],
                   'view': t['view'], 'side': t['side'],
                   'punch_type': t['punch_type']}
            for j, name in enumerate(KP_NAMES):
                row[f'{name}_dx'] = t['vec'][j*2]
                row[f'{name}_dy'] = t['vec'][j*2+1]
            w.writerow(row)

    print(f"\n[OK] {out_path}  ({len(templates)}개 템플릿)")

    # ── 통계 ──
    type_counts = Counter(t['punch_type'] for t in templates)
    side_counts = Counter((t['side'], t['punch_type']) for t in templates)
    view_counts = Counter(t['view'] for t in templates)
    print(f"\n타입별:  {dict(type_counts)}")
    print(f"뷰별:    {dict(view_counts)}")
    print(f"side·타입:")
    for (s, pt), c in sorted(side_counts.items()):
        print(f"    {s:4s} {pt:8s}: {c}")


if __name__ == '__main__':
    main()
