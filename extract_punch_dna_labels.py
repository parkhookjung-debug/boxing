"""
extract_punch_dna_labels.py
─────────────────────────────────────────────────────────────
수동 라벨(LIMN_labels.csv) 기반 펀치 DNA 추출.

기존 LIM punch extraction.py 는 속도 피크 기반 자동 감지(pseudo-GT)였지만,
이건 label_tool.py 로 직접 라벨링한 LIM1~8_labels.csv 를 사용해서
훨씬 정확한 펀치 DNA 를 만든다.

입력
  LIM_full_dataN.csv  + LIMN_labels.csv  (N=1..8)
  뷰 매핑: LIM1/3/5/7 = front, LIM2/4/6/8 = side

출력
  LIM_punch_DNA.csv        측면 (LIM 2,4,6,8 라벨 기반)
  LIM_punch_DNA_front.csv  정면 (LIM 1,3,5,7 라벨 기반)
  LIM_punch_DNA_all.csv    전체 (8개 모두 합산)
"""

import csv, math, os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 파일 매핑 ─────────────────────────────────────────────────────
# 사용자 명시: LIM1,3는 정면 / LIM2,4는 측면
# LIM5~8 은 일관성을 위해 동일 패턴 가정 (홀수=정면, 짝수=측면)
FRONT_IDS = [1, 3, 5, 7]
SIDE_IDS  = [2, 4, 6, 8]

# 펀치 프레임 직전 N프레임에서 활성 손목 판별
ACTIVE_WIN = 6
# 팔이 가장 뻗어진 시점 검색 윈도우 (피크 +N프레임)
EXT_SEARCH = 6


# ══════════════════════════════════════════════════════════════════
# CSV 로더
# ══════════════════════════════════════════════════════════════════
def load_full_data(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def load_labels(path):
    labels = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            labels.append((int(row['frame_number']), row['punch_type']))
    return labels


# ══════════════════════════════════════════════════════════════════
# 기하 헬퍼
# ══════════════════════════════════════════════════════════════════
def g(row, name, axis):
    return row[f'{name}_{axis}']

def sw_2d(row):
    """2D 어깨폭 (정면에서 잘 작동)"""
    dx = g(row,'right_shoulder','x') - g(row,'left_shoulder','x')
    dy = g(row,'right_shoulder','y') - g(row,'left_shoulder','y')
    return math.sqrt(dx*dx + dy*dy) + 1e-6

def sw_robust(row):
    """측면 뷰용: 어깨폭이 작아지면 어깨-엉덩이 토르소 길이로 대체.
    카메라가 정측면일 때 좌우 어깨가 겹쳐 sw_2d가 0에 가까워지는 문제 보완."""
    sw_v = sw_2d(row)
    sh_y = (g(row,'left_shoulder','y') + g(row,'right_shoulder','y')) / 2
    hi_y = (g(row,'left_hip','y')      + g(row,'right_hip','y'))      / 2
    torso = abs(hi_y - sh_y) + 1e-6
    return max(sw_v, torso)

def dist2d(row, a, b):
    dx = g(row,a,'x') - g(row,b,'x')
    dy = g(row,a,'y') - g(row,b,'y')
    return math.sqrt(dx*dx + dy*dy)

def angle3(ax, ay, bx, by, cx, cy):
    bax, bay = ax-bx, ay-by
    bcx, bcy = cx-bx, cy-by
    dot = bax*bcx + bay*bcy
    mag = math.sqrt(bax**2+bay**2) * math.sqrt(bcx**2+bcy**2) + 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))


# ══════════════════════════════════════════════════════════════════
# 활성 손목 판별 (라벨된 프레임 직전 N프레임에서 더 빨리 움직인 손)
# ══════════════════════════════════════════════════════════════════
def detect_active_wrist(rows, peak_idx, window=ACTIVE_WIN):
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
    """속도 피크 ± 윈도우에서 어깨-손목 거리가 최대인 시점"""
    sh_name = wrist_name.replace('wrist','shoulder')
    start   = max(0, peak_idx - 2)
    end     = min(len(rows)-1, peak_idx + EXT_SEARCH)
    best_idx, best_d = peak_idx, 0.0
    for i in range(start, end + 1):
        d = dist2d(rows[i], sh_name, wrist_name)
        if d > best_d:
            best_d = d
            best_idx = i
    return best_idx


# ══════════════════════════════════════════════════════════════════
# 메트릭 추출 (라벨된 펀치 프레임 기준)
# ══════════════════════════════════════════════════════════════════
def extract_metrics(rows, peak_idx, wrist_name, sw_fn=sw_2d):
    ext_idx = find_extension_peak(rows, peak_idx, wrist_name)
    row     = rows[ext_idx]
    sh_name = wrist_name.replace('wrist','shoulder')
    el_name = wrist_name.replace('wrist','elbow')

    sw_v  = sw_fn(row)
    sh_cy = (g(row,'left_shoulder','y') + g(row,'right_shoulder','y')) / 2
    sh_cx = (g(row,'left_shoulder','x') + g(row,'right_shoulder','x')) / 2
    hi_cx = (g(row,'left_hip','x')      + g(row,'right_hip','x'))      / 2

    arm_ext       = dist2d(row, sh_name, wrist_name) / sw_v
    elbow_height  = (g(row, el_name,    'y') - sh_cy) / sw_v
    wrist_height  = (g(row, wrist_name, 'y') - sh_cy) / sw_v
    lean_forward  = (sh_cx - hi_cx) / sw_v
    elbow_angle   = angle3(g(row, sh_name, 'x'), g(row, sh_name, 'y'),
                           g(row, el_name, 'x'), g(row, el_name, 'y'),
                           g(row, wrist_name,'x'), g(row, wrist_name,'y'))

    # dip: 어퍼컷 등 손목이 어깨 아래로 내려간 정도 (음수)
    dip = max(0.0, (sh_cy - g(row, wrist_name,'y')) / sw_v)

    return {
        'arm_extension': arm_ext,
        'elbow_height':  elbow_height,
        'wrist_height':  wrist_height,
        'lean_forward':  lean_forward,
        'elbow_angle':   elbow_angle,
        'dip':           dip,
    }


# ══════════════════════════════════════════════════════════════════
# 평균/표준편차 계산
# ══════════════════════════════════════════════════════════════════
def aggregate(samples_by_type):
    """
    samples_by_type: {'jab': [{metric: val, ...}, ...], ...}
    반환: 각 펀치 타입의 mean/std
    """
    out = {}
    for pt, samples in samples_by_type.items():
        if not samples: continue
        agg = {'count': len(samples)}
        keys = samples[0].keys()
        for k in keys:
            vals = [s[k] for s in samples]
            mean = sum(vals) / len(vals)
            var  = sum((v-mean)**2 for v in vals) / len(vals)
            agg[f'{k}_avg'] = round(mean, 4)
            agg[f'{k}_std'] = round(math.sqrt(var), 4)
        out[pt] = agg
    return out


# ══════════════════════════════════════════════════════════════════
# CSV 저장
# ══════════════════════════════════════════════════════════════════
PUNCH_ORDER = ['jab','cross','hook','uppercut']
METRIC_ORDER = ['arm_extension','elbow_height','wrist_height',
                'lean_forward','elbow_angle','dip']

def save_dna(agg, out_path):
    fieldnames = ['punch_type', 'count']
    for m in METRIC_ORDER:
        fieldnames.append(f'{m}_avg')
        fieldnames.append(f'{m}_std')

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pt in PUNCH_ORDER:
            if pt not in agg: continue
            row = {'punch_type': pt}
            row.update(agg[pt])
            # 누락된 필드 0으로 채움
            for fn in fieldnames:
                if fn not in row: row[fn] = 0
            w.writerow(row)
    print(f"[OK] {out_path}  ({sum(agg[pt]['count'] for pt in agg)} samples)")


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
def collect_samples(ids, sw_fn=sw_2d):
    """주어진 LIM ID 목록의 라벨된 펀치들에서 메트릭 수집"""
    samples_by_type = defaultdict(list)
    total = 0
    for i in ids:
        data_path  = os.path.join(BASE_DIR, f'LIM_full_data{i}.csv')
        label_path = os.path.join(BASE_DIR, f'LIM{i}_labels.csv')
        if not (os.path.exists(data_path) and os.path.exists(label_path)):
            print(f"  [건너뜀] LIM{i} 데이터/라벨 누락")
            continue
        rows   = load_full_data(data_path)
        labels = load_labels(label_path)
        per_type = defaultdict(int)
        for frame, pt in labels:
            if not (0 <= frame < len(rows)): continue
            wrist = detect_active_wrist(rows, frame)
            metrics = extract_metrics(rows, frame, wrist, sw_fn=sw_fn)
            samples_by_type[pt].append(metrics)
            per_type[pt] += 1
            total += 1
        dist = ', '.join(f'{k}:{v}' for k, v in sorted(per_type.items()))
        print(f"  LIM{i}: {len(labels)}개  ({dist})")
    return samples_by_type, total


def main():
    print("━━━ 펀치 DNA 추출 (수동 라벨 기반) ━━━\n")

    # 측면 (LIM2/4/6/8) — robust sw 사용 (어깨폭 폭주 방지)
    print("[측면 LIM 2,4,6,8]  sw=robust")
    side_samples, side_total = collect_samples(SIDE_IDS, sw_fn=sw_robust)
    side_agg = aggregate(side_samples)
    save_dna(side_agg, os.path.join(BASE_DIR, 'LIM_punch_DNA.csv'))

    # 정면 (LIM1/3/5/7) — 표준 2D 어깨폭
    print("\n[정면 LIM 1,3,5,7]  sw=2d")
    front_samples, front_total = collect_samples(FRONT_IDS, sw_fn=sw_2d)
    front_agg = aggregate(front_samples)
    save_dna(front_agg, os.path.join(BASE_DIR, 'LIM_punch_DNA_front.csv'))

    # 전체 합산
    print("\n[전체 8개]")
    all_samples = defaultdict(list)
    for pt, lst in side_samples.items():  all_samples[pt].extend(lst)
    for pt, lst in front_samples.items(): all_samples[pt].extend(lst)
    all_agg = aggregate(all_samples)
    save_dna(all_agg, os.path.join(BASE_DIR, 'LIM_punch_DNA_all.csv'))

    # 요약
    print(f"\n총 라벨: 측면 {side_total}개 + 정면 {front_total}개 = {side_total+front_total}개")
    print("\n=== 펀치 타입별 평균 (전체) ===")
    print(f"{'type':10s} {'n':>4s} {'arm':>7s} {'elbow_y':>8s} {'wrist_y':>8s} {'lean':>7s} {'ea':>7s} {'dip':>7s}")
    for pt in PUNCH_ORDER:
        if pt not in all_agg: continue
        a = all_agg[pt]
        print(f"{pt:10s} {a['count']:4d} "
              f"{a['arm_extension_avg']:7.3f} "
              f"{a['elbow_height_avg']:8.3f} "
              f"{a['wrist_height_avg']:8.3f} "
              f"{a['lean_forward_avg']:7.3f} "
              f"{a['elbow_angle_avg']:7.1f} "
              f"{a['dip_avg']:7.3f}")


if __name__ == '__main__':
    main()
