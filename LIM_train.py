"""
LIM_train.py — punch classifier 학습

입력: LIM_full_data{1..8}.csv (스켈레톤) + LIM{1..8}_labels.csv (펀치 라벨)
출력: lim_punch_model.pkl (분류기 + 피처 이름 + 클래스)

피처는 어깨폭(sw)으로 정규화하므로 픽셀 좌표/정규 좌표 무관.
라벨 프레임을 임팩트로 가정, 그 시점 + 직전 8프레임 윈도우에서 추출.

실행: python "LIM_train.py"
"""

import os, math, pickle, csv
import numpy as np
from collections import Counter

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
except ImportError:
    raise SystemExit("pip install scikit-learn numpy")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = [(f'LIM_full_data{i}.csv', f'LIM{i}_labels.csv') for i in range(1, 9)]

WINDOW = 8  # 라벨 프레임 직전 N프레임 윈도우

FEATURE_NAMES = [
    'arm_ext', 'elbow_angle',
    'elbow_y', 'elbow_lat', 'wrist_y',
    'ew', 'arm_dy_rise',
    'wr_dx_abs', 'wr_dy_min', 'wr_dy_rise', 'wr_peak_v',
    'opp_wr_dx_abs',  # 반대팔이 가만히 있나 — 가드/펀치 구분
]


def read_csv_dict_list(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def kp_xy(row, name):
    return float(row[f'{name}_x']), float(row[f'{name}_y'])


def sw_row(row):
    lx, ly = kp_xy(row, 'left_shoulder')
    rx, ry = kp_xy(row, 'right_shoulder')
    return math.hypot(rx - lx, ry - ly) + 1e-6


def angle3(ax, ay, bx, by, cx, cy):
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy) + 1e-9
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))


def features(rows, idx, side):
    """rows: list of dict (한 영상). idx: 임팩트 프레임의 리스트 인덱스. side: 'L' or 'R'."""
    row = rows[idx]
    s = sw_row(row)

    if side == 'L':
        wr, el, sh = 'left_wrist', 'left_elbow', 'left_shoulder'
        owr = 'right_wrist'
    else:
        wr, el, sh = 'right_wrist', 'right_elbow', 'right_shoulder'
        owr = 'left_wrist'

    wx, wy = kp_xy(row, wr)
    ex, ey = kp_xy(row, el)
    sx, sy = kp_xy(row, sh)

    arm_ext = math.hypot(wx - sx, wy - sy) / s
    ea = angle3(sx, sy, ex, ey, wx, wy)
    el_y = (ey - sy) / s
    el_lat = abs(ex - sx) / s
    wr_y = (wy - sy) / s
    ew = math.hypot(ex - wx, ey - wy) / s

    lo = max(0, idx - WINDOW)
    sub = rows[lo:idx + 1]
    wxs = np.array([float(r[f'{wr}_x']) for r in sub])
    wys = np.array([float(r[f'{wr}_y']) for r in sub])
    owxs = np.array([float(r[f'{owr}_x']) for r in sub])
    owys = np.array([float(r[f'{owr}_y']) for r in sub])

    if len(wxs) >= 2:
        dwx = np.diff(wxs); dwy = np.diff(wys)
        wr_dx_abs = (np.abs(dwx) / s).sum()
        wr_dy_min = float(((wys - wy) / s).min())  # 음수: 위로 갔다
        wr_dy_rise = float((wys[0] - wys.min()) / s)
        wr_peak_v = float((np.hypot(dwx, dwy) / s).max())
        odwx = np.diff(owxs); odwy = np.diff(owys)
        opp_wr_dx_abs = float((np.hypot(odwx, odwy) / s).sum())
    else:
        wr_dx_abs = wr_dy_min = wr_dy_rise = wr_peak_v = opp_wr_dx_abs = 0.0

    arm_dy_rise = (sy - wys.min()) / s if len(wys) else 0.0

    return [arm_ext, ea, el_y, el_lat, wr_y, ew, arm_dy_rise,
            wr_dx_abs, wr_dy_min, wr_dy_rise, wr_peak_v, opp_wr_dx_abs]


def assign_side(rows, idx):
    """라벨 프레임 기준 직전 6프레임에서 더 많이 움직인 손목 쪽."""
    lo = max(0, idx - 6)
    sub = rows[lo:idx + 1]
    s = sw_row(rows[idx])
    lwx = np.array([float(r['left_wrist_x']) for r in sub])
    lwy = np.array([float(r['left_wrist_y']) for r in sub])
    rwx = np.array([float(r['right_wrist_x']) for r in sub])
    rwy = np.array([float(r['right_wrist_y']) for r in sub])
    if len(sub) < 2:
        return 'L'
    l_disp = float(np.hypot(np.diff(lwx), np.diff(lwy)).sum()) / s
    r_disp = float(np.hypot(np.diff(rwx), np.diff(rwy)).sum()) / s
    return 'L' if l_disp >= r_disp else 'R'


def main():
    X, y = [], []
    per_video = Counter()
    side_counts = Counter()

    for full_csv, label_csv in DATASETS:
        fp = os.path.join(BASE_DIR, full_csv)
        lp = os.path.join(BASE_DIR, label_csv)
        if not (os.path.exists(fp) and os.path.exists(lp)):
            print(f'skip {full_csv} / {label_csv} (missing)')
            continue

        rows = read_csv_dict_list(fp)
        frame_idx = {int(r['frame_number']): i for i, r in enumerate(rows)}
        labels = read_csv_dict_list(lp)

        for lr in labels:
            f = int(lr['frame_number'])
            pt = lr['punch_type'].strip().lower()
            if f not in frame_idx:
                continue
            i = frame_idx[f]
            side = assign_side(rows, i)
            feat = features(rows, i, side)
            X.append(feat)
            y.append(pt)
            per_video[full_csv] += 1
            side_counts[(pt, side)] += 1

    X = np.array(X, dtype=float)
    y = np.array(y)

    print(f'총 샘플: {len(X)}')
    print(f'클래스 분포: {Counter(y)}')
    print(f'영상별 샘플: {dict(per_video)}')
    print(f'(punch,side) 분포: {dict(side_counts)}')

    if len(X) == 0:
        raise SystemExit('학습 데이터가 없음.')

    # 클래스 균형 가중치 (소수 클래스 부스팅)
    class_counts = Counter(y)
    n_total = len(y)
    n_classes = len(class_counts)
    weights = np.array([n_total / (n_classes * class_counts[yi]) for yi in y])

    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.06, random_state=0
    )
    clf.fit(X, y, sample_weight=weights)
    train_acc = clf.score(X, y)
    print(f'Train accuracy: {train_acc:.3f}')

    feat_imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    print('Feature importance:')
    for n, v in feat_imp:
        print(f'  {n:14s} {v:.3f}')

    out = {
        'model': clf,
        'features': FEATURE_NAMES,
        'classes': list(clf.classes_),
        'window': WINDOW,
    }
    out_path = os.path.join(BASE_DIR, 'lim_punch_model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'\n저장: {out_path}')


if __name__ == '__main__':
    main()
