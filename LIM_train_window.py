"""
LIM_train_window.py — 5-클래스 윈도우 펀치 분류기 학습

각 프레임을 ±HALF 윈도우로 보고 {none, jab, cross, hook, uppercut} 분류.
감지와 분류를 한 모델로 통합 — 별도 속도-피크 detector 불필요.

  · 양성 표본: 라벨 프레임 ±POS_HALF  (펀치 동작이 윈도우에 포함됨)
  · 음성 표본: 모든 라벨에서 NEG_GAP 이상 떨어진 프레임에서 샘플링
  · LOVO(leave-one-video-out) 교차검증으로 정직한 일반화 성능 리포트
  · 최종 모델은 전체 데이터로 학습 → lim_window_model.pkl

실행: python LIM_train_window.py
"""

import os
import sys
import io
import pickle
from collections import Counter

if __name__ == '__main__':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      errors='replace')

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report
except ImportError:
    raise SystemExit('pip install scikit-learn numpy')

import lim_punch_features as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 측면 영상만 사용 — 정면(LIM1·3·5)은 2D 분류기 입장에서 잽/크로스가
# 카메라축(Z) 모션이라 화면상 거의 안 움직여 어퍼/훅 쏠림을 유발한다.
# 측면 전용 코치로 한정 → 정면 데이터를 학습에서 제외 (CSV 자체는 보존).
SIDE_IDX = [2, 4, 6, 7, 8]
FILES = [f'LIM_full_data{i}.csv' for i in SIDE_IDX]
LABELS = [f'LIM{i}_labels.csv' for i in SIDE_IDX]

POS_HALF = 2      # 라벨 ±POS_HALF 프레임을 양성으로
NEG_GAP = 10      # 라벨에서 NEG_GAP 초과 떨어진 프레임만 음성 후보
NEG_PER_VIDEO = 110
SEED = 0


def _video_samples(kps, label_idx, neg_take, n):
    """한 kp 시퀀스 → 양성(라벨±POS_HALF)·음성(neg_take) 윈도우 표본."""
    X, y = [], []
    for i, lt in label_idx.items():
        for off in range(-POS_HALF, POS_HALF + 1):
            j = i + off
            if 0 <= j < n:
                X.append(F.window_feat(kps, j))
                y.append(lt)
    for i in neg_take:
        X.append(F.window_feat(kps, int(i)))
        y.append('none')
    return X, y


def build_video(fpath, lpath):
    """한 영상 → (X, y, n_orig).

    앞 n_orig 행은 원본, 뒤는 수평반전 증강 표본(손잡이 불변 — 잽은 어느
    손이든 잽). 같은 라벨·같은 음성 프레임을 반전해 분포를 맞춘다.
    LOVO 평가는 원본 행만 테스트하므로 호출부가 n_orig 로 잘라 쓴다."""
    frame_nums, kps = F.load_kp_csv(fpath)
    labels = F.load_label_csv(lpath)
    pos = {fn: i for i, fn in enumerate(frame_nums)}
    n = len(kps)

    label_idx = {}  # frame index → punch type
    for lf, lt in labels:
        if lf in pos:
            label_idx[pos[lf]] = lt

    # 음성 후보 — 라벨에서 먼 프레임 (원본·반전 표본이 동일 프레임 사용)
    far = [i for i in range(F.HALF, n - F.HALF)
           if all(abs(i - li) > NEG_GAP for li in label_idx)]
    rng = np.random.RandomState(SEED)
    neg_take = (rng.choice(far, min(NEG_PER_VIDEO, len(far)), replace=False)
                if far else [])

    Xo, yo = _video_samples(kps, label_idx, neg_take, n)
    Xf, yf = _video_samples(F.flip_kp_sequence(kps), label_idx, neg_take, n)
    X = np.array(Xo + Xf, dtype=float)
    y = np.array(yo + yf)
    return X, y, len(Xo)


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.07,
        l2_regularization=1.0, random_state=SEED)


def main():
    data = {}
    for fn, ln in zip(FILES, LABELS):
        fp = os.path.join(BASE_DIR, fn)
        lp = os.path.join(BASE_DIR, ln)
        if not (os.path.exists(fp) and os.path.exists(lp)):
            print(f'[skip] {fn}')
            continue
        X, y, n_orig = build_video(fp, lp)
        data[fn] = (X, y, n_orig)
        print(f'{fn:22s} 표본 {len(X):4d} (원본 {n_orig} + 반전 '
              f'{len(X) - n_orig})  {dict(Counter(y))}')

    if not data:
        raise SystemExit('데이터 없음')

    # ── LOVO 교차검증 (정직한 일반화 성능) ─────────────────
    print('\n=== Leave-one-video-out 교차검증 (프레임 단위) ===')
    print('  학습 폴드: 원본+반전증강 / 테스트 폴드: 원본만 (정직한 추정)')
    all_true, all_pred = [], []
    for test_fn in data:
        Xtr = np.vstack([data[f][0] for f in data if f != test_fn])
        ytr = np.concatenate([data[f][1] for f in data if f != test_fn])
        Xte, yte, n_orig = data[test_fn]
        Xte, yte = Xte[:n_orig], yte[:n_orig]   # 테스트는 원본만
        clf = make_model()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        all_true.extend(yte)
        all_pred.extend(pred)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    mask = all_true != 'none'
    acc = accuracy_score(all_true[mask], all_pred[mask])
    mf1 = f1_score(all_true[mask], all_pred[mask],
                   labels=F.PUNCH_CLASSES, average='macro', zero_division=0)
    print(f'펀치 프레임 분류: Acc={acc:.4f}  Macro-F1={mf1:.4f}')
    print(classification_report(all_true[mask], all_pred[mask],
                                labels=F.PUNCH_CLASSES, zero_division=0))

    bt = np.where(all_true == 'none', 'none', 'punch')
    bp = np.where(all_pred == 'none', 'none', 'punch')
    print(f'펀치/none 이진 감지: Acc={accuracy_score(bt, bp):.4f}')
    print(classification_report(bt, bp, zero_division=0))

    # ── 최종 모델: 전체 데이터(원본+반전증강) 학습 ──────────
    Xall = np.vstack([data[f][0] for f in data])
    yall = np.concatenate([data[f][1] for f in data])
    final = make_model()
    final.fit(Xall, yall)
    print(f'최종 모델 학습 완료 — 전체 {len(Xall)} 표본')

    out = {
        'model': final,
        'classes': list(final.classes_),
        'half': F.HALF,
        'window': F.WINDOW,
        'feat_dim': F.WINDOW_FEAT_DIM,
        'meta': {
            'lovo_punch_acc': round(float(acc), 4),
            'lovo_macro_f1': round(float(mf1), 4),
            'samples': int(len(Xall)),
            'class_counts': {k: int(v) for k, v in Counter(yall).items()},
        },
    }
    out_path = os.path.join(BASE_DIR, 'lim_window_model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'저장: {out_path}')


if __name__ == '__main__':
    main()
