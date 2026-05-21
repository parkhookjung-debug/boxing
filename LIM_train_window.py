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

FILES = [f'LIM_full_data{i}.csv' for i in range(1, 9)]
LABELS = [f'LIM{i}_labels.csv' for i in range(1, 9)]

POS_HALF = 2      # 라벨 ±POS_HALF 프레임을 양성으로
NEG_GAP = 10      # 라벨에서 NEG_GAP 초과 떨어진 프레임만 음성 후보
NEG_PER_VIDEO = 110
SEED = 0


def build_video(fpath, lpath):
    """한 영상 → (X, y, meta). meta 는 (kind, label) 표시용."""
    frame_nums, kps = F.load_kp_csv(fpath)
    labels = F.load_label_csv(lpath)
    pos = {fn: i for i, fn in enumerate(frame_nums)}
    n = len(kps)

    label_idx = {}  # frame index → punch type
    for lf, lt in labels:
        if lf in pos:
            label_idx[pos[lf]] = lt

    X, y = [], []
    # 양성
    for i, lt in label_idx.items():
        for off in range(-POS_HALF, POS_HALF + 1):
            j = i + off
            if 0 <= j < n:
                X.append(F.window_feat(kps, j))
                y.append(lt)
    # 음성 — 라벨에서 먼 프레임
    far = [i for i in range(F.HALF, n - F.HALF)
           if all(abs(i - li) > NEG_GAP for li in label_idx)]
    rng = np.random.RandomState(SEED)
    if far:
        take = rng.choice(far, min(NEG_PER_VIDEO, len(far)), replace=False)
        for i in take:
            X.append(F.window_feat(kps, int(i)))
            y.append('none')
    return np.array(X, dtype=float), np.array(y)


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
        X, y = build_video(fp, lp)
        data[fn] = (X, y)
        print(f'{fn:22s} 표본 {len(X):4d}  {dict(Counter(y))}')

    if not data:
        raise SystemExit('데이터 없음')

    # ── LOVO 교차검증 (정직한 일반화 성능) ─────────────────
    print('\n=== Leave-one-video-out 교차검증 (프레임 단위) ===')
    all_true, all_pred = [], []
    for test_fn in data:
        Xtr = np.vstack([data[f][0] for f in data if f != test_fn])
        ytr = np.concatenate([data[f][1] for f in data if f != test_fn])
        Xte, yte = data[test_fn]
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

    # ── 최종 모델: 전체 데이터 학습 ────────────────────────
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
