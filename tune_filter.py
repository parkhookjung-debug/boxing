"""
tune_filter.py — 1€ 필터 파라미터 스윕

학습 영상은 이미 깨끗해서(인식률 100%, vis 0.98) 평활이 강하면 펀치
동작까지 뭉개 손해만 본다. 이벤트 단위 LOVO Macro-F1 이 "필터 없음"
대비 떨어지지 않는 선에서 가장 강한 필터(낮은 min_cutoff)를 찾는다.

  · min_cutoff 낮을수록 정지 시 평활 강함 (떨림 제거↑)
  · beta 높을수록 빠른 동작(펀치)을 그대로 통과 (지연↓)
  → 목표: 떨림은 잡되 펀치는 살리는 (낮은 min_cutoff + 높은 beta)

실행: python tune_filter.py
"""

import io
import os
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import lim_punch_features as F
from LIM_train_window import build_video, make_model, FILES, LABELS
from eval_window import predict_events, match_events

BASE = os.path.dirname(os.path.abspath(__file__))
TRACKER_KW = dict(enter_thr=F.DEFAULT_ENTER_THR, min_run=F.DEFAULT_MIN_RUN,
                  refractory=F.DEFAULT_REFRACTORY)

# (이름, min_cutoff, beta) — 'off' 은 사실상 필터 없음(투명) 기준선
GRID = [
    ('off', 1000.0,  0.0),
    ('A',      1.0, 10.0),
    ('B',      1.0, 30.0),
    ('C',      1.0, 80.0),
    ('D',      2.0, 10.0),
    ('E',      2.0, 30.0),
    ('F',      4.0, 10.0),
]


def run_lovo():
    """현재 F.SMOOTH_* 전역값으로 이벤트 단위 LOVO 평가."""
    data = {}
    for fn, ln in zip(FILES, LABELS):
        fp, lp = os.path.join(BASE, fn), os.path.join(BASE, ln)
        if os.path.exists(fp) and os.path.exists(lp):
            data[fn] = build_video(fp, lp)

    all_true, all_pred, all_fn = [], [], 0
    for test_fn in data:
        Xtr = np.vstack([data[f][0] for f in data if f != test_fn])
        ytr = np.concatenate([data[f][1] for f in data if f != test_fn])
        clf = make_model()
        clf.fit(Xtr, ytr)

        idx = test_fn.replace('LIM_full_data', '').replace('.csv', '')
        gt = F.load_label_csv(os.path.join(BASE, f'LIM{idx}_labels.csv'))
        pred = predict_events(clf, os.path.join(BASE, test_fn), TRACKER_KW)
        yt, yp, _, _, fnl = match_events(gt, pred)
        all_true += yt
        all_pred += yp
        all_fn += len(fnl)

    mf1 = f1_score(all_true, all_pred, labels=F.PUNCH_CLASSES,
                   average='macro', zero_division=0)
    mt = [t for t, p in zip(all_true, all_pred) if p != 'NONE']
    mp = [p for p in all_pred if p != 'NONE']
    cls = accuracy_score(mt, mp) if mt else 0.0
    return mf1, cls, all_fn


def main():
    print(f"{'name':5s} {'min_cut':>8s} {'beta':>6s} "
          f"{'MacroF1':>8s} {'clsAcc':>7s} {'FN':>4s}")
    print('-' * 44)
    results = []
    for name, mc, beta in GRID:
        F.SMOOTH_MIN_CUTOFF = mc
        F.SMOOTH_BETA = beta
        mf1, cls, fn = run_lovo()
        results.append((name, mc, beta, mf1, cls, fn))
        print(f"{name:5s} {mc:8.1f} {beta:6.1f} "
              f"{mf1:8.4f} {cls:7.4f} {fn:4d}", flush=True)

    base = next(r for r in results if r[0] == 'off')[3]
    print('-' * 44)
    print(f"기준선(off) Macro-F1 = {base:.4f}")
    ok = [r for r in results if r[0] != 'off' and r[3] >= base - 0.005]
    if ok:
        best = min(ok, key=lambda r: (r[1], -r[4]))  # 가장 강한 필터
        print(f"권장: min_cutoff={best[1]} beta={best[2]}  "
              f"(Macro-F1={best[3]:.4f}, 기준선 거의 유지하며 평활 최대)")
    else:
        print("기준선 유지 조합 없음 — 필터를 더 약하게 (min_cutoff↑/beta↑).")


if __name__ == '__main__':
    main()
