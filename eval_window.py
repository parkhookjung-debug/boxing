"""
eval_window.py — 윈도우 분류기 이벤트 단위 LOVO 평가

각 영상을 테스트로 빼고 나머지 7개로 학습 → 테스트 영상 전 프레임을
분류 → PunchEventTracker 로 이벤트 묶음 → 수동 라벨과 ±MATCH_WIN 매칭.
앱과 동일한 트래커를 쓰므로 이 수치 = 앱 실제 성능.

출력 (eval_csv/):
  window_metrics.csv   per-class precision/recall/F1 + FP/FN
  window_summary.txt   요약 + failure cases
  window_events.csv    이벤트 상세

사용:
  python eval_window.py
  python eval_window.py --enter 0.5 --min-run 3   # 트래커 파라미터 탐색
"""

import argparse
import csv
import io
import os
import sys
from collections import Counter

if __name__ == '__main__':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      errors='replace')

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (accuracy_score, f1_score,
                                 classification_report)
    _SKL = True
except ImportError:
    raise SystemExit('pip install scikit-learn')

import lim_punch_features as F
from LIM_train_window import build_video, make_model, FILES, LABELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MATCH_WIN = 12
PUNCH_LABELS = ['jab', 'cross', 'hook', 'uppercut', 'NONE']


def predict_events(clf, fpath, tracker_kw):
    """학습된 clf 로 영상 전 프레임 분류 → 이벤트 목록."""
    frame_nums, kps = F.load_kp_csv(fpath)
    n = len(kps)
    if n < F.WINDOW:
        return []
    # 전 프레임 윈도우 피처 → 배치 예측
    feats = [F.window_feat(kps, i) for i in range(n)]
    probs = clf.predict_proba(np.array(feats, dtype=float))
    # 클래스 순서를 CLASSES 기준으로 정렬
    cls_order = list(clf.classes_)
    idx_map = [cls_order.index(c) for c in F.CLASSES]
    probs = probs[:, idx_map]

    tracker = F.PunchEventTracker(**tracker_kw)
    events = []
    for i in range(n):
        ev = tracker.push(frame_nums[i], probs[i])
        if ev:
            events.append((ev['frame'], ev['punch_type']))
    ev = tracker.flush()
    if ev:
        events.append((ev['frame'], ev['punch_type']))
    return events


def match_events(gt, pred):
    """GT ↔ Pred ±MATCH_WIN 매칭."""
    y_true, y_pred = [], []
    used = set()
    pairs, fn_list = [], []
    for gf, gl in gt:
        best_j, best_d = None, MATCH_WIN + 1
        for j, (pf, pl) in enumerate(pred):
            if j in used:
                continue
            d = abs(gf - pf)
            if d <= MATCH_WIN and d < best_d:
                best_d, best_j = d, j
        if best_j is not None:
            used.add(best_j)
            y_true.append(gl)
            y_pred.append(pred[best_j][1])
            pairs.append((gf, gl, pred[best_j][0], pred[best_j][1]))
        else:
            y_true.append(gl)
            y_pred.append('NONE')
            fn_list.append((gf, gl))
    fp_list = [pred[j] for j in range(len(pred)) if j not in used]
    return y_true, y_pred, pairs, fp_list, fn_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enter', type=float, default=F.DEFAULT_ENTER_THR)
    ap.add_argument('--min-run', type=int, default=F.DEFAULT_MIN_RUN)
    ap.add_argument('--refractory', type=int, default=F.DEFAULT_REFRACTORY)
    args = ap.parse_args()
    tracker_kw = dict(enter_thr=args.enter, min_run=args.min_run,
                      refractory=args.refractory)
    print(f'트래커: enter={args.enter} min_run={args.min_run} '
          f'refractory={args.refractory}')

    # 영상별 데이터 빌드
    data = {}
    for fn, ln in zip(FILES, LABELS):
        fp = os.path.join(BASE_DIR, fn)
        lp = os.path.join(BASE_DIR, ln)
        if not (os.path.exists(fp) and os.path.exists(lp)):
            continue
        data[fn] = build_video(fp, lp)

    all_true, all_pred = [], []
    all_fp = all_fn = 0
    per_file = []
    ev_rows = []

    for test_fn in data:
        Xtr = np.vstack([data[f][0] for f in data if f != test_fn])
        ytr = np.concatenate([data[f][1] for f in data if f != test_fn])
        clf = make_model()
        clf.fit(Xtr, ytr)

        fp = os.path.join(BASE_DIR, test_fn)
        lp = os.path.join(BASE_DIR, test_fn.replace('_full_data', '')
                          .replace('.csv', '_labels.csv'))
        # LIM_full_data1.csv → LIM1_labels.csv
        idx = test_fn.replace('LIM_full_data', '').replace('.csv', '')
        lp = os.path.join(BASE_DIR, f'LIM{idx}_labels.csv')

        gt = F.load_label_csv(lp)
        pred = predict_events(clf, fp, tracker_kw)
        y_true, y_pred, pairs, fp_list, fn_list = match_events(gt, pred)

        all_true += y_true
        all_pred += y_pred
        all_fp += len(fp_list)
        all_fn += len(fn_list)
        per_file.append((test_fn, len(gt), len(pred), len(pairs),
                         len(fn_list), len(fp_list)))
        for gf, gl, pf, pl in pairs:
            ev_rows.append({'file': test_fn, 'status': 'matched',
                            'gt_frame': gf, 'gt_label': gl,
                            'pred_frame': pf, 'pred_label': pl})
        for pf, pl in fp_list:
            ev_rows.append({'file': test_fn, 'status': 'fp', 'gt_frame': '',
                            'gt_label': '', 'pred_frame': pf, 'pred_label': pl})
        for gf, gl in fn_list:
            ev_rows.append({'file': test_fn, 'status': 'fn', 'gt_frame': gf,
                            'gt_label': gl, 'pred_frame': '',
                            'pred_label': 'NONE'})
        print(f'{test_fn:22s} GT={len(gt):3d}  Pred={len(pred):3d}  '
              f'match={len(pairs):3d}  FN={len(fn_list):2d}  FP={len(fp_list):3d}')

    # ── 집계 ────────────────────────────────────────────────
    acc = accuracy_score(all_true, all_pred)
    labels = [l for l in PUNCH_LABELS if l in set(all_true) | set(all_pred)]
    punch_only = [l for l in F.PUNCH_CLASSES
                  if l in set(all_true) | set(all_pred)]
    macro_f1 = f1_score(all_true, all_pred, labels=punch_only,
                        average='macro', zero_division=0)
    weight_f1 = f1_score(all_true, all_pred, labels=punch_only,
                         average='weighted', zero_division=0)
    # 매칭 정확도 (FN 제외 — 감지된 것만의 분류 정확도)
    matched_t = [t for t, p in zip(all_true, all_pred) if p != 'NONE']
    matched_p = [p for p in all_pred if p != 'NONE']
    cls_acc = accuracy_score(matched_t, matched_p) if matched_t else 0.0

    print('\n' + '=' * 60)
    print(f'GT={len(all_true)}  FP={all_fp}  FN={all_fn}')
    print(f'전체 정확도(Acc, FN포함)={acc:.4f}   '
          f'분류 정확도(매칭분만)={cls_acc:.4f}')
    print(f'펀치 Macro-F1={macro_f1:.4f}   Weighted-F1={weight_f1:.4f}')
    print()
    print(classification_report(all_true, all_pred, labels=labels,
                                zero_division=0))

    # ── 저장 ────────────────────────────────────────────────
    out_dir = os.path.join(BASE_DIR, 'eval_csv')
    os.makedirs(out_dir, exist_ok=True)

    rep = classification_report(all_true, all_pred, labels=labels,
                                output_dict=True, zero_division=0)
    with open(os.path.join(out_dir, 'window_metrics.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['class', 'precision', 'recall', 'f1', 'support'])
        for lbl in labels:
            m = rep.get(lbl, {})
            w.writerow([lbl, round(m.get('precision', 0), 4),
                        round(m.get('recall', 0), 4),
                        round(m.get('f1-score', 0), 4),
                        int(m.get('support', 0))])
        for avg in ('macro avg', 'weighted avg'):
            m = rep.get(avg, {})
            w.writerow([avg.replace(' ', '_').upper(),
                        round(m.get('precision', 0), 4),
                        round(m.get('recall', 0), 4),
                        round(m.get('f1-score', 0), 4),
                        int(m.get('support', 0))])
        w.writerow(['ACCURACY', '', '', '', round(acc, 4)])
        w.writerow(['MACRO_F1_PUNCH', '', '', '', round(macro_f1, 4)])
        w.writerow(['FALSE_NEG', '', '', '', all_fn])
        w.writerow(['FALSE_POS', '', '', '', all_fp])

    with open(os.path.join(out_dir, 'window_events.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'status', 'gt_frame',
                                          'gt_label', 'pred_frame',
                                          'pred_label'])
        w.writeheader()
        w.writerows(ev_rows)

    lines = [
        '=== 윈도우 분류기 — 이벤트 단위 LOVO 평가 ===',
        f'트래커: enter={args.enter} min_run={args.min_run} '
        f'refractory={args.refractory}  match_win=±{MATCH_WIN}',
        f'GT 이벤트: {len(all_true)}   FP: {all_fp}   FN: {all_fn}',
        f'전체 정확도: {acc:.4f}   분류 정확도(매칭분): {cls_acc:.4f}',
        f'펀치 Macro-F1: {macro_f1:.4f}   Weighted-F1: {weight_f1:.4f}',
        '',
        classification_report(all_true, all_pred, labels=labels,
                              zero_division=0),
        '── 파일별 ──',
    ]
    for it in per_file:
        lines.append(f'  {it[0]:22s} GT={it[1]:3d} Pred={it[2]:3d} '
                     f'match={it[3]:3d} FN={it[4]:2d} FP={it[5]:3d}')
    lines += ['', '── 오분류 (매칭됐지만 틀림) ──']
    fails = Counter((t, p) for t, p in zip(all_true, all_pred)
                    if t != p and p != 'NONE')
    for (gt, pred), c in fails.most_common():
        lines.append(f'  GT={gt:10s} → PRED={pred:10s}  ×{c}')
    lines += ['', '── 미감지 (FN) ──']
    for gl, c in Counter(t for t, p in zip(all_true, all_pred)
                         if p == 'NONE').most_common():
        lines.append(f'  GT={gl:10s} → 미감지  ×{c}')

    with open(os.path.join(out_dir, 'window_summary.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('저장: eval_csv/window_metrics.csv, window_summary.txt, '
          'window_events.csv')


if __name__ == '__main__':
    main()
