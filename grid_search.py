"""
grid_search.py — 펀치 감지 임계값 자동 최적화
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용법:
  python grid_search.py                    # LIM1~5 전체, pseudo-GT
  python grid_search.py --gt_labels l1.csv l2.csv ...  # 수동 라벨

출력:
  grid_results.csv   전체 조합 결과
  best_params.txt    최적 파라미터
"""

import sys, os, io, csv, math, itertools, time
from collections import deque, Counter
import numpy as np

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_from_csv import (
    load_full_data, get_gt_events, match_events, load_manual_labels,
    g, sw, elbow_angle_row, arm_ext_row, ew_dist_row,
    BASE_DIR, DEFAULT_FILES, MATCH_WIN,
)
# DEFAULT_FILES = LIM_full_data1~4.csv (LIM1/3 정면, LIM2/4 측면)
from sklearn.metrics import f1_score, accuracy_score

# ── 탐색 범위 ─────────────────────────────────────────────────────
GRID = {
    'vel_start':  [0.015, 0.018, 0.020, 0.022, 0.025],
    'dom_ratio':  [1.0, 1.05, 1.10, 1.15],
    'punch_cd':   [6, 8, 10, 12],
    'ew_hook':    [0.26, 0.28, 0.30, 0.32, 0.35],
    'up_vel':     [0.09, 0.11, 0.12, 0.14],
}
BUF_SZ = 12

# ── 파라미터화된 pred 생성 ─────────────────────────────────────────
def _classify_p(ea_buf, vx_buf, vy_buf, ew_buf, up_vel, ew_hook, side, hook_hint=False):
    min_ea    = min(ea_buf,  default=180.0)
    max_vx    = max(vx_buf,  default=0.0)
    max_vy_up = max((-v for v in vy_buf), default=0.0)
    min_ew    = min(ew_buf, default=1.0)

    if max_vy_up > up_vel and min_ea < 130.0 and max_vy_up > max_vx * 0.7:
        return 'uppercut'
    if hook_hint or min_ew < ew_hook:
        return 'hook'
    return 'jab' if side == 'lead' else 'cross'


def get_pred_p(rows, orthodox, p):
    vel  = p['vel_start']; dom = p['dom_ratio']
    cd   = p['punch_cd'];  ew  = p['ew_hook'];  up = p['up_vel']

    if orthodox:
        JWR, JSH, JEL       = 'left_wrist',  'left_shoulder',  'left_elbow'
        CWR, CSH, CEL       = 'right_wrist', 'right_shoulder', 'right_elbow'
    else:
        JWR, JSH, JEL       = 'right_wrist', 'right_shoulder', 'right_elbow'
        CWR, CSH, CEL       = 'left_wrist',  'left_shoulder',  'left_elbow'

    rv_l=deque(maxlen=BUF_SZ); rv_r=deque(maxlen=BUF_SZ)
    ea_l=deque(maxlen=BUF_SZ); ea_r=deque(maxlen=BUF_SZ)
    vx_l=deque(maxlen=BUF_SZ); vx_r=deque(maxlen=BUF_SZ)
    vy_l=deque(maxlen=BUF_SZ); vy_r=deque(maxlen=BUF_SZ)
    ew_l=deque(maxlen=BUF_SZ); ew_r=deque(maxlen=BUF_SZ)

    prv_l=prv_r=0.0; pw_l=pw_r=None
    events=[]; last_fired=-cd

    for idx, row in enumerate(rows):
        sv = sw(row)
        jx,jy   = g(row,JWR,'x'), g(row,JWR,'y')
        cx_,cy_ = g(row,CWR,'x'), g(row,CWR,'y')

        raw_l = math.hypot(jx-pw_l[0], jy-pw_l[1])   if pw_l else 0.0
        raw_r = math.hypot(cx_-pw_r[0],cy_-pw_r[1])  if pw_r else 0.0
        dvx_l = abs(jx-pw_l[0])/sv  if pw_l else 0.0
        dvy_l = (jy-pw_l[1])/sv     if pw_l else 0.0
        dvx_r = abs(cx_-pw_r[0])/sv if pw_r else 0.0
        dvy_r = (cy_-pw_r[1])/sv    if pw_r else 0.0

        rv_l.append(raw_l); rv_r.append(raw_r)
        ea_l.append(elbow_angle_row(row,JSH,JEL,JWR))
        ea_r.append(elbow_angle_row(row,CSH,CEL,CWR))
        vx_l.append(dvx_l); vx_r.append(dvx_r)
        vy_l.append(dvy_l); vy_r.append(dvy_r)
        ew_l.append(ew_dist_row(row,JEL,JWR,sv))
        ew_r.append(ew_dist_row(row,CEL,CWR,sv))

        pk_l=max(rv_l); pk_r=max(rv_r); cd_ok=(idx-last_fired>=cd)
        r_l=list(ew_l)[-4:]; r_r=list(ew_r)[-4:]
        hl = (sum(r_l)/len(r_l) if r_l else 1.0) < ew
        hr = (sum(r_r)/len(r_r) if r_r else 1.0) < ew
        vt_l = vel*0.5 if hl else vel; vt_r = vel*0.5 if hr else vel
        dm_l = 1.0 if hl else dom;     dm_r = 1.0 if hr else dom

        fired=False
        if prv_l>vt_l and raw_l<=vt_l and pk_l>pk_r*dm_l and cd_ok:
            events.append((idx, _classify_p(ea_l,vx_l,vy_l,ew_l,up,ew,'lead',hook_hint=hl)))
            last_fired=idx; fired=True
        if not fired and prv_r>vt_r and raw_r<=vt_r and pk_r>pk_l*dm_r and cd_ok:
            events.append((idx, _classify_p(ea_r,vx_r,vy_r,ew_r,up,ew,'rear',hook_hint=hr)))
            last_fired=idx; fired=True
        if fired:
            rv_l.clear(); rv_r.clear(); prv_l=prv_r=0.0
            pw_l=(jx,jy); pw_r=(cx_,cy_); continue
        prv_l=raw_l; prv_r=raw_r; pw_l=(jx,jy); pw_r=(cx_,cy_)
    return events


def evaluate(all_data, all_gt, orthodox, p):
    y_true_all, y_pred_all = [], []
    fp_all = 0
    for rows, gt_events in zip(all_data, all_gt):
        pred = get_pred_p(rows, orthodox, p)
        yt, yp, _, fp = match_events(gt_events, pred)
        y_true_all.extend(yt); y_pred_all.extend(yp)
        fp_all += fp
    # weighted F1 (jab+cross만, NONE 제외)
    labels = [l for l in ['jab','cross','hook','uppercut']
              if l in set(y_true_all)]
    if not labels:
        return 0.0, 0.0, 0
    wf1 = f1_score(y_true_all, y_pred_all, labels=labels,
                   average='weighted', zero_division=0)
    acc = accuracy_score(y_true_all, y_pred_all)
    return wf1, acc, fp_all


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--files',     nargs='+', default=DEFAULT_FILES)
    parser.add_argument('--stance',    default='orthodox')
    parser.add_argument('--gt_labels', nargs='+', default=None)
    parser.add_argument('--out',       default='grid_results.csv')
    parser.add_argument('--top',       default=20, type=int, help='상위 N개 출력')
    args = parser.parse_args()

    orthodox = (args.stance == 'orthodox')

    # 데이터 로드
    print("데이터 로드 중...")
    all_data, all_gt = [], []
    for i, fname in enumerate(args.files):
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            continue
        rows = load_full_data(fpath)
        all_data.append(rows)

        if args.gt_labels and i < len(args.gt_labels):
            lp = os.path.join(BASE_DIR, args.gt_labels[i])
            all_gt.append(load_manual_labels(lp))
            print(f"  {fname}: 수동 GT {len(all_gt[-1])}개")
        else:
            all_gt.append(get_gt_events(rows, orthodox=orthodox))
            print(f"  {fname}: pseudo-GT {len(all_gt[-1])}개")

    total_gt = sum(len(g) for g in all_gt)
    print(f"\n총 GT 이벤트: {total_gt}개  |  조합 수: "
          f"{math.prod(len(v) for v in GRID.values()):,}개\n")

    # 그리드 서치
    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
    results = []
    t0 = time.time()

    for ci, combo in enumerate(combos):
        p = dict(zip(keys, combo))
        wf1, acc, fp = evaluate(all_data, all_gt, orthodox, p)
        results.append({**p, 'wf1': round(wf1,4), 'acc': round(acc,4), 'fp': fp})

        if (ci+1) % 100 == 0 or ci+1 == total:
            elapsed = time.time() - t0
            eta     = elapsed / (ci+1) * (total - ci - 1)
            print(f"\r  {ci+1}/{total}  best_wf1={max(r['wf1'] for r in results):.4f}"
                  f"  ETA {eta:.0f}s   ", end='', flush=True)

    print(f"\n완료: {time.time()-t0:.1f}초\n")

    # 결과 정렬 (weighted F1 내림차순)
    results.sort(key=lambda r: (-r['wf1'], -r['acc']))

    # CSV 저장
    out_path = os.path.join(BASE_DIR, args.out)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"[OK] 전체 결과 → {out_path}\n")

    # 상위 N개 출력
    print(f"{'─'*70}")
    print(f"{'순위':>4}  {'vel':>5}  {'dom':>5}  {'cd':>3}  {'ew':>5}  "
          f"{'up':>5}  {'wF1':>6}  {'acc':>6}  {'FP':>4}")
    print(f"{'─'*70}")
    for rank, r in enumerate(results[:args.top], 1):
        print(f"{rank:4d}  {r['vel_start']:5.3f}  {r['dom_ratio']:5.2f}  "
              f"{r['punch_cd']:3d}  {r['ew_hook']:5.2f}  {r['up_vel']:5.2f}  "
              f"{r['wf1']:6.4f}  {r['acc']:6.4f}  {r['fp']:4d}")

    best = results[0]
    print(f"\n{'━'*70}")
    print(f"최적 파라미터 (weighted F1 = {best['wf1']:.4f}):")
    for k, v in best.items():
        if k not in ('wf1', 'acc', 'fp'):
            print(f"  {k:15s} = {v}")

    # best_params.txt 저장
    best_path = os.path.join(BASE_DIR, 'best_params.txt')
    with open(best_path, 'w', encoding='utf-8') as f:
        f.write(f"# 최적 파라미터  (weighted F1={best['wf1']:.4f}, acc={best['acc']:.4f})\n")
        for k, v in best.items():
            if k not in ('wf1', 'acc', 'fp'):
                f.write(f"{k} = {v}\n")
    print(f"[OK] 최적 파라미터 → {best_path}")


if __name__ == '__main__':
    main()
