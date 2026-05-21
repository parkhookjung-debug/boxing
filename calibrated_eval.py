
"""
calibrated_eval.py - Manual-label calibrated punch classifier evaluation.

This is a personalization/calibration experiment: it learns punch labels from the
current manual labels and re-labels detected trigger events using trigger-level
features. It is not an unbiased holdout benchmark; use it to estimate the ceiling
for a user/session-specific calibrated coach.
"""

import csv
import os
import sys
from collections import Counter

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_from_csv as ev

BASE_DIR = ev.BASE_DIR
FILES = ['LIM_full_data1.csv', 'LIM_full_data2.csv', 'LIM_full_data3.csv', 'LIM_full_data4.csv']
LABEL_FILES = ['LIM1_labels.csv', 'LIM2_labels.csv', 'LIM3_labels.csv', 'LIM4_labels.csv']
PUNCH_LABELS = ['jab', 'cross', 'hook', 'uppercut']
FEATURES = [
    'peak', 'opp_peak', 'dom_ratio', 'min_ea', 'max_vx', 'max_vy_up',
    'max_ext', 'min_ext', 'min_ew', 'avg_ew4', 'max_lat', 'max_eratio',
    'view_side', 'side_rear',
]
TRIGGER_CSV = os.path.join(BASE_DIR, 'eval_csv', 'trigger_analysis.csv')
OUT_DIR = os.path.join(BASE_DIR, 'eval_csv')
MATCH_WIN = 12


def load_trigger_rows(path=TRIGGER_CSV):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            row['pred_frame'] = int(row['pred_frame'])
            for key in FEATURES:
                if key == 'view_side':
                    row[key] = 1.0 if row['view'] == 'side' else 0.0
                elif key == 'side_rear':
                    row[key] = 1.0 if row['side'] == 'rear' else 0.0
                else:
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def match_rows_to_gt(gt_events, pred_rows, win=MATCH_WIN):
    used = set()
    pairs = []
    missed = []
    for gi, (gf, gl) in enumerate(gt_events):
        best_j = None
        best_d = win + 1
        for pj, row in enumerate(pred_rows):
            if pj in used:
                continue
            d = abs(gf - row['pred_frame'])
            if d <= win and d < best_d:
                best_j = pj
                best_d = d
        if best_j is None:
            missed.append(gi)
        else:
            used.add(best_j)
            pairs.append((gi, best_j, best_d))
    fp = [pj for pj in range(len(pred_rows)) if pj not in used]
    return pairs, missed, fp


def build_training_set(rows, gt_by_file):
    samples = []
    for fname in FILES:
        file_rows = [r for r in rows if r['file'] == fname]
        pairs, _, _ = match_rows_to_gt(gt_by_file[fname], file_rows)
        for gi, pj, dist in pairs:
            sample = dict(file_rows[pj])
            sample['target'] = gt_by_file[fname][gi][1]
            sample['gt_frame'] = gt_by_file[fname][gi][0]
            sample['gt_dist'] = dist
            samples.append(sample)
    return samples


def make_model(kind='gb'):
    if kind == 'knn':
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance'))
    return GradientBoostingClassifier(random_state=7, max_depth=2, n_estimators=80, learning_rate=0.05)


def evaluate(rows, gt_by_file, model):
    y_true, y_pred = [], []
    total_fp = total_fn = 0
    event_rows = []
    per_file = []

    for fname in FILES:
        file_rows = [dict(r) for r in rows if r['file'] == fname]
        for row in file_rows:
            row['calibrated_label'] = model.predict([[row[k] for k in FEATURES]])[0]

        pred_events = [(r['pred_frame'], r['calibrated_label']) for r in file_rows]
        pairs, missed, fp = match_rows_to_gt(gt_by_file[fname], file_rows)
        used_pred = set()
        matched = 0

        for gi, pj, dist in pairs:
            used_pred.add(pj)
            gt_frame, gt_label = gt_by_file[fname][gi]
            pred_label = file_rows[pj]['calibrated_label']
            y_true.append(gt_label)
            y_pred.append(pred_label)
            matched += 1
            event_rows.append({
                'file': fname,
                'status': 'matched',
                'gt_frame': gt_frame,
                'gt_label': gt_label,
                'pred_frame': file_rows[pj]['pred_frame'],
                'pred_label': pred_label,
                'dist': dist,
            })

        for gi in missed:
            gt_frame, gt_label = gt_by_file[fname][gi]
            y_true.append(gt_label)
            y_pred.append('NONE')
            event_rows.append({
                'file': fname,
                'status': 'fn',
                'gt_frame': gt_frame,
                'gt_label': gt_label,
                'pred_frame': '',
                'pred_label': 'NONE',
                'dist': '',
            })

        for pj in fp:
            event_rows.append({
                'file': fname,
                'status': 'fp',
                'gt_frame': '',
                'gt_label': '',
                'pred_frame': file_rows[pj]['pred_frame'],
                'pred_label': file_rows[pj]['calibrated_label'],
                'dist': '',
            })

        total_fn += len(missed)
        total_fp += len(fp)
        per_file.append((fname, len(gt_by_file[fname]), len(file_rows), matched, len(missed), len(fp)))

    labels = [l for l in PUNCH_LABELS if l in set(y_true) | set(y_pred)]
    acc = accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    return acc, wf1, total_fp, total_fn, y_true, y_pred, event_rows, per_file


def save_outputs(acc, wf1, fp, fn, y_true, y_pred, event_rows, per_file, model_kind):
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUT_DIR, 'calibrated_metrics.csv')
    events_path = os.path.join(OUT_DIR, 'calibrated_events.csv')
    summary_path = os.path.join(OUT_DIR, 'calibrated_summary.txt')

    report = classification_report(
        y_true, y_pred,
        labels=PUNCH_LABELS + (['NONE'] if 'NONE' in y_pred else []),
        output_dict=True,
        zero_division=0,
    )
    with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['class', 'precision', 'recall', 'f1', 'support']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label in PUNCH_LABELS + (['NONE'] if 'NONE' in y_pred else []):
            m = report.get(label, {})
            w.writerow({
                'class': label,
                'precision': round(m.get('precision', 0.0), 4),
                'recall': round(m.get('recall', 0.0), 4),
                'f1': round(m.get('f1-score', 0.0), 4),
                'support': int(m.get('support', 0)),
            })
        for avg in ['macro avg', 'weighted avg']:
            m = report.get(avg, {})
            w.writerow({
                'class': avg.replace(' ', '_').upper(),
                'precision': round(m.get('precision', 0.0), 4),
                'recall': round(m.get('recall', 0.0), 4),
                'f1': round(m.get('f1-score', 0.0), 4),
                'support': int(m.get('support', 0)),
            })
        w.writerow({'class': 'ACCURACY', 'precision': '', 'recall': '', 'f1': '', 'support': round(acc, 4)})
        w.writerow({'class': 'FALSE_NEG(missed)', 'precision': '', 'recall': '', 'f1': '', 'support': fn})
        w.writerow({'class': 'FALSE_POS(extra)', 'precision': '', 'recall': '', 'f1': '', 'support': fp})

    with open(events_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['file', 'status', 'gt_frame', 'gt_label', 'pred_frame', 'pred_label', 'dist']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(event_rows)

    fails = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    lines = [
        'Calibrated punch classifier evaluation',
        'NOTE: This is a personalization/calibration score, not a holdout benchmark.',
        f'Model: {model_kind}',
        f'Accuracy: {acc:.4f}',
        f'Weighted F1: {wf1:.4f}',
        f'False Neg: {fn}',
        f'False Pos: {fp}',
        '',
        'Per file:',
    ]
    for item in per_file:
        lines.append(f'  {item}')
    lines += ['', classification_report(y_true, y_pred, labels=PUNCH_LABELS + (['NONE'] if 'NONE' in y_pred else []), zero_division=0), '', 'Failure cases:']
    for (gt, pred), count in fails.most_common():
        lines.append(f'  GT={gt:10s} -> PRED={pred:10s} x{count}')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    return metrics_path, events_path, summary_path


def main():
    model_kind = sys.argv[1] if len(sys.argv) > 1 else 'gb'
    rows = load_trigger_rows()
    gt_by_file = {f: ev.load_manual_labels(os.path.join(BASE_DIR, l)) for f, l in zip(FILES, LABEL_FILES)}
    samples = build_training_set(rows, gt_by_file)

    model = make_model(model_kind)
    X = [[s[k] for k in FEATURES] for s in samples]
    y = [s['target'] for s in samples]
    model.fit(X, y)

    acc, wf1, fp, fn, y_true, y_pred, event_rows, per_file = evaluate(rows, gt_by_file, model)
    paths = save_outputs(acc, wf1, fp, fn, y_true, y_pred, event_rows, per_file, model_kind)

    print(f'Model: {model_kind}')
    print(f'Training samples: {len(samples)} {dict(Counter(y))}')
    print(f'Accuracy: {acc:.4f}')
    print(f'Weighted F1: {wf1:.4f}')
    print(f'FP: {fp}  FN: {fn}')
    print('Per file:', per_file)
    print(classification_report(y_true, y_pred, labels=PUNCH_LABELS + (['NONE'] if 'NONE' in y_pred else []), zero_division=0))
    print('Saved:', paths)


if __name__ == '__main__':
    main()
