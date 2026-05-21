"""
train_lstm.py — LSTM 펀치 분류기 학습
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용법:
  python train_lstm.py                      # LIM1~4 학습, LIM5 테스트
  python train_lstm.py --gt_labels l1.csv ...   # 수동 GT 사용
  python train_lstm.py --eval_only          # 저장된 모델 평가만

출력:
  punch_lstm.pt         학습된 모델 가중치
  lstm_metrics.txt      테스트셋 분류 보고서
"""

import sys, os, math, argparse
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_from_csv import (
    load_full_data, get_gt_events, match_events, load_manual_labels,
    g, sw, elbow_angle_row, arm_ext_row, ew_dist_row,
    BASE_DIR, DEFAULT_FILES, MATCH_WIN,
    VEL_START_RAW, DOM_RATIO_C5, PUNCH_CD_FRAMES, EW_HOOK_MAX_C5,
)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ── 설정 ──────────────────────────────────────────────────────────
SEQ_LEN    = 20        # 이벤트 전후로 수집할 프레임 수 (전 14 + 후 6)
WIN_BEFORE = 14        # 트리거 프레임 기준 앞쪽 프레임
WIN_AFTER  = 6         # 트리거 프레임 기준 뒤쪽 프레임
N_FEATURES = 10        # 프레임당 특징 수
EPOCHS     = 60
BATCH_SZ   = 32
LR         = 1e-3
HIDDEN     = 64
N_CLASSES  = 4
LABELS     = ['jab', 'cross', 'hook', 'uppercut']
MODEL_PATH = os.path.join(BASE_DIR, 'punch_lstm.pt')
_BUF_SZ    = 12        # eval_from_csv 와 동일

assert WIN_BEFORE + WIN_AFTER == SEQ_LEN, "SEQ_LEN 불일치"


# ══════════════════════════════════════════════════════════════════
# 특징 추출 — get_pred_events 의 버퍼 계산 로직을 재사용
# ══════════════════════════════════════════════════════════════════
def _extract_features(rows, orthodox):
    """
    각 프레임의 10차원 특징 벡터를 계산해서 리스트로 반환.
    인덱스는 rows 와 1:1 대응.

    features[i] = [
      raw_l/sw,  raw_r/sw,    # 손목 이동 속도 (정규화)
      dvx_l,     dvx_r,       # 손목 가로 속도
      dvy_l,     dvy_r,       # 손목 세로 속도 (위=양수)
      ew_l,      ew_r,        # 팔꿈치-손목 거리 / sw
      ext_l,     ext_r,       # 어깨-손목 거리 / sw  (팔 연장)
    ]
    """
    if orthodox:
        JWR, JSH, JEL = 'left_wrist',  'left_shoulder',  'left_elbow'
        CWR, CSH, CEL = 'right_wrist', 'right_shoulder', 'right_elbow'
    else:
        JWR, JSH, JEL = 'right_wrist', 'right_shoulder', 'right_elbow'
        CWR, CSH, CEL = 'left_wrist',  'left_shoulder',  'left_elbow'

    feats = []
    pw_l = pw_r = None
    for row in rows:
        sv = sw(row)
        jx, jy   = g(row, JWR, 'x'), g(row, JWR, 'y')
        cx_, cy_ = g(row, CWR, 'x'), g(row, CWR, 'y')

        raw_l = math.hypot(jx - pw_l[0], jy - pw_l[1])   if pw_l else 0.0
        raw_r = math.hypot(cx_ - pw_r[0], cy_ - pw_r[1]) if pw_r else 0.0
        dvx_l = abs(jx - pw_l[0]) / sv   if pw_l else 0.0
        dvy_l = -(jy - pw_l[1])   / sv   if pw_l else 0.0   # 위=양수
        dvx_r = abs(cx_ - pw_r[0]) / sv  if pw_r else 0.0
        dvy_r = -(cy_ - pw_r[1])   / sv  if pw_r else 0.0

        ew_l = ew_dist_row(row, JEL, JWR, sv)
        ew_r = ew_dist_row(row, CEL, CWR, sv)
        ext_l = arm_ext_row(row, JSH, JWR, sv)
        ext_r = arm_ext_row(row, CSH, CWR, sv)

        feats.append([
            raw_l / sv, raw_r / sv,
            dvx_l, dvx_r,
            dvy_l, dvy_r,
            ew_l, ew_r,
            ext_l, ext_r,
        ])
        pw_l = (jx, jy)
        pw_r = (cx_, cy_)
    return feats  # list[list[float]], shape (n_frames, N_FEATURES)


def _get_trigger_frames(rows, orthodox):
    """
    eval_from_csv get_pred_events 와 동일한 falling-edge 트리거로
    (frame_idx, side) 튜플 목록 반환.
    side: 'lead' | 'rear'
    """
    if orthodox:
        JWR = 'left_wrist';  CWR = 'right_wrist'
    else:
        JWR = 'right_wrist'; CWR = 'left_wrist'

    rv_l = deque(maxlen=_BUF_SZ); rv_r = deque(maxlen=_BUF_SZ)
    ew_l = deque(maxlen=_BUF_SZ); ew_r = deque(maxlen=_BUF_SZ)
    prv_l = prv_r = 0.0
    pw_l = pw_r = None
    last_fired = -PUNCH_CD_FRAMES
    triggers = []

    if orthodox:
        JEL = 'left_elbow';  CEL = 'right_elbow'
    else:
        JEL = 'right_elbow'; CEL = 'left_elbow'

    for idx, row in enumerate(rows):
        sv = sw(row)
        jx, jy   = g(row, JWR, 'x'), g(row, JWR, 'y')
        cx_, cy_ = g(row, CWR, 'x'), g(row, CWR, 'y')

        raw_l = math.hypot(jx - pw_l[0], jy - pw_l[1])   if pw_l else 0.0
        raw_r = math.hypot(cx_ - pw_r[0], cy_ - pw_r[1]) if pw_r else 0.0

        rv_l.append(raw_l); rv_r.append(raw_r)
        ew_l.append(ew_dist_row(row, JEL, JWR, sv))
        ew_r.append(ew_dist_row(row, CEL, CWR, sv))

        pk_l = max(rv_l); pk_r = max(rv_r)
        cd_ok = (idx - last_fired) >= PUNCH_CD_FRAMES

        r_l = list(ew_l)[-4:]; r_r = list(ew_r)[-4:]
        hl = (sum(r_l)/len(r_l) if r_l else 1.0) < EW_HOOK_MAX_C5
        hr = (sum(r_r)/len(r_r) if r_r else 1.0) < EW_HOOK_MAX_C5
        vt_l = VEL_START_RAW * 0.5 if hl else VEL_START_RAW
        vt_r = VEL_START_RAW * 0.5 if hr else VEL_START_RAW
        dm_l = 1.0 if hl else DOM_RATIO_C5
        dm_r = 1.0 if hr else DOM_RATIO_C5

        fired = False
        if prv_l > vt_l and raw_l <= vt_l and pk_l > pk_r * dm_l and cd_ok:
            triggers.append((idx, 'lead'))
            last_fired = idx; fired = True
        if not fired and prv_r > vt_r and raw_r <= vt_r and pk_r > pk_l * dm_r and cd_ok:
            triggers.append((idx, 'rear'))
            last_fired = idx; fired = True
        if fired:
            rv_l.clear(); rv_r.clear(); prv_l = prv_r = 0.0
        prv_l = raw_l; prv_r = raw_r
        pw_l = (jx, jy); pw_r = (cx_, cy_)
    return triggers


# ══════════════════════════════════════════════════════════════════
# 학습 샘플 생성
# ══════════════════════════════════════════════════════════════════
def build_samples(rows, gt_events, orthodox):
    """
    GT 이벤트와 가장 가까운 트리거 프레임을 매칭해
    (sequence, label) 쌍 목록 반환.

    sequence shape : (SEQ_LEN, N_FEATURES)
    label           : 0=jab, 1=cross, 2=hook, 3=uppercut
    """
    feats = _extract_features(rows, orthodox)    # (n_frames, N_FEATURES)
    feats_arr = np.array(feats, dtype=np.float32)

    triggers = _get_trigger_frames(rows, orthodox)
    trigger_frames = [t[0] for t in triggers]

    samples = []
    label_map = {l: i for i, l in enumerate(LABELS)}
    n = len(rows)

    for gt_frame, gt_label in gt_events:
        if gt_label not in label_map:
            continue
        # 가장 가까운 트리거 찾기 (±MATCH_WIN 이내)
        best = None; best_dist = MATCH_WIN + 1
        for tf in trigger_frames:
            d = abs(tf - gt_frame)
            if d <= MATCH_WIN and d < best_dist:
                best_dist = d; best = tf

        anchor = best if best is not None else gt_frame
        start = max(0, anchor - WIN_BEFORE)
        end   = min(n, anchor + WIN_AFTER)
        seq = feats_arr[start:end]

        # 길이 부족 시 패딩 (앞뒤 0 패딩)
        if len(seq) < SEQ_LEN:
            pad_b = SEQ_LEN - len(seq)
            seq = np.concatenate([np.zeros((pad_b, N_FEATURES), dtype=np.float32), seq], axis=0)
        seq = seq[:SEQ_LEN]

        samples.append((seq, label_map[gt_label]))
    return samples


# ══════════════════════════════════════════════════════════════════
# 데이터셋
# ══════════════════════════════════════════════════════════════════
class PunchDataset(Dataset):
    def __init__(self, samples):
        self.X = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32)
        self.y = torch.tensor([s[1] for s in samples], dtype=torch.long)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════
# 모델
# ══════════════════════════════════════════════════════════════════
class PunchLSTM(nn.Module):
    def __init__(self, n_feat=N_FEATURES, hidden=HIDDEN, n_cls=N_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True,
                            dropout=0.3)
        self.fc   = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_cls),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # 마지막 타임스텝


# ══════════════════════════════════════════════════════════════════
# 추론 함수 (eval pipeline 교체용)
# ══════════════════════════════════════════════════════════════════
_lstm_model  = None
_lstm_device = None

def load_lstm_model(path=MODEL_PATH):
    global _lstm_model, _lstm_device
    _lstm_device = torch.device('cpu')
    _lstm_model  = PunchLSTM().to(_lstm_device)
    _lstm_model.load_state_dict(torch.load(path, map_location=_lstm_device,
                                           weights_only=True))
    _lstm_model.eval()


def classify_lstm(seq_np):
    """
    seq_np : numpy array shape (SEQ_LEN, N_FEATURES)
    반환값 : str  ('jab' | 'cross' | 'hook' | 'uppercut')
    """
    if _lstm_model is None:
        load_lstm_model()
    x = torch.tensor(seq_np[np.newaxis], dtype=torch.float32).to(_lstm_device)
    with torch.no_grad():
        logits = _lstm_model(x)
    return LABELS[logits.argmax(dim=1).item()]


def get_pred_lstm(rows, orthodox):
    """
    LSTM 분류기를 사용하는 전체 예측 파이프라인.
    반환: [(frame_idx, label), ...]
    """
    if _lstm_model is None:
        load_lstm_model()
    feats_arr = np.array(_extract_features(rows, orthodox), dtype=np.float32)
    triggers  = _get_trigger_frames(rows, orthodox)
    n = len(rows)
    events = []
    for tf, _side in triggers:
        start = max(0, tf - WIN_BEFORE)
        end   = min(n, tf + WIN_AFTER)
        seq   = feats_arr[start:end]
        if len(seq) < SEQ_LEN:
            pad_b = SEQ_LEN - len(seq)
            seq = np.concatenate([np.zeros((pad_b, N_FEATURES), dtype=np.float32), seq], axis=0)
        seq = seq[:SEQ_LEN]
        label = classify_lstm(seq)
        events.append((tf, label))
    return events


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files',      nargs='+', default=DEFAULT_FILES)
    parser.add_argument('--stance',     default='orthodox')
    parser.add_argument('--gt_labels',  nargs='+', default=None)
    parser.add_argument('--test_idx',   type=int, default=-1,
                        help='테스트로 뺄 파일 인덱스 (-1=마지막)')
    parser.add_argument('--eval_only',  action='store_true')
    parser.add_argument('--epochs',     type=int, default=EPOCHS)
    args = parser.parse_args()

    orthodox = (args.stance == 'orthodox')

    # ── 데이터 로드 ──────────────────────────────────────────────
    print("데이터 로드 중...")
    all_data, all_gt, fnames = [], [], []
    for i, fname in enumerate(args.files):
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [SKIP] {fname}")
            continue
        rows = load_full_data(fpath)
        if args.gt_labels and i < len(args.gt_labels):
            lp = os.path.join(BASE_DIR, args.gt_labels[i])
            gt = load_manual_labels(lp)
            tag = f"수동 GT {len(gt)}개"
        else:
            gt = get_gt_events(rows, orthodox=orthodox)
            tag = f"pseudo-GT {len(gt)}개"
        all_data.append(rows); all_gt.append(gt); fnames.append(fname)
        print(f"  {fname}: {tag}")

    if not all_data:
        print("로드된 파일 없음"); return

    # ── 학습/테스트 분할 ──────────────────────────────────────────
    test_i = args.test_idx % len(all_data)
    train_idx = [i for i in range(len(all_data)) if i != test_i]
    print(f"\n학습: {[fnames[i] for i in train_idx]}")
    print(f"테스트: {fnames[test_i]}\n")

    # ── 샘플 생성 ─────────────────────────────────────────────────
    train_samples, test_samples = [], []
    for i in train_idx:
        s = build_samples(all_data[i], all_gt[i], orthodox)
        train_samples.extend(s)
        print(f"  [train] {fnames[i]}: {len(s)}샘플")
    s = build_samples(all_data[test_i], all_gt[test_i], orthodox)
    test_samples.extend(s)
    print(f"  [test]  {fnames[test_i]}: {len(s)}샘플")

    if not train_samples:
        print("학습 샘플 없음"); return

    # 클래스 분포 출력
    from collections import Counter
    train_dist = Counter(LABELS[s[1]] for s in train_samples)
    test_dist  = Counter(LABELS[s[1]] for s in test_samples)
    print(f"\n학습 클래스 분포: {dict(train_dist)}")
    print(f"테스트 클래스 분포: {dict(test_dist)}\n")

    if args.eval_only:
        if not os.path.exists(MODEL_PATH):
            print(f"모델 파일 없음: {MODEL_PATH}"); return
        load_lstm_model()
        _run_eval(test_samples)
        return

    # ── 학습 ─────────────────────────────────────────────────────
    device = torch.device('cpu')
    model  = PunchLSTM().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit   = nn.CrossEntropyLoss()
    loader = DataLoader(PunchDataset(train_samples), batch_size=BATCH_SZ,
                        shuffle=True, drop_last=False)

    # 학습률 스케줄러
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0; correct = 0; total = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss   = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)
        sched.step()

        avg_loss = total_loss / total
        acc      = correct / total
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  acc={acc:.3f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n[OK] 모델 저장 → {MODEL_PATH}")

    # ── 테스트 평가 ───────────────────────────────────────────────
    load_lstm_model()
    _run_eval(test_samples)

    # ── LSTM 기반 전체 F1 평가 ────────────────────────────────────
    print("\n[LSTM 파이프라인 전체 평가]")
    y_true_all, y_pred_all = [], []
    for i, (rows, gt_events) in enumerate(zip(all_data, all_gt)):
        pred = get_pred_lstm(rows, orthodox)
        yt, yp, _, _ = match_events(gt_events, pred)
        y_true_all.extend(yt); y_pred_all.extend(yp)
    labels = [l for l in LABELS if l in set(y_true_all)]
    from sklearn.metrics import f1_score, accuracy_score
    wf1 = f1_score(y_true_all, y_pred_all, labels=labels,
                   average='weighted', zero_division=0)
    acc = accuracy_score(y_true_all, y_pred_all)
    print(f"  전체 weighted F1 = {wf1:.4f}  accuracy = {acc:.4f}")


def _run_eval(test_samples):
    if not test_samples:
        print("테스트 샘플 없음"); return
    seqs = np.stack([s[0] for s in test_samples]).astype(np.float32)
    true_labels = [LABELS[s[1]] for s in test_samples]

    preds = []
    for seq in seqs:
        preds.append(classify_lstm(seq))

    report = classification_report(true_labels, preds,
                                   labels=LABELS, zero_division=0)
    print("\n[테스트셋 분류 보고서]")
    print(report)

    metrics_path = os.path.join(BASE_DIR, 'lstm_metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[OK] 결과 → {metrics_path}")


if __name__ == '__main__':
    main()
