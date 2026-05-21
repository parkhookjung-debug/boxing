"""
eval_boxing.py — Boxing Game 5 & LIM Coach 5 성능 평가
═══════════════════════════════════════════════════════
Confusion Matrix + Failure Case 분석

━━ 사용법 ━━
  [게임 방어 평가]
    python eval_boxing.py --mode game --video test.mp4 --ann game_ann.csv

  [코치 펀치 평가]
    python eval_boxing.py --mode coach --video test.mp4 --ann coach_ann.csv

━━ 어노테이션 CSV 형식 ━━
  game_ann.csv:
    frame_start,frame_end,label
    150,185,BLOCK_L
    240,275,SLIP_R
    (label: BLOCK_L / BLOCK_R / SLIP_L / SLIP_R / NONE)

  coach_ann.csv:
    frame_peak,punch_type
    120,jab
    230,cross
    (punch_type: jab / cross / hook / uppercut)

━━ 출력 ━━
  eval_result/confusion_matrix.png   (raw count + normalized)
  eval_result/metrics.csv            (per-class precision/recall/F1)
  eval_result/failure_cases/         (오분류된 프레임 — 키포인트 오버레이 포함)
  eval_result/summary.txt            (텍스트 요약 리포트)
"""

import argparse, os, csv, math, sys
from collections import deque, Counter

import cv2
import numpy as np

# ── 선택 의존성 ────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOT_OK = True
except ImportError:
    _PLOT_OK = False
    print("[WARN] matplotlib/seaborn 없음 — 그래프 생략. pip install matplotlib seaborn")

try:
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        precision_score, recall_score, f1_score, accuracy_score
    )
    _SKL_OK = True
except ImportError:
    _SKL_OK = False
    print("[WARN] sklearn 없음 — pip install scikit-learn")

try:
    from rtmlib import RTMO
    import onnxruntime as ort
    _RTML_OK = True
except ImportError:
    _RTML_OK = False
    print("[WARN] rtmlib/onnxruntime 없음 — pip install rtmlib onnxruntime")

# ══════════════════════════════════════════════════════════
# 모델 초기화
# ══════════════════════════════════════════════════════════
RTMO_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
    'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip'
)
_pose_model = None

def _init_model():
    global _pose_model
    if _pose_model is not None or not _RTML_OK:
        return
    print("RTMPose (RTMO-s) 로드 중...")
    dev = 'dml' if 'DmlExecutionProvider' in ort.get_available_providers() else 'cpu'
    print(f"  device: {dev}")
    _pose_model = RTMO(RTMO_URL, backend='onnxruntime', device=dev)
    print("모델 준비 완료\n")

# ══════════════════════════════════════════════════════════
# COCO 17 키포인트 인덱스
# ══════════════════════════════════════════════════════════
KP_NOSE=0; KP_L_SH=5; KP_R_SH=6; KP_L_EL=7; KP_R_EL=8
KP_L_WR=9; KP_R_WR=10; KP_L_HI=11; KP_R_HI=12
KP_L_KN=13; KP_R_KN=14; KP_L_AN=15; KP_R_AN=16

COCO_CONN = [
    (KP_L_SH,KP_R_SH),(KP_L_SH,KP_L_EL),(KP_L_EL,KP_L_WR),
    (KP_R_SH,KP_R_EL),(KP_R_EL,KP_R_WR),
    (KP_L_SH,KP_L_HI),(KP_R_SH,KP_R_HI),(KP_L_HI,KP_R_HI),
    (KP_L_HI,KP_L_KN),(KP_L_KN,KP_L_AN),(KP_R_HI,KP_R_KN),(KP_R_KN,KP_R_AN),
    (KP_NOSE,KP_L_SH),(KP_NOSE,KP_R_SH),
]
VIS_MIN = 0.30
NEEDED  = [KP_L_SH, KP_R_SH, KP_L_WR, KP_R_WR, KP_L_EL, KP_R_EL, KP_NOSE]

# ══════════════════════════════════════════════════════════
# 공통 헬퍼
# ══════════════════════════════════════════════════════════
def get_pose(frame):
    """RTMPose 추론 → (kp, sc) or (None, None)"""
    if _pose_model is None:
        return None, None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    kps, scs = _pose_model(rgb)
    if len(kps) == 0:
        return None, None
    kp, sc = kps[0], scs[0]
    if not all(sc[i] > VIS_MIN for i in NEEDED):
        return None, None
    return kp, sc

def sw_m(kp):
    dx = kp[KP_R_SH][0]-kp[KP_L_SH][0]
    dy = kp[KP_R_SH][1]-kp[KP_L_SH][1]
    return math.sqrt(dx*dx+dy*dy) + 1e-6

def angle3pt(ax, ay, bx, by, cx, cy):
    bax, bay = ax-bx, ay-by
    bcx, bcy = cx-bx, cy-by
    dot = bax*bcx + bay*bcy
    mag = math.sqrt(bax**2+bay**2) * math.sqrt(bcx**2+bcy**2) + 1e-9
    return math.degrees(math.acos(max(-1, min(1, dot/mag))))

def arm_ext(kp, sh_i, wr_i, sw):
    return math.hypot(kp[wr_i][0]-kp[sh_i][0], kp[wr_i][1]-kp[sh_i][1]) / sw

def draw_pose_overlay(frame, kp, sc):
    """키포인트 + 스켈레톤 오버레이"""
    if kp is None:
        return frame
    vis = frame.copy()
    for a, b in COCO_CONN:
        if sc[a] > VIS_MIN and sc[b] > VIS_MIN:
            cv2.line(vis, (int(kp[a][0]), int(kp[a][1])),
                     (int(kp[b][0]), int(kp[b][1])), (60, 60, 200), 2)
    for i in range(17):
        if sc[i] > VIS_MIN:
            cv2.circle(vis, (int(kp[i][0]), int(kp[i][1])), 5, (0, 200, 255), -1)
    return vis

# ══════════════════════════════════════════════════════════
# [GAME MODE] 방어 감지 — boxing game 5 로직 그대로
# ══════════════════════════════════════════════════════════
BLOCK_NOSE_THRESH = -0.15
SLIP_THRESH       =  0.20

def spatial_arms(kp):
    if kp[KP_L_SH][0] > kp[KP_R_SH][0]:
        return {'r': [KP_L_SH,KP_L_EL,KP_L_WR], 'l': [KP_R_SH,KP_R_EL,KP_R_WR]}
    return {'r': [KP_R_SH,KP_R_EL,KP_R_WR], 'l': [KP_L_SH,KP_L_EL,KP_L_WR]}

def detect_defense(kp, sw):
    """게임 5 get_defense() 동일 로직"""
    nose_y = kp[KP_NOSE][1]
    thr    = nose_y - BLOCK_NOSE_THRESH * sw
    arms   = spatial_arms(kp)
    r_up   = kp[arms['r'][2]][1] < thr
    l_up   = kp[arms['l'][2]][1] < thr
    if r_up and not l_up:
        block = 'BLOCK_R'
    elif l_up and not r_up:
        block = 'BLOCK_L'
    else:
        block = None

    sh_cx = (kp[KP_L_SH][0]+kp[KP_R_SH][0]) / 2
    dev   = (kp[KP_NOSE][0]-sh_cx) / sw
    if   dev >  SLIP_THRESH: slip = 'SLIP_R'
    elif dev < -SLIP_THRESH: slip = 'SLIP_L'
    else:                     slip = None

    return block or slip or 'NONE'

def eval_game_segment(cap, start_f, end_f, label, fail_dir, idx):
    """
    세그먼트 [start_f, end_f] 내 방어 감지 → predicted label
    프레임별 감지 결과를 투표해서 최다 결과 반환.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    counts   = Counter()
    mid_frame = None
    mid_kp, mid_sc = None, None

    for fn in range(start_f, end_f + 1):
        ret, frame = cap.read()
        if not ret:
            break
        kp, sc = get_pose(frame)
        if kp is None:
            counts['NONE'] += 1
            continue
        sw   = sw_m(kp)
        pred = detect_defense(kp, sw)
        counts[pred] += 1
        # 세그먼트 중간 프레임 저장 (failure case 이미지용)
        if fn == (start_f + end_f) // 2:
            mid_frame = frame.copy()
            mid_kp, mid_sc = kp, sc

    # NONE 을 제외한 가장 많이 감지된 방어; 모두 NONE이면 NONE
    non_none = {k: v for k, v in counts.items() if k != 'NONE'}
    final = max(non_none, key=non_none.get) if non_none else 'NONE'

    if final != label and mid_frame is not None:
        os.makedirs(fail_dir, exist_ok=True)
        vis = draw_pose_overlay(mid_frame, mid_kp, mid_sc)
        _add_label_banner(vis, f"GT: {label}  |  PRED: {final}", ok=False)
        cv2.imwrite(os.path.join(fail_dir, f'fail_{idx:04d}_gt{label}_pred{final}.jpg'), vis)

    return final

# ══════════════════════════════════════════════════════════
# [COACH MODE] 펀치 분류 — LIM coach 5 로직 그대로
# ══════════════════════════════════════════════════════════
VEL_START   = 0.08
DOM_RATIO   = 1.15
UP_VEL_THR  = 0.07
UP_EA_THR   = 130.0
HOOK_EA_THR = 112.0
HOOK_FRAMES = 3
HOOK_EXT_MAX = 0.38
_BUF_SZ     = 12


def classify_punch(side, ea_buf, vx_buf, vy_buf, ext_buf):
    """코치 5 classify_from_buf() 동일 로직"""
    min_ea    = min(ea_buf,  default=180.0)
    max_vx    = max(vx_buf,  default=0.0)
    max_vy_up = max((-v for v in vy_buf), default=0.0)
    ext_list  = list(ext_buf)
    ext_delta = (max(ext_list) - min(ext_list)) if ext_list else 0.0

    if max_vy_up > UP_VEL_THR and min_ea < UP_EA_THR and max_vy_up > max_vx * 0.7:
        return 'uppercut'
    hook_frames = sum(1 for ea in ea_buf if ea < HOOK_EA_THR)
    if hook_frames >= HOOK_FRAMES and ext_delta < HOOK_EXT_MAX:
        return 'hook'
    return 'jab' if side == 'lead' else 'cross'


def eval_coach_window(cap, peak_f, fps, facing_right, fail_dir, idx, label):
    """
    peak_f 주변 ±0.5초 윈도우에서 펀치 감지 → predicted label
    처음 감지된 펀치 타입 반환 (미감지 시 'NONE').
    """
    half  = int(fps * 0.5)
    start = max(0, peak_f - half)
    end   = peak_f + half

    if facing_right:
        JAB_WR, JAB_SH, JAB_EL       = KP_R_WR, KP_R_SH, KP_R_EL
        CROSS_WR, CROSS_SH, CROSS_EL = KP_L_WR, KP_L_SH, KP_L_EL
    else:
        JAB_WR, JAB_SH, JAB_EL       = KP_L_WR, KP_L_SH, KP_L_EL
        CROSS_WR, CROSS_SH, CROSS_EL = KP_R_WR, KP_R_SH, KP_R_EL

    v_l  = deque(maxlen=_BUF_SZ); v_r  = deque(maxlen=_BUF_SZ)
    ea_l = deque(maxlen=_BUF_SZ); ea_r = deque(maxlen=_BUF_SZ)
    vx_l = deque(maxlen=_BUF_SZ); vx_r = deque(maxlen=_BUF_SZ)
    vy_l = deque(maxlen=_BUF_SZ); vy_r = deque(maxlen=_BUF_SZ)
    ex_l = deque(maxlen=_BUF_SZ); ex_r = deque(maxlen=_BUF_SZ)

    pv_l = 0.0; pv_r = 0.0
    pw_l = None; pw_r = None
    detected = []
    peak_frame_img, peak_kp, peak_sc = None, None, None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for fn in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        kp, sc = get_pose(frame)
        if kp is None:
            pw_l = None; pw_r = None
            continue
        sw = sw_m(kp)
        jx, jy   = kp[JAB_WR][0],   kp[JAB_WR][1]
        cx_, cy_ = kp[CROSS_WR][0], kp[CROSS_WR][1]

        vel_l = math.hypot(jx-pw_l[0], jy-pw_l[1])/sw if pw_l else 0.0
        vel_r = math.hypot(cx_-pw_r[0], cy_-pw_r[1])/sw if pw_r else 0.0
        dvx_l = abs(jx-pw_l[0])/sw if pw_l else 0.0
        dvy_l = (jy-pw_l[1])/sw    if pw_l else 0.0
        dvx_r = abs(cx_-pw_r[0])/sw if pw_r else 0.0
        dvy_r = (cy_-pw_r[1])/sw    if pw_r else 0.0

        jea_ = (angle3pt(kp[JAB_SH][0],kp[JAB_SH][1],
                         kp[JAB_EL][0],kp[JAB_EL][1],
                         kp[JAB_WR][0],kp[JAB_WR][1])
                if sc[JAB_EL]>VIS_MIN else 180.0)
        rea_ = (angle3pt(kp[CROSS_SH][0],kp[CROSS_SH][1],
                         kp[CROSS_EL][0],kp[CROSS_EL][1],
                         kp[CROSS_WR][0],kp[CROSS_WR][1])
                if sc[CROSS_EL]>VIS_MIN else 180.0)

        v_l.append(vel_l);  v_r.append(vel_r)
        ea_l.append(jea_);  ea_r.append(rea_)
        vx_l.append(dvx_l); vx_r.append(dvx_r)
        vy_l.append(dvy_l); vy_r.append(dvy_r)
        ex_l.append(arm_ext(kp, JAB_SH,   JAB_WR,   sw))
        ex_r.append(arm_ext(kp, CROSS_SH, CROSS_WR, sw))

        pk_l = max(v_l); pk_r = max(v_r)
        if pv_l > VEL_START and vel_l <= VEL_START and pk_l > pk_r * DOM_RATIO:
            detected.append(classify_punch('lead', ea_l, vx_l, vy_l, ex_l))
        if pv_r > VEL_START and vel_r <= VEL_START and pk_r > pk_l * DOM_RATIO:
            detected.append(classify_punch('rear', ea_r, vx_r, vy_r, ex_r))

        pv_l = vel_l; pv_r = vel_r
        pw_l = (jx, jy); pw_r = (cx_, cy_)

        if fn == peak_f:
            peak_frame_img = frame.copy()
            peak_kp, peak_sc = kp, sc

    pred = detected[0] if detected else 'NONE'

    if pred != label and peak_frame_img is not None:
        os.makedirs(fail_dir, exist_ok=True)
        vis = draw_pose_overlay(peak_frame_img, peak_kp, peak_sc)
        _add_label_banner(vis, f"GT: {label}  |  PRED: {pred}  |  all={detected}", ok=False)
        cv2.imwrite(os.path.join(fail_dir, f'fail_{idx:04d}_gt{label}_pred{pred}.jpg'), vis)

    return pred

# ══════════════════════════════════════════════════════════
# 이미지 배너 유틸
# ══════════════════════════════════════════════════════════
def _add_label_banner(img, text, ok=True):
    col = (0, 200, 80) if ok else (0, 60, 230)
    h = img.shape[0]
    cv2.rectangle(img, (0, h-36), (img.shape[1], h), (20, 20, 20), -1)
    cv2.putText(img, text, (8, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════
# 시각화 & 리포트
# ══════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, labels, out_path, title):
    if not (_SKL_OK and _PLOT_OK):
        return
    cm      = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 왼쪽: raw count
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=axes[0], linewidths=0.6, linecolor='#333')
    axes[0].set_title('Count', fontsize=12)
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

    # 오른쪽: row-normalized (recall 관점)
    annot_norm = np.array([[f'{v:.2f}' for v in row] for row in cm_norm])
    sns.heatmap(cm_norm, annot=annot_norm, fmt='', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels,
                ax=axes[1], linewidths=0.6, linecolor='#333',
                vmin=0, vmax=1)
    axes[1].set_title('Normalized (row = Recall)', fontsize=12)
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix 저장: {out_path}")


def save_metrics_csv(y_true, y_pred, labels, out_path):
    if not _SKL_OK:
        return
    rep  = classification_report(y_true, y_pred, labels=labels,
                                  output_dict=True, zero_division=0)
    rows = []
    for lbl in labels:
        m = rep.get(lbl, {})
        rows.append({
            'class':     lbl,
            'precision': round(m.get('precision', 0), 4),
            'recall':    round(m.get('recall',    0), 4),
            'f1':        round(m.get('f1-score',  0), 4),
            'support':   int(m.get('support',     0)),
        })
    for avg_key in ('macro avg', 'weighted avg'):
        m = rep.get(avg_key, {})
        rows.append({
            'class':     avg_key.upper().replace(' ','_'),
            'precision': round(m.get('precision', 0), 4),
            'recall':    round(m.get('recall',    0), 4),
            'f1':        round(m.get('f1-score',  0), 4),
            'support':   int(m.get('support',     0)),
        })
    rows.append({
        'class': 'ACCURACY', 'precision': '',
        'recall': '', 'f1': '',
        'support': round(accuracy_score(y_true, y_pred), 4),
    })
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['class','precision','recall','f1','support'])
        w.writeheader(); w.writerows(rows)
    print(f"[OK] Metrics CSV 저장: {out_path}")


def save_summary_txt(y_true, y_pred, labels, mode, out_path):
    lines = [
        f"══ Boxing {mode.upper()} 성능 평가 리포트 ══",
        f"샘플 수: {len(y_true)}  (정답: {sum(t==p for t,p in zip(y_true,y_pred))} / 오답: {sum(t!=p for t,p in zip(y_true,y_pred))})",
    ]
    if _SKL_OK:
        lines.append(f"전체 Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
        lines.append(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    lines.append("━━ Failure Cases ━━")
    fails = [(gt, pred) for gt, pred in zip(y_true, y_pred) if gt != pred]
    if not fails:
        lines.append("  오분류 없음!")
    else:
        for (gt, pred), cnt in Counter(fails).most_common():
            lines.append(f"  GT={gt:12s} → PRED={pred:12s}  ×{cnt}")

    txt = '\n'.join(lines)
    print('\n' + txt)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(f"\n[OK] Summary 저장: {out_path}")


def print_progress(idx, total, gt, pred):
    status = '✓' if gt == pred else '✗'
    print(f"  [{idx+1:4d}/{total}] {status}  GT={gt:12s}  PRED={pred}")

# ══════════════════════════════════════════════════════════
# 샘플 어노테이션 CSV 생성 (--make_sample 옵션)
# ══════════════════════════════════════════════════════════
def make_sample_annotations(mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if mode == 'game':
        path = os.path.join(out_dir, 'game_ann_sample.csv')
        rows = [
            ['frame_start','frame_end','label'],
            [30, 65,  'BLOCK_L'],
            [100,140, 'SLIP_R'],
            [180,215, 'BLOCK_R'],
            [260,300, 'SLIP_L'],
            [350,385, 'NONE'],
        ]
        note = ("# 형식: frame_start,frame_end,label\n"
                "# label: BLOCK_L / BLOCK_R / SLIP_L / SLIP_R / NONE\n"
                "# frame_start ~ frame_end 구간에서 해당 방어 동작 수행\n")
    else:
        path = os.path.join(out_dir, 'coach_ann_sample.csv')
        rows = [
            ['frame_peak','punch_type'],
            [60,  'jab'],
            [130, 'cross'],
            [200, 'hook'],
            [275, 'uppercut'],
            [340, 'jab'],
        ]
        note = ("# 형식: frame_peak,punch_type\n"
                "# punch_type: jab / cross / hook / uppercut\n"
                "# frame_peak: 펀치 임팩트(최대 신전) 프레임 번호\n")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        f.write(note)
        w = csv.writer(f); w.writerows(rows)
    print(f"샘플 어노테이션 생성: {path}")

# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Boxing Game 5 & LIM Coach 5 성능 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mode',  choices=['game','coach'], help='평가 모드')
    parser.add_argument('--video', help='테스트 비디오 경로')
    parser.add_argument('--ann',   help='어노테이션 CSV 경로')
    parser.add_argument('--out',   default='eval_result', help='출력 폴더 (기본: eval_result)')
    parser.add_argument('--facing_right', default=True,  type=lambda x: x.lower()!='false',
                        help='[coach] 오르토독스(우향) 여부 (기본: True)')
    parser.add_argument('--make_sample', action='store_true',
                        help='샘플 어노테이션 CSV 생성 후 종료')
    args = parser.parse_args()

    # 샘플 생성 모드
    if args.make_sample:
        if not args.mode:
            parser.error('--make_sample 사용 시 --mode 필요')
        make_sample_annotations(args.mode, args.out)
        return

    if not (args.mode and args.video and args.ann):
        parser.error('--mode, --video, --ann 모두 필요합니다.')

    # 출력 폴더
    os.makedirs(args.out, exist_ok=True)
    fail_dir = os.path.join(args.out, 'failure_cases')

    # 모델 로드
    _init_model()

    # 비디오 열기
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"비디오를 열 수 없음: {args.video}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_vid  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video : {args.video}")
    print(f"  해상도={w_vid}×{h_vid}  FPS={fps:.1f}  총 프레임={total}\n")

    # 어노테이션 로드
    with open(args.ann, newline='', encoding='utf-8') as f:
        # '#' 주석 행 제거
        lines  = [l for l in f if not l.strip().startswith('#')]
    reader = csv.DictReader(lines)
    ann_rows = list(reader)

    y_true, y_pred = [], []

    # ── GAME 모드 ──────────────────────────────────────────
    if args.mode == 'game':
        LABELS = ['BLOCK_L', 'BLOCK_R', 'SLIP_L', 'SLIP_R', 'NONE']
        print(f"[GAME] 방어 감지 평가 — {len(ann_rows)}개 세그먼트")
        print("-" * 55)
        for i, row in enumerate(ann_rows):
            sf  = int(row['frame_start'].strip())
            ef  = int(row['frame_end'].strip())
            gt  = row['label'].strip().upper()
            pred = eval_game_segment(cap, sf, ef, gt, fail_dir, i)
            y_true.append(gt); y_pred.append(pred)
            print_progress(i, len(ann_rows), gt, pred)

    # ── COACH 모드 ─────────────────────────────────────────
    elif args.mode == 'coach':
        LABELS = ['jab', 'cross', 'hook', 'uppercut']
        print(f"[COACH] 펀치 분류 평가 — {len(ann_rows)}개 이벤트")
        print(f"  facing_right = {args.facing_right}")
        print("-" * 55)
        for i, row in enumerate(ann_rows):
            pf  = int(row['frame_peak'].strip())
            gt  = row['punch_type'].strip().lower()
            pred = eval_coach_window(cap, pf, fps, args.facing_right,
                                     fail_dir, i, gt)
            y_true.append(gt); y_pred.append(pred)
            print_progress(i, len(ann_rows), gt, pred)

    cap.release()
    print("-" * 55)

    # ── 결과 저장 ──────────────────────────────────────────
    mode_title = ('Game 5 — Defense Detection'
                  if args.mode == 'game'
                  else 'Coach 5 — Punch Classification')

    plot_confusion_matrix(
        y_true, y_pred, LABELS,
        os.path.join(args.out, 'confusion_matrix.png'),
        title=f'Boxing {mode_title} | n={len(y_true)}',
    )
    save_metrics_csv(
        y_true, y_pred, LABELS,
        os.path.join(args.out, 'metrics.csv'),
    )
    save_summary_txt(
        y_true, y_pred, LABELS, args.mode,
        os.path.join(args.out, 'summary.txt'),
    )

    n_fail = sum(gt != pred for gt, pred in zip(y_true, y_pred))
    print(f"\n완료! 실패 케이스 이미지 → {fail_dir}/  ({n_fail}장)")


if __name__ == '__main__':
    main()
