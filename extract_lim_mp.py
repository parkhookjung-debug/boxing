"""
extract_lim_mp.py — MediaPipe(BlazePose) 경량 모델로 LIM 영상 포즈 추출
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
기존 extract_lim.py 는 RTMO(무거운 모델)로 CSV를 뽑았다. 모바일/실시간을
위해 런타임을 MediaPipe 경량 모델로 바꾸려면, 학습 데이터(CSV)도 같은
모델로 다시 뽑아 좌표 규칙을 통일해야 한다.

MediaPipe Pose 는 33개 랜드마크를 내지만 그중 17개가 COCO-17 과 1:1
대응되므로, 33→17 매핑만 하면 기존 CSV 포맷·피처 모듈·윈도우 분류기를
그대로 재사용할 수 있다. (출력 헤더는 extract_lim.py 와 완전히 동일)

사용법:
  python extract_lim_mp.py              # 8개 전체 추출
  python extract_lim_mp.py --only 1 3   # 1,3번만
  python extract_lim_mp.py --skip 2     # 2번만 건너뜀
  python extract_lim_mp.py --model full # full.task 사용 (기본: lite)
  python extract_lim_mp.py --show       # 추출 화면 표시 (기본: 콘솔만)

출력: LIM_full_data1~8.csv  (덮어쓰기 전 csv_backup_rtmo/ 에 자동 백업)
"""

import argparse
import csv
import os
import shutil
import time

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEOS = [
    ("LIM 1.mp4", "LIM_full_data1.csv"),
    ("LIM 2.mp4", "LIM_full_data2.csv"),
    ("LIM 3.mp4", "LIM_full_data3.csv"),
    ("LIM 4.mp4", "LIM_full_data4.csv"),
    ("LIM 5.mp4", "LIM_full_data5.csv"),
    ("LIM 6.mp4", "LIM_full_data6.csv"),
    ("LIM 7.mp4", "LIM_full_data7.csv"),
    ("LIM 8.mp4", "LIM_full_data8.csv"),
]

# ── COCO-17 (사람 기준 left/right — MediaPipe 와 동일 규칙) ──────────
COCO17_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# COCO-17 인덱스 → MediaPipe BlazePose 33 랜드마크 인덱스
#   MP: 0 nose / 2 L_eye / 5 R_eye / 7 L_ear / 8 R_ear /
#       11 L_shoulder / 12 R_shoulder / 13 L_elbow / 14 R_elbow /
#       15 L_wrist / 16 R_wrist / 23 L_hip / 24 R_hip /
#       25 L_knee / 26 R_knee / 27 L_ankle / 28 R_ankle
COCO_FROM_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

header = ['frame_number']
for name in COCO17_NAMES:
    header.extend([f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_v'])

# 시각화용 스켈레톤 (COCO-17 인덱스)
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


def make_landmarker(model_path):
    """영상마다 새로 만든다 — VIDEO 모드는 타임스탬프가 단조 증가해야 하므로."""
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_one(video_file, csv_file, model_path, show):
    video_path = os.path.join(BASE_DIR, video_file)
    csv_path = os.path.join(BASE_DIR, csv_file)

    if not os.path.exists(video_path):
        print(f"[SKIP] 파일 없음: {video_file}")
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"\n{'='*58}")
    print(f"  {video_file}  ({total_frames}프레임 @ {fps:.1f}fps)  ->  {csv_file}")
    print(f"{'='*58}")

    landmarker = make_landmarker(model_path)
    frame_count = saved = skipped = 0
    vis_sum = 0.0
    t0 = time.time()
    last_ts = -1

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            fh, fw = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # 단조 증가 타임스탬프 (ms)
            ts = int(frame_count * 1000.0 / fps)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            result = landmarker.detect_for_video(mp_image, ts)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]   # 33개 NormalizedLandmark
                row = [frame_count]
                kp_px = []
                for mp_idx in COCO_FROM_MP:
                    p = lm[mp_idx]
                    row.extend([round(p.x, 6), round(p.y, 6),
                                round(p.z, 6), round(p.visibility, 4)])
                    kp_px.append((p.x * fw, p.y * fh, p.visibility))
                    vis_sum += p.visibility
                writer.writerow(row)
                saved += 1

                if show:
                    ann = frame.copy()
                    for a, b in SKELETON:
                        if kp_px[a][2] > 0.3 and kp_px[b][2] > 0.3:
                            cv2.line(ann, (int(kp_px[a][0]), int(kp_px[a][1])),
                                     (int(kp_px[b][0]), int(kp_px[b][1])),
                                     (170, 170, 180), 2)
                    for x, y, v in kp_px:
                        if v > 0.3:
                            cv2.circle(ann, (int(x), int(y)), 4,
                                       (0, 200, 255), -1)
                    cv2.putText(ann,
                                f"{video_file}  {frame_count}/{total_frames}"
                                f"  saved:{saved}",
                                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 150), 2)
                    cv2.imshow('LIM Extract (MediaPipe)', ann)
            else:
                skipped += 1
                if show:
                    cv2.putText(frame, f"[No pose] {frame_count}",
                                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 60, 255), 2)
                    cv2.imshow('LIM Extract (MediaPipe)', frame)

            if show and (cv2.waitKey(1) & 0xFF == ord('q')):
                print("  중단됨")
                break
            elif frame_count % 200 == 0:
                print(f"    ...{frame_count}/{total_frames}  (saved {saved})")

    cap.release()
    landmarker.close()

    dt = time.time() - t0
    proc_fps = frame_count / dt if dt > 0 else 0.0
    det_rate = saved / frame_count * 100 if frame_count else 0.0
    avg_vis = vis_sum / (saved * 17) if saved else 0.0
    print(f"  완료: 저장 {saved} / 미인식 {skipped}  "
          f"(인식률 {det_rate:.1f}%, 평균 visibility {avg_vis:.2f})")
    print(f"  추출 속도: {proc_fps:.1f} fps  ({dt:.1f}s)  =>  {csv_file}")
    return dict(video=video_file, saved=saved, skipped=skipped,
                det_rate=det_rate, avg_vis=avg_vis, proc_fps=proc_fps)


def backup_existing(targets):
    """덮어쓸 CSV 들을 csv_backup_rtmo/ 로 1회 백업 (이미 있으면 건너뜀)."""
    backup_dir = os.path.join(BASE_DIR, 'csv_backup_rtmo')
    os.makedirs(backup_dir, exist_ok=True)
    n = 0
    for _, csv_file in targets:
        src = os.path.join(BASE_DIR, csv_file)
        dst = os.path.join(backup_dir, csv_file)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            n += 1
    if n:
        print(f"기존 RTMO CSV {n}개 백업 -> csv_backup_rtmo/")


def main():
    ap = argparse.ArgumentParser(description='LIM 영상 MediaPipe 포즈 추출')
    ap.add_argument('--skip', nargs='+', type=int, default=[],
                    help='건너뜀 인덱스 (1-based)')
    ap.add_argument('--only', nargs='+', type=int, default=[],
                    help='이것만 추출 (1-based)')
    ap.add_argument('--model', choices=['lite', 'full'], default='lite',
                    help='MediaPipe 모델 (기본: lite — 모바일/실시간용)')
    ap.add_argument('--show', action='store_true',
                    help='추출 화면 표시 (기본: 콘솔 진행만)')
    args = ap.parse_args()

    model_path = os.path.join(BASE_DIR, f'pose_landmarker_{args.model}.task')
    if not os.path.exists(model_path):
        raise SystemExit(f'모델 파일 없음: {model_path}')
    print(f'MediaPipe 모델: pose_landmarker_{args.model}.task')

    targets = [(i, v, c) for i, (v, c) in enumerate(VIDEOS, 1)
               if (not args.only or i in args.only) and i not in args.skip]
    backup_existing([(v, c) for _, v, c in targets])

    summary = []
    for _, vf, cf in targets:
        r = extract_one(vf, cf, model_path, args.show)
        if r:
            summary.append(r)

    if args.show:
        cv2.destroyAllWindows()

    print(f"\n{'='*58}\n  추출 완료 요약\n{'='*58}")
    for r in summary:
        print(f"  {r['video']:12s}  saved {r['saved']:4d}  "
              f"인식률 {r['det_rate']:5.1f}%  vis {r['avg_vis']:.2f}  "
              f"{r['proc_fps']:.0f}fps")
    print("\n다음 단계: python LIM_train_window.py  (윈도우 분류기 재학습)")


if __name__ == '__main__':
    main()
