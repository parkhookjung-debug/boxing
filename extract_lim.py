"""
extract_lim.py — LIM 1~4.mp4 포즈 추출
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용법:
  python extract_lim.py            # 전체 4개 추출
  python extract_lim.py --skip 2  # 2번만 건너뜀
  python extract_lim.py --only 1 3  # 1,3번만 추출

출력: LIM_full_data1~4.csv  (기존 파일 덮어씌움)

뷰 정보 (참고용):
  LIM1, LIM3 → 정면
  LIM2, LIM4 → 측면
"""

import cv2, csv, os, argparse

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

COCO17_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
RTMO_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
    'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip'
)

header = ['frame_number']
for name in COCO17_NAMES:
    header.extend([f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_v'])


def extract_one(video_file, csv_file, pose_model):
    video_path = os.path.join(BASE_DIR, video_file)
    csv_path   = os.path.join(BASE_DIR, csv_file)

    if not os.path.exists(video_path):
        print(f"[SKIP] 파일 없음: {video_file}")
        return

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n{'='*55}")
    print(f"  {video_file}  ({total_frames}프레임 @ {fps:.1f}fps)  →  {csv_file}")
    print(f"{'='*55}")

    frame_count = saved = skipped = 0

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
            kps, scs = pose_model(rgb)

            if len(kps) > 0:
                kp = kps[0]
                sc = scs[0]
                row_data = [frame_count]
                for i in range(17):
                    row_data.extend([round(kp[i][0]/fw, 6), round(kp[i][1]/fh, 6),
                                     0.0, round(float(sc[i]), 4)])
                writer.writerow(row_data)
                saved += 1

                annotated = frame.copy()
                for i in range(17):
                    if sc[i] > 0.30:
                        cv2.circle(annotated, (int(kp[i][0]), int(kp[i][1])), 4, (0, 200, 255), -1)
                bar_w = int(fw * frame_count / max(total_frames, 1))
                cv2.rectangle(annotated, (0, fh-14), (fw, fh), (50,50,50), -1)
                cv2.rectangle(annotated, (0, fh-14), (bar_w, fh), (0,200,100), -1)
                cv2.putText(annotated,
                    f"{video_file}  {frame_count}/{total_frames}  saved:{saved}",
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,150), 2)
                cv2.imshow('LIM Extract', annotated)
            else:
                skipped += 1
                cv2.putText(frame, f"[No pose] {frame_count}",
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,60,255), 2)
                cv2.imshow('LIM Extract', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("  중단됨")
                break

    cap.release()
    print(f"  완료: 저장 {saved}  /  미인식 {skipped}  =>  {csv_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', nargs='+', type=int, default=[],
                        help='건너뜀 인덱스 (1-based)')
    parser.add_argument('--only', nargs='+', type=int, default=[],
                        help='이것만 추출 (1-based)')
    args = parser.parse_args()

    from rtmlib import RTMO
    print("RTMPose (RTMO-s) 로드 중...")
    pose_model = RTMO(RTMO_URL, backend='onnxruntime', device='cpu')
    print("모델 준비 완료\n")

    for i, (vf, cf) in enumerate(VIDEOS, 1):
        if args.only and i not in args.only:
            continue
        if i in args.skip:
            print(f"[SKIP] {i}번 건너뜀")
            continue
        extract_one(vf, cf, pose_model)

    cv2.destroyAllWindows()
    print("\n추출 완료!")
    print("다음: python label_tool.py \"LIM 1.mp4\" --out LIM1_labels.csv")


if __name__ == '__main__':
    main()
