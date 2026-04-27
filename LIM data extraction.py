"""
LIM data extraction.py — RTMPose (rtmlib) 버전
LIM1~5.mp4 → LIM_full_data1~5.csv (COCO 17, 정규화 좌표)

설치: pip install rtmlib onnxruntime opencv-python
"""

import cv2, csv, os
from rtmlib import RTMO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEOS = [
    ("LIM1.mp4", "LIM_full_data1.csv"),
    ("LIM2.mp4", "LIM_full_data2.csv"),
    ("LIM3.mp4", "LIM_full_data3.csv"),
    ("LIM4.mp4", "LIM_full_data4.csv"),
    ("LIM5.mp4", "LIM_full_data5.csv"),
]

# COCO 17 키포인트 이름 (MediaPipe left/right 규칙과 동일: 사람 기준)
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

print("RTMPose (RTMO-s) 로드 중...")
pose_model = RTMO(RTMO_URL, backend='onnxruntime', device='cpu')
print("모델 준비 완료")

for video_file, csv_file in VIDEOS:
    video_path = os.path.join(BASE_DIR, video_file)
    csv_path   = os.path.join(BASE_DIR, csv_file)

    if not os.path.exists(video_path):
        print(f"\n[건너뜀] 파일 없음: {video_file}")
        continue

    print(f"\n{'='*55}")
    print(f"  추출 시작: {video_file}  →  {csv_file}")
    print(f"{'='*55}")

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"  총 프레임: {total_frames}  /  FPS: {fps:.1f}")

    frame_count   = 0
    saved_count   = 0
    skipped_count = 0

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            fh, fw = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            kps, scs = pose_model(rgb)

            if len(kps) > 0:
                kp = kps[0]   # (17, 2) pixel coordinates
                sc = scs[0]   # (17,)   confidence scores

                row_data = [frame_count]
                for i in range(17):
                    nx = kp[i][0] / fw   # normalize 0~1
                    ny = kp[i][1] / fh
                    row_data.extend([round(nx, 6), round(ny, 6), 0.0, round(float(sc[i]), 4)])
                writer.writerow(row_data)
                saved_count += 1

                # 시각화
                annotated = frame.copy()
                for i in range(17):
                    if sc[i] > 0.30:
                        cx, cy = int(kp[i][0]), int(kp[i][1])
                        cv2.circle(annotated, (cx, cy), 4, (0, 200, 255), -1)

                progress = frame_count / max(total_frames, 1)
                bar_w = int(fw * progress)
                cv2.rectangle(annotated, (0, fh-14), (fw, fh), (50, 50, 50), -1)
                cv2.rectangle(annotated, (0, fh-14), (bar_w, fh), (0, 200, 100), -1)
                cv2.putText(annotated,
                    f"{video_file}  Frame {frame_count}/{total_frames}  저장:{saved_count}",
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 150), 2)
                cv2.imshow('LIM Data Extraction', annotated)
            else:
                skipped_count += 1
                cv2.putText(frame, f"[인식 실패] Frame {frame_count}",
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 60, 255), 2)
                cv2.imshow('LIM Data Extraction', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("  사용자에 의해 추출 중단.")
                break

    cap.release()
    print(f"  완료!  저장: {saved_count}프레임  /  미인식: {skipped_count}프레임  →  {csv_file}")

cv2.destroyAllWindows()
print("\n모든 LIM 영상 추출 완료!")
print("다음 단계: LIM punch extraction front.py 를 실행하세요.")
