import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import urllib.request
import os

# ── 모델 다운로드 ─────────────────────────────────────────────────
MODEL_PATH = 'pose_landmarker_full.task'
if not os.path.exists(MODEL_PATH):
    print("포즈 모델을 다운로드 중입니다 (약 30MB)...")
    url = ('https://storage.googleapis.com/mediapipe-models/'
           'pose_landmarker/pose_landmarker_full/float16/latest/'
           'pose_landmarker_full.task')
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("다운로드 완료!")

# ── 처리할 영상 목록 ──────────────────────────────────────────────
VIDEOS = [
    ("LIM1.mp4", "LIM_full_data1.csv"),
    ("LIM2.mp4", "LIM_full_data2.csv"),
    ("LIM3.mp4", "LIM_full_data3.csv"),
    ("LIM4.mp4", "LIM_full_data4.csv"),
    ("LIM5.mp4", "LIM_full_data5.csv"),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 33개 관절 이름 ────────────────────────────────────────────────
LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

header = ['frame_number']
for name in LANDMARK_NAMES:
    header.extend([f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_v'])

# ── PoseLandmarker 옵션 ───────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

# ── 영상별 추출 루프 ──────────────────────────────────────────────
for video_file, csv_file in VIDEOS:
    video_path = os.path.join(BASE_DIR, video_file)
    csv_path   = os.path.join(BASE_DIR, csv_file)

    if not os.path.exists(video_path):
        print(f"\n[건너뜀] 파일 없음: {video_file}")
        continue

    print(f"\n{'='*55}")
    print(f"  추출 시작: {video_file}  →  {csv_file}")
    print(f"{'='*55}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"  총 프레임: {total_frames}  /  FPS: {fps:.1f}")

    frame_count   = 0
    saved_count   = 0
    skipped_count = 0

    with open(csv_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results   = landmarker.detect(mp_image)

                if results.pose_landmarks and len(results.pose_landmarks) > 0:
                    frame_data = [frame_count]
                    for lm in results.pose_landmarks[0]:
                        vis = lm.visibility if lm.visibility is not None else 0.0
                        frame_data.extend([lm.x, lm.y, lm.z, vis])
                    csv_writer.writerow(frame_data)
                    saved_count += 1

                    # 시각화
                    annotated = frame.copy()
                    h, w = frame.shape[:2]
                    for lm in results.pose_landmarks[0]:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(annotated, (cx, cy), 4, (0, 200, 255), -1)

                    progress = frame_count / max(total_frames, 1)
                    bar_w    = int(w * progress)
                    cv2.rectangle(annotated, (0, h - 14), (w, h), (50, 50, 50), -1)
                    cv2.rectangle(annotated, (0, h - 14), (bar_w, h), (0, 200, 100), -1)
                    cv2.putText(annotated,
                                f"{video_file}  Frame {frame_count}/{total_frames}  "
                                f"저장:{saved_count}",
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
print("\n\n모든 LIM 영상 추출 완료!")
print("다음 단계: LIM master average.py 를 실행하세요.")
