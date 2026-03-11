import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import urllib.request
import os

# 1. 포즈 모델 다운로드 (없을 경우 자동 다운로드)
MODEL_PATH = 'pose_landmarker_full.task'
if not os.path.exists(MODEL_PATH):
    print("포즈 모델을 다운로드 중입니다 (약 30MB)...")
    url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("다운로드 완료!")

# 2. 파일 경로 설정
VIDEO_PATH = 'garcia6.mp4'
CSV_PATH = 'garcia_full_data6.csv'

# 3. 33개 관절 이름 (mediapipe 표준 순서)
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

# 132개 컬럼 헤더 생성 (33관절 × 4값)
header = ['frame_number']
for name in LANDMARK_NAMES:
    header.extend([f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_v'])

# 4. PoseLandmarker 초기화
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

print("데이터 추출을 시작합니다. 영상이 끝날 때까지 기다려주세요...")

with open(CSV_PATH, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(header)

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # BGR → RGB 변환 후 mediapipe Image로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = landmarker.detect(mp_image)

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                frame_data = [frame_count]

                for lm in results.pose_landmarks[0]:
                    vis = lm.visibility if lm.visibility is not None else 0.0
                    frame_data.extend([lm.x, lm.y, lm.z, vis])

                csv_writer.writerow(frame_data)

                # --- 진행 상황 시각화 ---
                annotated = frame.copy()
                h, w = frame.shape[:2]
                for lm in results.pose_landmarks[0]:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (cx, cy), 3, (245, 117, 66), -1)

                cv2.putText(annotated, f"Deep Extraction... Frame: {frame_count}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Bivol Deep Data Extraction', annotated)

            # 'q' 키를 누르면 강제 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("사용자에 의해 추출이 중단되었습니다.")
                break

cap.release()
cv2.destroyAllWindows()
print(f"정밀 추출 완료! 총 {frame_count} 프레임의 전체 관절 데이터가 {CSV_PATH}에 성공적으로 저장되었습니다.")
