import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# ── 모델 파일 확인 ────────────────────────────────────────────────────────────
MODEL_PATH = 'pose_landmarker_full.task'
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("포즈 모델 다운로드 중 (약 30MB)...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
        MODEL_PATH
    )
    print("다운로드 완료!")

# ── mediapipe Tasks API 초기화 ────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── 뼈대 연결선 (인덱스 기준) ──────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # 팔
    (11, 23), (12, 24), (23, 24),                          # 몸통
    (23, 25), (25, 27), (24, 26), (26, 28),               # 다리
    (27, 29), (27, 31), (28, 30), (28, 32),               # 발
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),      # 얼굴
]

# ── 카넬로 기준값 (DNA) ───────────────────────────────────────────────────────
# 마스터 평균 데이터 (5개 영상, 2268프레임) 기반 정규화 임계값
# 어깨너비(raw ~0.106) 대비 비율
#
# 비볼 vs 카넬로 차이점:
#   가드: 카넬로는 더 낮게(0.84) vs 비볼(0.73) → 카넬로 스타일은 가드를 살짝 낮춤
#   스탠스: 카넬로 1.55x vs 비볼 1.73x → 카넬로가 더 좁은 스탠스
CANELO = {
    # 가드: 카넬로 평균 왼손 0.878 / 오른손 0.794, std ~0.83
    "guard_perfect":      0.84,   # 카넬로 평균 (이하 = 완벽)
    "guard_ok":           1.26,   # 평균 + 0.5std (이하 = 양호)
    "guard_max_ratio":    1.71,   # 평균 + 1std (초과 = 가드 다운)
    # 스탠스: 카넬로 평균 1.547x
    "stance_ratio_min":   1.10,   # 이 미만 = 너무 좁음
    "stance_ideal_min":   1.35,   # 이상적 하한
    "stance_ideal_max":   1.75,   # 이상적 상한
    "stance_ratio_max":   2.10,   # 이 초과 = 너무 넓음
    # 어깨 수평 (공통)
    "shoulder_tilt_ok":   0.15,
    "shoulder_tilt_max":  0.30,
    # 헤드: 카넬로 평균 0.316
    "head_height_good":   0.25,
    "head_height_min":    0.15,
    # 바운스 (공통)
    "bounce_target":      0.10,
    "bounce_min":         0.04,
}

# ── 유틸 함수 ─────────────────────────────────────────────────────────────────
def lm(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y])

def dist(a, b):
    return np.linalg.norm(a - b)

def draw_skeleton(image, landmarks):
    h, w = image.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(image, pts[a], pts[b], (180, 180, 180), 2)
    for pt in pts:
        cv2.circle(image, pt, 4, (255, 255, 255), -1)

def score_bar(image, x, y, score, label, color):
    bar_w = 160
    cv2.rectangle(image, (x, y), (x + bar_w, y + 14), (60, 60, 60), -1)
    fill = int(bar_w * score / 100)
    cv2.rectangle(image, (x, y), (x + fill, y + 14), color, -1)
    cv2.putText(image, f"{label}: {score}pt", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

# ── 메인 루프 ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
ankle_y_history = []
frame_count = 0
print("카넬로 모드 AI 코치 활성화! 폼을 잡아보세요. 종료: Q")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms == 0:
            timestamp_ms = frame_count * 33

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 배경 패널 (카넬로 테마: 빨간/금색)
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (310, 310), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "CANELO AI COACH", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 60, 220), 2)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            lms = results.pose_landmarks[0]
            draw_skeleton(frame, lms)

            # ── 기준 거리: 어깨너비 ──────────────────────────────────────────
            l_sh = lm(lms, 11)
            r_sh = lm(lms, 12)
            shoulder_w = dist(l_sh, r_sh)
            if shoulder_w < 1e-4:
                shoulder_w = 0.2

            nose  = lm(lms, 0)
            l_wr  = lm(lms, 15)
            r_wr  = lm(lms, 16)
            l_ank = lm(lms, 27)
            r_ank = lm(lms, 28)

            # ── [1] 가드 높이 (30pt) ─────────────────────────────────────────
            l_guard_ratio = (l_wr[1] - nose[1]) / shoulder_w
            r_guard_ratio = (r_wr[1] - nose[1]) / shoulder_w

            def guard_pts(ratio):
                if ratio <= CANELO["guard_perfect"]:   return 15
                if ratio <= CANELO["guard_ok"]:        return 10
                if ratio <= CANELO["guard_max_ratio"]: return 5
                return 0

            def guard_status(ratio, side):
                if ratio <= CANELO["guard_perfect"]:   return f"{side} Guard Perfect!", (0, 255, 100)
                if ratio <= CANELO["guard_ok"]:        return f"{side} Guard OK",        (0, 210, 100)
                if ratio <= CANELO["guard_max_ratio"]: return f"{side} Guard LOW!",      (0, 165, 255)
                return f"{side} Guard DOWN!",                                             (0, 0, 255)

            l_pts = guard_pts(l_guard_ratio)
            r_pts = guard_pts(r_guard_ratio)
            guard_score = l_pts + r_pts

            if l_guard_ratio >= r_guard_ratio:
                guard_msg, guard_col = guard_status(l_guard_ratio, "Left")
            else:
                guard_msg, guard_col = guard_status(r_guard_ratio, "Right")
            if l_pts == 15 and r_pts == 15:
                guard_msg, guard_col = "Perfect Guard!", (0, 255, 100)

            # ── [2] 스탠스 너비 (20pt) ──────────────────────────────────────
            stance = dist(l_ank, r_ank)
            stance_ratio = stance / shoulder_w
            if CANELO["stance_ideal_min"] <= stance_ratio <= CANELO["stance_ideal_max"]:
                stance_score = 20
                stance_msg, stance_col = f"Perfect Stance! ({stance_ratio:.2f}x)", (0, 255, 100)
            elif CANELO["stance_ratio_min"] <= stance_ratio < CANELO["stance_ideal_min"]:
                stance_score = 12
                stance_msg, stance_col = f"Slightly Narrow ({stance_ratio:.2f}x)", (0, 210, 100)
            elif CANELO["stance_ideal_max"] < stance_ratio <= CANELO["stance_ratio_max"]:
                stance_score = 12
                stance_msg, stance_col = f"Slightly Wide ({stance_ratio:.2f}x)", (0, 210, 100)
            elif stance_ratio < CANELO["stance_ratio_min"]:
                deficit = CANELO["stance_ratio_min"] - stance_ratio
                stance_score = max(0, int(12 - deficit * 30))
                stance_msg, stance_col = f"TOO NARROW ({stance_ratio:.2f}x)", (0, 165, 255)
            else:
                deficit = stance_ratio - CANELO["stance_ratio_max"]
                stance_score = max(0, int(12 - deficit * 30))
                stance_msg, stance_col = f"TOO WIDE ({stance_ratio:.2f}x)", (0, 165, 255)

            # ── [3] 어깨 수평 (15pt) ────────────────────────────────────────
            tilt = abs(l_sh[1] - r_sh[1]) / shoulder_w
            if tilt <= CANELO["shoulder_tilt_ok"]:
                shoulder_score = 15
                shoulder_msg, shoulder_col = "Shoulders Level!", (0, 255, 100)
            elif tilt <= CANELO["shoulder_tilt_max"]:
                shoulder_score = 8
                shoulder_msg, shoulder_col = f"Slightly Tilted ({tilt:.2f})", (0, 210, 100)
            else:
                shoulder_score = 0
                shoulder_msg, shoulder_col = f"Shoulders Tilted! ({tilt:.2f})", (0, 0, 255)

            # ── [4] 헤드 포지션 (15pt) ──────────────────────────────────────
            head_height = (((l_sh[1] + r_sh[1]) / 2) - nose[1]) / shoulder_w
            if head_height >= CANELO["head_height_good"]:
                head_score = 15
                head_msg, head_col = "Head Up!", (0, 255, 100)
            elif head_height >= CANELO["head_height_min"]:
                head_score = 8
                head_msg, head_col = f"Head Slightly Low ({head_height:.2f})", (0, 210, 100)
            else:
                head_score = 0
                head_msg, head_col = "Chin DOWN! Head up!", (0, 0, 255)

            # ── [5] 스텝 바운스 (20pt) ──────────────────────────────────────
            norm_ankle_y = l_ank[1] / shoulder_w
            ankle_y_history.append(norm_ankle_y)
            if len(ankle_y_history) > 30:
                ankle_y_history.pop(0)
            bounce = np.std(ankle_y_history) if len(ankle_y_history) == 30 else 0
            if bounce >= CANELO["bounce_target"]:
                bounce_score = 20
                bounce_msg, bounce_col = f"Active Step! ({bounce:.3f})", (0, 255, 100)
            elif bounce >= CANELO["bounce_min"]:
                ratio = (bounce - CANELO["bounce_min"]) / (CANELO["bounce_target"] - CANELO["bounce_min"])
                bounce_score = int(ratio * 20)
                bounce_msg, bounce_col = f"Move your feet ({bounce:.3f})", (0, 210, 100)
            else:
                bounce_score = 0
                bounce_msg, bounce_col = f"STATIC! Step! ({bounce:.3f})", (0, 0, 255)

            # ── 종합 점수 ────────────────────────────────────────────────────
            total = guard_score + stance_score + shoulder_score + head_score + bounce_score
            if total >= 85:
                total_col = (0, 255, 100)
                grade = "S"
            elif total >= 70:
                total_col = (0, 200, 255)
                grade = "A"
            elif total >= 50:
                total_col = (0, 165, 255)
                grade = "B"
            else:
                total_col = (0, 0, 255)
                grade = "C"

            # ── 화면 출력 ────────────────────────────────────────────────────
            y = 50
            score_bar(frame, 10, y, guard_score,    "[1] Guard",    guard_col);    y += 38
            score_bar(frame, 10, y, stance_score,   "[2] Stance",   stance_col);   y += 38
            score_bar(frame, 10, y, shoulder_score, "[3] Shoulder", shoulder_col); y += 38
            score_bar(frame, 10, y, head_score,     "[4] Head",     head_col);     y += 38
            score_bar(frame, 10, y, bounce_score,   "[5] Bounce",   bounce_col);   y += 38

            cv2.putText(frame, f"TOTAL: {total}/100  [{grade}]", (10, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, total_col, 2)

            # 피드백 메시지 (하단)
            h_img = frame.shape[0]
            for i, (msg, col) in enumerate([
                (guard_msg, guard_col), (stance_msg, stance_col),
                (shoulder_msg, shoulder_col), (head_msg, head_col), (bounce_msg, bounce_col)
            ]):
                cv2.putText(frame, msg, (10, h_img - 130 + i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        else:
            cv2.putText(frame, "자세를 감지할 수 없습니다", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

        cv2.imshow('Canelo Style AI Coach', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
