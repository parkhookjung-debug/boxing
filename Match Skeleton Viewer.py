"""
Match Skeleton Viewer
----------------------
Match Data Extractor로 추출한 CSV를 원본 영상 위에 스켈레톤으로 재생합니다.

키 조작:
  Space : 일시정지 / 재생
  ← → : 한 프레임 이동 (일시정지 상태)
  B    : 배경 토글 (원본 영상 ↔ 검정 배경)
  Q    : 종료
"""

import cv2
import csv
import numpy as np

# ─── 설정 ────────────────────────────────────────────────────────────────────
VIDEO_PATH = "bivol match.mp4"
CSV_PATH   = "bivol_match_data.csv"

# COCO 17 관절 이름 순서
KP_NAMES = [
    "nose",
    "left_eye",    "right_eye",
    "left_ear",    "right_ear",
    "left_shoulder","right_shoulder",
    "left_elbow",  "right_elbow",
    "left_wrist",  "right_wrist",
    "left_hip",    "right_hip",
    "left_knee",   "right_knee",
    "left_ankle",  "right_ankle",
]
KP_IDX = {name: i for i, name in enumerate(KP_NAMES)}

# 스켈레톤 연결선 정의 (start, end, color BGR)
SKELETON = [
    # 얼굴
    ("nose",          "left_eye",       (180, 180, 180)),
    ("nose",          "right_eye",      (180, 180, 180)),
    ("left_eye",      "left_ear",       (160, 160, 160)),
    ("right_eye",     "right_ear",      (160, 160, 160)),
    # 상체
    ("left_shoulder", "right_shoulder", (0,  220, 255)),
    ("left_shoulder", "left_elbow",     (0,  255, 80)),
    ("left_elbow",    "left_wrist",     (0,  200, 60)),
    ("right_shoulder","right_elbow",    (80, 120, 255)),
    ("right_elbow",   "right_wrist",    (60,  90, 220)),
    # 몸통
    ("left_shoulder", "left_hip",       (0,  200, 200)),
    ("right_shoulder","right_hip",      (0,  200, 200)),
    ("left_hip",      "right_hip",      (0,  200, 200)),
    # 하체
    ("left_hip",      "left_knee",      (200, 160,  0)),
    ("left_knee",     "left_ankle",     (200, 130,  0)),
    ("right_hip",     "right_knee",     (200,  80, 80)),
    ("right_knee",    "right_ankle",    (200,  60, 60)),
]

# 관절별 색상
KP_COLOR = {
    "left_wrist":  (0, 255, 80),   "right_wrist":  (80, 120, 255),
    "left_elbow":  (0, 220, 80),   "right_elbow":  (60,  90, 220),
    "left_shoulder":(0,220,255),   "right_shoulder":(0, 220, 255),
    "left_hip":    (0, 200, 200),  "right_hip":    (0, 200, 200),
    "left_knee":   (200,160, 0),   "right_knee":   (200, 80, 80),
    "left_ankle":  (200,130, 0),   "right_ankle":  (200, 60, 60),
}
DEFAULT_KP_COLOR = (220, 220, 220)

CONF_THRESHOLD = 0.3   # 이 신뢰도 미만 관절은 그리지 않음

# ─── CSV 로드 ─────────────────────────────────────────────────────────────────
def load_csv(path):
    """frame → keypoints 배열 딕셔너리로 반환
    keypoints: ndarray (17, 3) = [x, y, conf]
    """
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_no = int(row["frame"])
            kps = np.zeros((17, 3), dtype=np.float32)
            for i, name in enumerate(KP_NAMES):
                kps[i, 0] = float(row[f"{name}_x"])
                kps[i, 1] = float(row[f"{name}_y"])
                kps[i, 2] = float(row[f"{name}_conf"])
            data[frame_no] = kps
    return data

# ─── 스켈레톤 그리기 ──────────────────────────────────────────────────────────
def draw_skeleton(canvas, kps):
    """kps: (17, 3) ndarray"""
    # 연결선
    for s_name, e_name, color in SKELETON:
        si, ei = KP_IDX[s_name], KP_IDX[e_name]
        if kps[si, 2] < CONF_THRESHOLD or kps[ei, 2] < CONF_THRESHOLD:
            continue
        pt1 = (int(kps[si, 0]), int(kps[si, 1]))
        pt2 = (int(kps[ei, 0]), int(kps[ei, 1]))
        cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)

    # 관절 점
    for i, name in enumerate(KP_NAMES):
        if kps[i, 2] < CONF_THRESHOLD:
            continue
        pt  = (int(kps[i, 0]), int(kps[i, 1]))
        col = KP_COLOR.get(name, DEFAULT_KP_COLOR)
        cv2.circle(canvas, pt, 5, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 6, (30, 30, 30), 1, cv2.LINE_AA)

# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("CSV 로딩 중...")
    frame_data = load_csv(CSV_PATH)
    if not frame_data:
        print(f"[오류] 데이터가 없습니다: {CSV_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[오류] 영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wait_ms      = max(1, int(1000 / fps))

    print(f"영상: {w_vid}×{h_vid}  /  {total_frames}프레임  /  {fps:.1f}fps")
    print(f"CSV 데이터: {len(frame_data)}프레임")

    paused      = False
    show_video  = True   # B키: 원본↔검정 배경
    frame_no    = 0

    cv2.namedWindow("Match Skeleton Viewer", cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, raw_frame = cap.read()
            if not ret:
                break
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 배경 선택
        if show_video:
            canvas = raw_frame.copy()
        else:
            canvas = np.zeros((h_vid, w_vid, 3), dtype=np.uint8)

        # 스켈레톤 그리기
        if frame_no in frame_data:
            draw_skeleton(canvas, frame_data[frame_no])
        else:
            # CSV에 없는 프레임 (감지 실패 프레임)
            cv2.putText(canvas, f"No detection  frame {frame_no}",
                        (20, h_vid // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)

        # HUD
        pct     = frame_no / max(total_frames, 1) * 100
        bar_w   = int(w_vid * 0.6)
        bar_x, bar_y = 20, h_vid - 28
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (40, 40, 40), -1)
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + int(bar_w * frame_no / max(total_frames, 1)), bar_y + 12),
                      (0, 200, 80), -1)

        status = "II PAUSE" if paused else "▶ PLAY"
        bg_lbl = "원본" if show_video else "검정"
        cv2.putText(canvas,
                    f"{status}  Frame {frame_no}/{total_frames} ({pct:.1f}%)  "
                    f"B:배경[{bg_lbl}]  Space:일시정지  ←→:프레임이동  Q:종료",
                    (bar_x, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Match Skeleton Viewer", canvas)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('b'):
            show_video = not show_video
        elif key == 81 or key == ord(','):   # ← 또는 ,
            if paused:
                frame_no = max(1, frame_no - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
                ret, raw_frame = cap.read()
        elif key == 83 or key == ord('.'):   # → 또는 .
            if paused:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, raw_frame = cap.read()
                frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
