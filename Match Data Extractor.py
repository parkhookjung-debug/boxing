"""
Match Data Extractor
---------------------
시합 영상에서 특정 선수(타겟)의 관절 데이터를 안정적으로 추출합니다.

주요 기능:
  - YOLOv8-pose + BoT-SORT: 외형 특징 기반 추적 (클린치에서 ID 유지)
  - Anti-ID Switching: 타겟 ID가 사라지면 마지막 위치 기준 가장 가까운 사람을 재연결
  - 심판 필터링: 면적 + 위치 휴리스틱으로 심판 박스 제거
  - 첫 프레임에서 클릭으로 타겟 선수 지정
  - CSV 저장 (COCO 17개 관절)

사용법:
  1. VIDEO_PATH / CSV_PATH 설정
  2. 실행 후 첫 화면에서 타겟 선수를 클릭
  3. 'q': 종료 / 'r': 타겟 재선택

의존성:
  pip install ultralytics lapx
"""

import cv2
import csv
import math
import numpy as np
from ultralytics import YOLO

# ─── 설정 ─────────────────────────────────────────────────────────────────────
VIDEO_PATH   = "bivol match.mp4"
CSV_PATH     = "bivol_match_data.csv"
MODEL_PATH   = "yolov8x-pose.pt"    # 없으면 자동 다운로드

# ID 스위칭 방지: 타겟이 사라졌을 때 재연결 허용 반경 (픽셀)
RECONNECT_DIST = 220

# 심판 필터: 타겟 박스 면적 대비 이 배수 이상 작은 박스는 심판 후보로 무시
REFEREE_AREA_RATIO = 0.35

# COCO 17개 관절 이름 (YOLOv8-pose 표준)
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

# ─── 전역 상태 ────────────────────────────────────────────────────────────────
TARGET_ID        = None
last_known_center = None
last_known_area   = None
click_point       = None   # 마우스 클릭 좌표

def on_mouse(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def point_in_box(pt, box):
    x1, y1, x2, y2 = box
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

def is_referee(box, target_area):
    """박스 면적이 타겟보다 훨씬 작으면 심판/코너 후보로 무시"""
    if target_area is None or target_area == 0:
        return False
    return box_area(box) < target_area * REFEREE_AREA_RATIO

# ─── CSV 헤더 ─────────────────────────────────────────────────────────────────
header = ["frame", "track_id", "id_switch_count"]
for name in KP_NAMES:
    header.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])

# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    global TARGET_ID, last_known_center, last_known_area, click_point

    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[오류] 영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"영상: {VIDEO_PATH}  /  총 {total_frames}프레임  /  {fps:.1f}fps")
    print("▶ 화면에서 추적할 선수를 클릭하세요.")

    cv2.namedWindow("Match Data Extractor")
    cv2.setMouseCallback("Match Data Extractor", on_mouse)

    frame_count   = 0
    switch_count  = 0
    rows          = []
    selecting     = True  # 타겟 선택 모드

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # BoT-SORT로 추적
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

        boxes_raw = results[0].boxes
        if boxes_raw is None or boxes_raw.id is None:
            cv2.imshow("Match Data Extractor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        boxes     = boxes_raw.xyxy.cpu().numpy()
        track_ids = boxes_raw.id.int().cpu().numpy()
        kps_all   = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

        # ── 타겟 선택 모드 ─────────────────────────────────────────────────
        if selecting or TARGET_ID is None:
            vis = frame.copy()
            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (100, 200, 255), 2)
                cv2.putText(vis, f"ID:{tid}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

            cv2.putText(vis, "클릭으로 선수 선택  |  R: 재선택  |  Q: 종료",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
            cv2.imshow("Match Data Extractor", vis)

            if click_point is not None:
                for box, tid in zip(boxes, track_ids):
                    if point_in_box(click_point, box):
                        TARGET_ID        = tid
                        last_known_center = box_center(box)
                        last_known_area   = box_area(box)
                        switch_count      = 0
                        selecting         = False
                        click_point       = None
                        print(f"[선택] 타겟 ID = {TARGET_ID}")
                        break
                else:
                    click_point = None

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        # ── Anti-ID Switching ──────────────────────────────────────────────
        current_ids = track_ids.tolist()

        if TARGET_ID not in current_ids and last_known_center is not None:
            best_id, best_dist = None, float('inf')
            for box, tid in zip(boxes, track_ids):
                if is_referee(box, last_known_area):
                    continue
                d = math.hypot(*(a - b for a, b in zip(box_center(box), last_known_center)))
                if d < best_dist:
                    best_dist, best_id = d, tid

            if best_id is not None and best_dist < RECONNECT_DIST:
                switch_count += 1
                print(f"[프레임 {frame_count}] ID 스위칭 감지! {TARGET_ID} → {best_id}  "
                      f"(거리 {best_dist:.0f}px, 누적 {switch_count}회)")
                TARGET_ID = best_id

        # ── 데이터 추출 ───────────────────────────────────────────────────
        vis = frame.copy()
        found = False

        for box, tid, kp in zip(boxes, track_ids, kps_all):
            x1, y1, x2, y2 = map(int, box)

            if tid == TARGET_ID:
                found = True
                last_known_center = box_center(box)
                last_known_area   = box_area(box)

                # CSV 행 구성
                row = [frame_count, int(tid), switch_count]
                for pt in kp:   # pt = [x, y, conf]
                    row.extend([float(pt[0]), float(pt[1]), float(pt[2])])
                rows.append(row)

                # 시각화
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 80), 3)
                label = f"TARGET ID:{tid}  switches:{switch_count}"
                cv2.putText(vis, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)
                for pt in kp:
                    px, py, conf = int(pt[0]), int(pt[1]), float(pt[2])
                    if conf > 0.3:
                        cv2.circle(vis, (px, py), 4, (0, 230, 80), -1)
            else:
                col = (60, 60, 60) if is_referee(box, last_known_area) else (150, 150, 150)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 1)
                cv2.putText(vis, f"ID:{tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # 진행 표시
        pct = frame_count / max(total_frames, 1) * 100
        bar_w = int(frame.shape[1] * 0.4)
        bar_fill = int(bar_w * frame_count / max(total_frames, 1))
        bx, by = 10, frame.shape[0] - 30
        cv2.rectangle(vis, (bx, by), (bx + bar_w, by + 14), (40, 40, 40), -1)
        cv2.rectangle(vis, (bx, by), (bx + bar_fill, by + 14), (0, 200, 80), -1)
        cv2.putText(vis, f"Frame {frame_count}/{total_frames}  ({pct:.1f}%)  "
                        f"Rows:{len(rows)}  R: 재선택",
                    (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Match Data Extractor", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            selecting  = True
            TARGET_ID  = None
            click_point = None
            print("[재선택 모드]")

    cap.release()
    cv2.destroyAllWindows()

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n완료! 총 {len(rows)}프레임 저장 → {CSV_PATH}")
    print(f"ID 스위칭 발생 횟수: {switch_count}회")

if __name__ == "__main__":
    main()
