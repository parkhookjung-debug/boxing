import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import random
import os

# ── 모델 파일 확인 ────────────────────────────────────────────────────────────
MODEL_PATH = 'pose_landmarker_full.task'
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("포즈 모델 다운로드 중 (약 30MB)...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_full/float16/latest/pose_landmarker_full.task',
        MODEL_PATH
    )
    print("다운로드 완료!")

# ── MediaPipe Tasks API 초기화 ────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── 랜드마크 인덱스 ───────────────────────────────────────────────────────────
NOSE        = 0
L_EYE       = 2
R_EYE       = 5
L_SHOULDER  = 11
R_SHOULDER  = 12
L_ELBOW     = 13
R_ELBOW     = 14
L_WRIST     = 15
R_WRIST     = 16
L_INDEX     = 19   # 왼손 검지 끝
R_INDEX     = 20   # 오른손 검지 끝
L_HIP       = 23
R_HIP       = 24

# 뼈대 연결선 (상체만)
SKELETON = [
    (NOSE, L_EYE), (NOSE, R_EYE),
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_WRIST, L_INDEX),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_WRIST, R_INDEX),
    (L_SHOULDER, L_HIP),  (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
]

# ── 게임 상수 ─────────────────────────────────────────────────────────────────
FLASH_TIMEOUT = 2.0   # 판정 제한 시간(초)

# ── 게임 상태 ─────────────────────────────────────────────────────────────────
# MENU → INIT → COUNTDOWN → WAITING → FLASHING → COMBO_WAIT → FLASHING ...
state           = "MENU"
countdown_start = 0.0
flash_side      = None
flash_start     = 0.0
wait_until      = 0.0
records         = []
result_label    = ""
result_until    = 0.0
frame_count     = 0
max_combo        = 1   # 1=단타, 2=최대2연속, 3=최대3연속
combo_remaining  = 0   # 현재 콤보에서 남은 공격 수
combo_wait_until = 0.0
combo_buffer     = []  # 현재 콤보 회차별 결과 임시 저장


def get_xy(lms, idx):
    p = lms[idx]
    return p.x, p.y


def draw_skeleton(frame, lms, h, w):
    """풀 스켈레톤 + 주요 포인트 표시"""
    pts = []
    for p in lms:
        pts.append((int(p.x * w), int(p.y * h)))

    # 뼈대 선
    for a, b in SKELETON:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (160, 160, 160), 2)

    # 모든 관절 점
    for pt in pts:
        cv2.circle(frame, pt, 4, (220, 220, 220), -1)

    # 주요 포인트 강조
    # 코 (노란색)
    cv2.circle(frame, pts[NOSE], 10, (0, 230, 255), -1)
    # 눈 (하늘색)
    cv2.circle(frame, pts[L_EYE], 7, (255, 200, 0), -1)
    cv2.circle(frame, pts[R_EYE], 7, (255, 200, 0), -1)
    # 손목 (초록색)
    cv2.circle(frame, pts[L_WRIST], 8, (0, 200, 60), -1)
    cv2.circle(frame, pts[R_WRIST], 8, (0, 200, 60), -1)
    # 검지 끝 (밝은 초록, 크게) - 가드 판정 기준
    cv2.circle(frame, pts[L_INDEX], 14, (0, 255, 80), -1)
    cv2.circle(frame, pts[R_INDEX], 14, (0, 255, 80), -1)
    # 어깨 (파란색)
    cv2.circle(frame, pts[L_SHOULDER], 8, (255, 100, 0), -1)
    cv2.circle(frame, pts[R_SHOULDER], 8, (255, 100, 0), -1)


def both_hands_up(lms):
    """양쪽 손목이 각 어깨보다 위에 있으면 True"""
    _, lw_y = get_xy(lms, L_WRIST)
    _, ls_y = get_xy(lms, L_SHOULDER)
    _, rw_y = get_xy(lms, R_WRIST)
    _, rs_y = get_xy(lms, R_SHOULDER)
    return lw_y < ls_y and rw_y < rs_y


def check_dodge(lms, side):
    """
    코 또는 눈 중 하나라도 중앙선(0.5)을 넘으면 회피 성공
    LEFT 공격 → 코/눈이 중앙선 오른쪽 (x > 0.5)
    RIGHT 공격 → 코/눈이 중앙선 왼쪽 (x < 0.5)
    """
    nx, _ = get_xy(lms, NOSE)
    lx, _ = get_xy(lms, L_EYE)
    rx, _ = get_xy(lms, R_EYE)

    if side == "LEFT":
        return nx > 0.60 and lx > 0.60 and rx > 0.60
    else:
        return nx < 0.40 and lx < 0.40 and rx < 0.40


def check_block(lms, side):
    """
    검지 끝이 코에서 (어깨→코 거리의 0.4배) 만큼 위에 있어야 가드 성공
    현재보다 약 2배 더 높이 들어야 함
    """
    _, nose_y = get_xy(lms, NOSE)
    _, ls_y   = get_xy(lms, L_SHOULDER)
    _, rs_y   = get_xy(lms, R_SHOULDER)
    shoulder_y = (ls_y + rs_y) / 2
    # threshold = 코 위치에서 (어깨→코 거리 × 0.4) 만큼 더 위
    threshold = nose_y - (shoulder_y - nose_y) * 0.4
    # flip 후 처리하므로 MediaPipe L/R이 화면 기준으로 반전
    # 화면 LEFT 공격 → 사용자 왼손(화면 왼쪽) = MediaPipe R_INDEX
    if side == "LEFT":
        _, iy = get_xy(lms, R_INDEX)
    else:
        _, iy = get_xy(lms, L_INDEX)
    return iy < threshold


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("복싱 피하기 게임! 1/2/3 키로 콤보 선택 후 양손을 들어 시작. 종료: Q")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx  = w // 2
        now = time.time()

        # MediaPipe 추론
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if ts_ms == 0:
            ts_ms = frame_count * 33
        results  = landmarker.detect_for_video(mp_img, ts_ms)
        lms      = results.pose_landmarks[0] if results.pose_landmarks else None

        # 스켈레톤 항상 표시
        if lms:
            draw_skeleton(frame, lms, h, w)

        # 중앙선
        cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)

        # ── MENU: 1/2/3 콤보 선택 ───────────────────────────────────────────
        if state == "MENU":
            labels = {1: "1  Single", 2: "2  Max 2-Combo", 3: "3  Max 3-Combo"}
            cv2.putText(frame, "SELECT COMBO MODE",
                        (w // 2 - 200, h // 2 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            for i, (k, txt) in enumerate(labels.items()):
                col = (0, 255, 0) if k == max_combo else (180, 180, 180)
                cv2.putText(frame, f"[{k}] {txt}",
                            (w // 2 - 140, h // 2 - 20 + i * 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.95, col, 2)
            cv2.putText(frame, "Then raise both hands to start",
                        (w // 2 - 255, h // 2 + 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2)
            if lms and both_hands_up(lms):
                state           = "COUNTDOWN"
                countdown_start = now

        # ── INIT (더 이상 사용 안 함 - MENU로 대체) ──────────────────────────
        elif state == "INIT":
            pass

        # ── COUNTDOWN ────────────────────────────────────────────────────────
        elif state == "COUNTDOWN":
            remaining = 3 - int(now - countdown_start)
            if remaining > 0:
                cv2.putText(frame, str(remaining),
                            (w // 2 - 50, h // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 255, 0), 10)
            else:
                state      = "WAITING"
                wait_until = now + random.uniform(1.0, 5.0)

        # ── WAITING ──────────────────────────────────────────────────────────
        elif state == "WAITING":
            cv2.putText(frame, "READY...",
                        (w // 2 - 80, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
            if now > wait_until:
                combo_length    = random.randint(1, max_combo)
                combo_remaining = combo_length - 1
                combo_buffer    = []
                flash_side      = random.choice(["LEFT", "RIGHT"])
                flash_start     = now
                state           = "FLASHING"

        # ── COMBO_WAIT: 콤보 간 대기 ─────────────────────────────────────────
        elif state == "COMBO_WAIT":
            cv2.putText(frame, "COMBO!",
                        (w // 2 - 70, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            if now > combo_wait_until:
                flash_side  = random.choice(["LEFT", "RIGHT"])
                flash_start = now
                state       = "FLASHING"

        # ── FLASHING ─────────────────────────────────────────────────────────
        elif state == "FLASHING":
            elapsed = now - flash_start

            # 빨간 반투명 오버레이
            overlay = frame.copy()
            if flash_side == "LEFT":
                cv2.rectangle(overlay, (0, 0), (cx, h), (0, 0, 200), -1)
            else:
                cv2.rectangle(overlay, (cx, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            # 공격 방향 텍스트
            tx = cx // 2 - 70 if flash_side == "LEFT" else cx + cx // 2 - 85
            cv2.putText(frame, flash_side + "!",
                        (tx, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)

            # 타이머 바
            bar_len = int((elapsed / FLASH_TIMEOUT) * w)
            bar_col = ((0, 220, 0)    if elapsed < 1.0
                       else (0, 165, 255) if elapsed < 1.5
                       else (0, 0, 255))
            cv2.rectangle(frame, (0, h - 18), (bar_len, h), bar_col, -1)

            # 판정
            resolved   = False
            hit_result = None
            if lms:
                if check_dodge(lms, flash_side):
                    hit_result   = {"type": "DODGE", "label": f"DODGE {elapsed:.2f}s"}
                    result_label = f"DODGE  {elapsed:.2f}s"
                    resolved     = True
                elif check_block(lms, flash_side):
                    hit_result   = {"type": "BLOCK", "label": f"BLOCK {elapsed:.2f}s"}
                    result_label = f"BLOCK  {elapsed:.2f}s"
                    resolved     = True

            if not resolved and elapsed >= FLASH_TIMEOUT:
                hit_result   = {"type": "F", "label": "F"}
                result_label = "F"
                resolved     = True

            if resolved:
                combo_buffer.append(hit_result)
                result_until = now + 0.6
                if combo_remaining > 0:
                    combo_remaining  -= 1
                    state             = "COMBO_WAIT"
                    combo_wait_until  = now + random.uniform(0.5, 1.2)
                else:
                    # 콤보 종료 → 한 줄로 합쳐서 records에 저장
                    joined   = " / ".join(r["label"] for r in combo_buffer)
                    has_f    = any(r["type"] == "F" for r in combo_buffer)
                    all_good = all(r["type"] in ("DODGE", "BLOCK") for r in combo_buffer)
                    rec_type = "F" if has_f else ("DODGE" if all_good and combo_buffer[0]["type"] == "DODGE" else "BLOCK")
                    records.append({"type": rec_type, "label": joined})
                    state      = "WAITING"
                    wait_until = now + random.uniform(1.0, 5.0)

        # ── 결과 팝업 ────────────────────────────────────────────────────────
        if result_label and now < result_until:
            color = ((0, 255, 0)    if "DODGE" in result_label
                     else (0, 165, 255) if "BLOCK" in result_label
                     else (0, 0, 255))
            cv2.putText(frame, result_label,
                        (w // 2 - 200, h // 2 + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)

        # ── 기록 패널 (최근 7개) ──────────────────────────────────────────────
        for i, rec in enumerate(records[-7:]):
            has_f = "F" in rec["label"]
            color = ((0, 0, 255)    if has_f
                     else (0, 255, 0) if "DODGE" in rec["label"]
                     else (0, 165, 255))
            idx = len(records) - min(7, len(records)) + i + 1
            cv2.putText(frame, f"{idx:>2}. {rec['label']}",
                        (10, 32 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)

        # 콤보 선택 표시 (항상)
        combo_names = {1: "SINGLE", 2: "2-COMBO", 3: "3-COMBO"}
        cv2.putText(frame, f"MODE: {combo_names[max_combo]}",
                    (w - 200, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 0), 2)

        cv2.imshow("Boxing Dodge Game", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            max_combo = 1
            if state != "FLASHING" and state != "COMBO_WAIT":
                state = "MENU"
        elif key == ord("2"):
            max_combo = 2
            if state != "FLASHING" and state != "COMBO_WAIT":
                state = "MENU"
        elif key == ord("3"):
            max_combo = 3
            if state != "FLASHING" and state != "COMBO_WAIT":
                state = "MENU"

cap.release()
cv2.destroyAllWindows()
