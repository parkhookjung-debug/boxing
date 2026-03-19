"""
Boxing Game 2 - AI 대전 복싱 게임
웹캠 포즈 감지 기반 실시간 복싱 게임
펀치로 AI를 공격하고, AI 공격을 피하거나 막아라!
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import random
from collections import deque
from PIL import ImageFont, ImageDraw, Image as PILImage

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

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (27, 31), (28, 30), (28, 32),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
]

# ══════════════════════════════════════════════════════════════════════════════
# 게임 설정
# ══════════════════════════════════════════════════════════════════════════════
ROUND_DURATION  = 180
REST_DURATION   = 30
MAX_ROUNDS      = 3
PLAYER_MAX_HP   = 100
AI_MAX_HP       = 100
HP_RECOVERY     = 10

PUNCH_DMG = {
    "원(잽)":     3,
    "투(크로스)": 5,
    "훅":         7,
    "어퍼컷":    10,
}

AI_DMG_FULL    = 10
AI_DMG_BLOCK   = 3

ATTACK_TYPES = {
    "왼쪽":   {"warn": "← 왼쪽 공격!", "defenses": ["SLIP_RIGHT", "DUCK", "BLOCK"]},
    "오른쪽": {"warn": "오른쪽 공격! →", "defenses": ["SLIP_LEFT", "DUCK", "BLOCK"]},
    "바디":   {"warn": "↓ 바디 공격!",   "defenses": ["BLOCK"]},
}

# ── 펀치 감지 설정 ───────────────────────────────────────────────────────────
STANCE = "orthodox"
_JAB_IDX   = (16, 14, 12)
_CROSS_IDX = (15, 13, 11)
if STANCE == "southpaw":
    _JAB_IDX, _CROSS_IDX = _CROSS_IDX, _JAB_IDX

DYN_EXTEND_THRESH = 0.28
DYN_HOOK_THRESH   = 0.42
STRAIGHT_ANGLE  = 140
HOOK_ANGLE_MIN  = 50
HOOK_ANGLE_MAX  = 110
PUNCH_COOLDOWN  = 0.75
UPPERCUT_Y_THRESH  = 0.18
UPPERCUT_ANGLE_MIN = 50
UPPERCUT_ANGLE_MAX = 120

FRONT_SW  = 0.11
SIDE_SW   = 0.05
_view_mode = 0.0

_jab_hist        = deque(maxlen=8)
_cross_hist      = deque(maxlen=8)
_jab_dist_hist   = deque(maxlen=12)
_cross_dist_hist = deque(maxlen=12)
_last_pt    = {"jab": 0.0, "cross": 0.0}
_pstats     = {"원(잽)": 0, "투(크로스)": 0, "훅": 0, "어퍼컷": 0}
_dbg = {
    "j_ext": 0.0, "j_el_ang": 0.0, "j_hook_n": 0.0, "j_arm": 0.0,
    "c_ext": 0.0, "c_el_ang": 0.0, "c_hook_n": 0.0, "c_arm": 0.0,
}

_nose_hist = deque(maxlen=10)

# ══════════════════════════════════════════════════════════════════════════════
# 게임 상태
# ══════════════════════════════════════════════════════════════════════════════
_game_state   = "TITLE"
_round_num    = 0
_round_start  = 0.0
_rest_start   = 0.0
_countdown_start = 0.0

_player_hp    = PLAYER_MAX_HP
_ai_hp        = AI_MAX_HP

_ai_state     = "IDLE"
_ai_atk_type  = None
_ai_wind_start = 0.0
_ai_atk_start  = 0.0
_ai_last_atk   = 0.0
_ai_cooldown   = 4.0
_ai_windup_dur = 1.5

_combo_count  = 0
_combo_best   = 0
_last_hit_time = 0.0

_hit_flash     = 0.0
_dodge_flash   = 0.0
_punch_display = {"text": "", "col": (255, 255, 255), "until": 0.0}
_defense_display = {"text": "", "col": (255, 255, 255), "until": 0.0}

_rnd_punches_landed = 0
_rnd_dodges   = 0
_rnd_hits_taken = 0
_rnd_dmg_dealt = 0
_round_history = []

_ko_target  = None
_ko_time    = 0.0

# ── 한글 렌더링 ──────────────────────────────────────────────────────────────
def _load_font(size):
    for path in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc",
                 "C:/Windows/Fonts/batang.ttc"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

_KR_SM = _load_font(17)
_KR_MD = _load_font(26)
_KR_LG = _load_font(40)
_KR_XL = _load_font(60)
_KR_XXL = _load_font(80)

def put_kr(img, text, pos, color, font):
    pil  = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════════════════
def lm(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y])

def lm3(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y, p.z])

def dist(a, b):
    return np.linalg.norm(a - b)

def angle3pt(a, b, c):
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def draw_skeleton(image, landmarks):
    h, w = image.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(image, pts[a], pts[b], (180, 180, 180), 2)
    for pt in pts:
        cv2.circle(image, pt, 4, (255, 255, 255), -1)


# ══════════════════════════════════════════════════════════════════════════════
# 펀치 감지
# ══════════════════════════════════════════════════════════════════════════════
def detect_punch(lms, now):
    global _view_mode, _combo_count, _combo_best, _last_hit_time
    global _ai_hp, _rnd_punches_landed, _rnd_dmg_dealt
    jw, je, js = _JAB_IDX
    cw, ce, cs = _CROSS_IDX

    _jab_hist.append(lm3(lms, jw))
    _cross_hist.append(lm3(lms, cw))
    if len(_jab_hist) < 7:
        return

    nose  = lm(lms, 0)
    l_sh_ = lm(lms, 11); r_sh_ = lm(lms, 12)
    j_wr  = lm(lms, jw); j_el = lm(lms, je); j_sh = lm(lms, js)
    c_wr  = lm(lms, cw); c_el = lm(lms, ce); c_sh = lm(lms, cs)

    sw_raw = float(np.linalg.norm(l_sh_[:2] - r_sh_[:2]))
    if sw_raw >= FRONT_SW:      target = 0.0
    elif sw_raw <= SIDE_SW:     target = 1.0
    else:                       target = (FRONT_SW - sw_raw) / (FRONT_SW - SIDE_SW)
    _view_mode = _view_mode * 0.85 + target * 0.15
    side = _view_mode

    head_to_sh = abs(((l_sh_[1] + r_sh_[1]) / 2) - nose[1])
    side_ref   = max(head_to_sh * 0.55, 0.05)
    sw = max(sw_raw * (1 - side) + side_ref * side, 0.05)

    j_arm_len = float(dist(j_sh, j_wr)) / sw
    c_arm_len = float(dist(c_sh, c_wr)) / sw
    _jab_dist_hist.append(j_arm_len)
    _cross_dist_hist.append(c_arm_len)

    if len(_jab_dist_hist) < 7:
        return

    j_extend = max(
        float(_jab_dist_hist[-1] - _jab_dist_hist[-7]),
        float(_jab_dist_hist[-2] - _jab_dist_hist[-8]) if len(_jab_dist_hist) >= 8 else 0,
        float(_jab_dist_hist[-3] - _jab_dist_hist[-9]) if len(_jab_dist_hist) >= 9 else 0,
    )
    c_extend = max(
        float(_cross_dist_hist[-1] - _cross_dist_hist[-7]),
        float(_cross_dist_hist[-2] - _cross_dist_hist[-8]) if len(_cross_dist_hist) >= 8 else 0,
        float(_cross_dist_hist[-3] - _cross_dist_hist[-9]) if len(_cross_dist_hist) >= 9 else 0,
    )

    j_xn = abs(float(_jab_hist[-1][0] - _jab_hist[-5][0])) / sw
    c_xn = abs(float(_cross_hist[-1][0] - _cross_hist[-5][0])) / sw
    j_yn = abs(float(_jab_hist[-1][1] - _jab_hist[-5][1])) / sw
    c_yn = abs(float(_cross_hist[-1][1] - _cross_hist[-5][1])) / sw
    j_hook_n = (1 - side) * j_xn + side * j_yn
    c_hook_n = (1 - side) * c_xn + side * c_yn

    j_el_ang = angle3pt(j_sh, j_el, j_wr)
    c_el_ang = angle3pt(c_sh, c_el, c_wr)

    _dbg.update({
        "j_ext": j_extend, "j_el_ang": j_el_ang,
        "j_hook_n": j_hook_n, "j_arm": j_arm_len,
        "c_ext": c_extend, "c_el_ang": c_el_ang,
        "c_hook_n": c_hook_n, "c_arm": c_arm_len,
    })

    def fire(key, ptype):
        global _ai_hp, _combo_count, _combo_best, _last_hit_time
        global _rnd_punches_landed, _rnd_dmg_dealt
        _pstats[ptype] += 1

        base_dmg = PUNCH_DMG.get(ptype, 3)
        combo_bonus = _combo_count // 5
        total_dmg = base_dmg + combo_bonus

        _ai_hp = max(0, _ai_hp - total_dmg)
        _rnd_punches_landed += 1
        _rnd_dmg_dealt += total_dmg

        if now - _last_hit_time < 2.0:
            _combo_count += 1
        else:
            _combo_count = 1
        _last_hit_time = now
        _combo_best = max(_combo_best, _combo_count)

        combo_txt = f" [{_combo_count} COMBO!]" if _combo_count >= 3 else ""
        _punch_display["text"]  = f"{ptype}! -{total_dmg}HP{combo_txt}"
        _punch_display["col"]   = (0, 255, 100)
        _punch_display["until"] = now + 1.2
        _last_pt[key] = now

    if now - _last_pt["jab"] > PUNCH_COOLDOWN:
        if j_extend > DYN_EXTEND_THRESH:
            if j_el_ang >= STRAIGHT_ANGLE:
                fire("jab", "원(잽)")
        elif j_hook_n > DYN_HOOK_THRESH:
            if HOOK_ANGLE_MIN <= j_el_ang <= HOOK_ANGLE_MAX:
                if abs(j_wr[1] - j_sh[1]) < 0.22:
                    fire("jab", "훅")
        else:
            j_yn_up = float(_jab_hist[-5][1] - _jab_hist[-1][1]) / sw
            if j_yn_up > UPPERCUT_Y_THRESH:
                if UPPERCUT_ANGLE_MIN <= j_el_ang <= UPPERCUT_ANGLE_MAX:
                    if float(_jab_hist[-5][1]) > nose[1]:
                        fire("jab", "어퍼컷")

    if now - _last_pt["cross"] > PUNCH_COOLDOWN:
        if c_extend > DYN_EXTEND_THRESH:
            if c_el_ang >= STRAIGHT_ANGLE:
                fire("cross", "투(크로스)")
        elif c_hook_n > DYN_HOOK_THRESH:
            if HOOK_ANGLE_MIN <= c_el_ang <= HOOK_ANGLE_MAX:
                if abs(c_wr[1] - c_sh[1]) < 0.22:
                    fire("cross", "훅")
        else:
            c_yn_up = float(_cross_hist[-5][1] - _cross_hist[-1][1]) / sw
            if c_yn_up > UPPERCUT_Y_THRESH:
                if UPPERCUT_ANGLE_MIN <= c_el_ang <= UPPERCUT_ANGLE_MAX:
                    if float(_cross_hist[-5][1]) > nose[1]:
                        fire("cross", "어퍼컷")


# ══════════════════════════════════════════════════════════════════════════════
# 방어 감지
# ══════════════════════════════════════════════════════════════════════════════
def detect_defense(lms, sw):
    nose = lm(lms, 0)
    l_wr = lm(lms, 15)
    r_wr = lm(lms, 16)
    _nose_hist.append(nose.copy())

    if len(_nose_hist) < 5:
        return None

    dx = float(nose[0] - _nose_hist[-5][0])
    dy = float(nose[1] - _nose_hist[-5][1])

    if l_wr[1] < nose[1] - 0.02 and r_wr[1] < nose[1] - 0.02:
        return "BLOCK"

    slip_thresh = sw * 0.4
    if dx < -slip_thresh:
        return "SLIP_LEFT"
    if dx > slip_thresh:
        return "SLIP_RIGHT"

    duck_thresh = sw * 0.35
    if dy > duck_thresh:
        return "DUCK"

    return None


# ══════════════════════════════════════════════════════════════════════════════
# AI 로직
# ══════════════════════════════════════════════════════════════════════════════
def update_ai(now, player_defense):
    global _ai_state, _ai_atk_type, _ai_wind_start, _ai_atk_start, _ai_last_atk
    global _player_hp, _hit_flash, _dodge_flash, _combo_count
    global _rnd_dodges, _rnd_hits_taken

    if _ai_state == "IDLE":
        if now - _ai_last_atk > _ai_cooldown:
            _ai_atk_type = random.choice(list(ATTACK_TYPES.keys()))
            _ai_state = "WIND_UP"
            _ai_wind_start = now

    elif _ai_state == "WIND_UP":
        if now - _ai_wind_start >= _ai_windup_dur:
            _ai_state = "ATTACKING"
            _ai_atk_start = now

    elif _ai_state == "ATTACKING":
        if now - _ai_atk_start >= 0.5:
            atk_info = ATTACK_TYPES[_ai_atk_type]
            if player_defense and player_defense in atk_info["defenses"]:
                if player_defense == "BLOCK":
                    _player_hp = max(0, _player_hp - AI_DMG_BLOCK)
                    _defense_display["text"]  = f"BLOCK! (-{AI_DMG_BLOCK}HP)"
                    _defense_display["col"]   = (0, 200, 255)
                    _defense_display["until"] = now + 1.0
                    _dodge_flash = now
                else:
                    _defense_display["text"]  = "PERFECT DODGE!"
                    _defense_display["col"]   = (0, 255, 100)
                    _defense_display["until"] = now + 1.0
                    _dodge_flash = now
                _rnd_dodges += 1
            else:
                _player_hp = max(0, _player_hp - AI_DMG_FULL)
                _defense_display["text"]  = f"HIT! (-{AI_DMG_FULL}HP)"
                _defense_display["col"]   = (0, 0, 255)
                _defense_display["until"] = now + 1.0
                _hit_flash = now
                _combo_count = 0
                _rnd_hits_taken += 1

            _ai_state = "IDLE"
            _ai_last_atk = now


def set_ai_difficulty(round_num):
    global _ai_cooldown, _ai_windup_dur
    if round_num <= 1:
        _ai_cooldown   = 4.0
        _ai_windup_dur = 1.5
    elif round_num == 2:
        _ai_cooldown   = 3.2
        _ai_windup_dur = 1.2
    else:
        _ai_cooldown   = max(2.0, 4.0 - round_num * 0.4)
        _ai_windup_dur = max(0.7, 1.5 - round_num * 0.15)


# ══════════════════════════════════════════════════════════════════════════════
# UI 그리기
# ══════════════════════════════════════════════════════════════════════════════
def draw_hp_bars(frame):
    h, w = frame.shape[:2]
    bar_w, bar_h = 200, 22
    px, py = 20, 15
    p_ratio = max(0, _player_hp / PLAYER_MAX_HP)
    p_col = (0, 255, 100) if p_ratio > 0.5 else (0, 200, 255) if p_ratio > 0.25 else (0, 0, 255)
    cv2.rectangle(frame, (px, py), (px + bar_w, py + bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (px, py), (px + int(bar_w * p_ratio), py + bar_h), p_col, -1)
    cv2.rectangle(frame, (px, py), (px + bar_w, py + bar_h), (200, 200, 200), 1)
    cv2.putText(frame, f"YOU  {_player_hp}/{PLAYER_MAX_HP}", (px, py - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    ax = w - 20 - bar_w
    a_ratio = max(0, _ai_hp / AI_MAX_HP)
    a_col = (0, 255, 100) if a_ratio > 0.5 else (0, 200, 255) if a_ratio > 0.25 else (0, 0, 255)
    cv2.rectangle(frame, (ax, py), (ax + bar_w, py + bar_h), (40, 40, 40), -1)
    fill_start = ax + bar_w - int(bar_w * a_ratio)
    cv2.rectangle(frame, (fill_start, py), (ax + bar_w, py + bar_h), a_col, -1)
    cv2.rectangle(frame, (ax, py), (ax + bar_w, py + bar_h), (200, 200, 200), 1)
    cv2.putText(frame, f"AI  {_ai_hp}/{AI_MAX_HP}", (ax + bar_w - 110, py - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_round_timer(frame, now):
    h, w = frame.shape[:2]
    remain = max(0, ROUND_DURATION - (now - _round_start))
    minutes = int(remain) // 60
    seconds = int(remain) % 60
    timer_col = (0, 0, 255) if remain <= 30 else (0, 165, 255) if remain <= 60 else (255, 255, 255)
    cv2.putText(frame, f"R{_round_num}", (w//2 - 20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(frame, f"{minutes}:{seconds:02d}", (w//2 - 35, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, timer_col, 2)


def draw_combo(frame):
    if _combo_count >= 2:
        combo_col = (0, 255, 255) if _combo_count >= 10 else (0, 255, 100) if _combo_count >= 5 else (0, 220, 255)
        cv2.putText(frame, f"{_combo_count} COMBO", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, combo_col, 2)


def draw_punch_stats(frame):
    h, w = frame.shape[:2]
    px, py = w - 160, 55
    ov = frame.copy()
    cv2.rectangle(ov, (px - 5, py - 5), (w - 5, py + 115), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "PUNCHES", (px, py + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    for i, (k, v) in enumerate(_pstats.items()):
        put_kr(frame, f"{k}: {v}", (px, py + 18 + i * 22), (0, 220, 255), _KR_SM)


def draw_ai_attack_warning(frame, now):
    h, w = frame.shape[:2]
    if _ai_state == "WIND_UP":
        elapsed = now - _ai_wind_start
        progress = min(elapsed / _ai_windup_dur, 1.0)
        warn_txt = ATTACK_TYPES[_ai_atk_type]["warn"]
        blink = int(time.time() * 8) % 2 == 0
        if blink:
            ov = frame.copy()
            cv2.rectangle(ov, (w//4, h//2 - 50), (w*3//4, h//2 + 50), (0, 0, 80), -1)
            cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
            put_kr(frame, warn_txt, (w//2 - 100, h//2 - 40), (0, 0, 255), _KR_LG)
        bar_x, bar_w_px = w//4, w//2
        bar_y = h//2 + 55
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_px, bar_y + 8), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w_px * progress), bar_y + 8), (0, 0, 255), -1)
        defenses = ATTACK_TYPES[_ai_atk_type]["defenses"]
        hints = []
        for d in defenses:
            if d == "SLIP_LEFT":    hints.append("← 슬립")
            elif d == "SLIP_RIGHT": hints.append("슬립 →")
            elif d == "DUCK":       hints.append("↓ 덕")
            elif d == "BLOCK":      hints.append("블록(양손↑)")
        put_kr(frame, " / ".join(hints), (w//2 - 120, h//2 + 70), (200, 200, 200), _KR_SM)
    elif _ai_state == "ATTACKING":
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 6)


def draw_hit_effects(frame, now):
    if now - _hit_flash < 0.3:
        alpha = 1.0 - (now - _hit_flash) / 0.3
        ov = np.zeros_like(frame)
        ov[:] = (0, 0, 200)
        cv2.addWeighted(ov, alpha * 0.3, frame, 1.0, 0, frame)
    if now - _dodge_flash < 0.3:
        alpha = 1.0 - (now - _dodge_flash) / 0.3
        ov = np.zeros_like(frame)
        ov[:] = (0, 200, 0)
        cv2.addWeighted(ov, alpha * 0.15, frame, 1.0, 0, frame)


def draw_feedback(frame, now):
    h, w = frame.shape[:2]
    if now < _punch_display["until"] and _punch_display["text"]:
        txt = _punch_display["text"]
        bbox = _KR_MD.getbbox(txt)
        tw = bbox[2] - bbox[0]
        put_kr(frame, txt, ((w - tw) // 2, h - 50), _punch_display["col"], _KR_MD)
    if now < _defense_display["until"] and _defense_display["text"]:
        txt = _defense_display["text"]
        bbox = _KR_MD.getbbox(txt)
        tw = bbox[2] - bbox[0]
        put_kr(frame, txt, ((w - tw) // 2, h//2 + 100), _defense_display["col"], _KR_MD)


def draw_title_screen(frame):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
    put_kr(frame, "BOXING", (w//2 - 120, h//2 - 140), (0, 200, 255), _KR_XXL)
    put_kr(frame, "GAME 2", (w//2 - 100, h//2 - 60), (255, 255, 255), _KR_XXL)
    put_kr(frame, f"{MAX_ROUNDS}R x {ROUND_DURATION//60}분", (w//2 - 60, h//2 + 30), (180, 180, 180), _KR_SM)
    put_kr(frame, "R키를 눌러 시작!", (w//2 - 100, h//2 + 70), (0, 255, 100), _KR_MD)
    rules = [
        "펀치(잽/크로스/훅/어퍼컷)로 AI를 공격!",
        "AI 경고가 뜨면 슬립/덕/블록으로 방어!",
        "AI HP를 0으로 만들면 KO 승리!",
    ]
    for i, txt in enumerate(rules):
        put_kr(frame, txt, (w//2 - 190, h//2 + 120 + i * 24), (140, 140, 140), _KR_SM)


def draw_countdown(frame, now):
    h, w = frame.shape[:2]
    elapsed = now - _countdown_start
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    if elapsed < 1.0:
        txt, col = "3", (0, 200, 255)
    elif elapsed < 2.0:
        txt, col = "2", (0, 255, 200)
    elif elapsed < 3.0:
        txt, col = "1", (0, 255, 100)
    else:
        txt, col = "FIGHT!", (0, 0, 255)
    put_kr(frame, txt, (w//2 - 60, h//2 - 50), col, _KR_XXL)
    put_kr(frame, f"ROUND {_round_num}", (w//2 - 80, h//2 + 40), (255, 255, 255), _KR_MD)


def draw_ko_screen(frame, now):
    h, w = frame.shape[:2]
    elapsed = now - _ko_time
    ov = frame.copy()
    alpha = min(elapsed / 2.0, 0.8)
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0, frame)
    if int(elapsed * 4) % 2 == 0:
        put_kr(frame, "K.O.!", (w//2 - 80, h//2 - 80), (0, 0, 255), _KR_XXL)
    if _ko_target == "ai":
        put_kr(frame, "YOU WIN!", (w//2 - 90, h//2 + 10), (0, 255, 100), _KR_LG)
    else:
        put_kr(frame, "YOU LOSE...", (w//2 - 110, h//2 + 10), (0, 100, 255), _KR_LG)
    if elapsed > 3.0:
        put_kr(frame, "R: 재도전  |  Q: 종료", (w//2 - 120, h//2 + 80), (180, 180, 180), _KR_SM)


def draw_rest_screen(frame, now):
    h, w = frame.shape[:2]
    remain = max(0, REST_DURATION - (now - _rest_start))
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    put_kr(frame, f"ROUND {_round_num} 종료", (w//2 - 110, 30), (0, 200, 255), _KR_LG)
    put_kr(frame, f"휴식 중... {int(remain)}초", (w//2 - 80, 80), (0, 255, 200), _KR_MD)
    if _round_history:
        st = _round_history[-1]
        sy = 130
        gap = 28
        put_kr(frame, "── 라운드 통계 ──", (w//2 - 90, sy), (255, 255, 255), _KR_SM); sy += gap
        put_kr(frame, f"펀치 적중: {st['landed']}회", (w//2 - 70, sy), (0, 220, 255), _KR_SM); sy += gap
        put_kr(frame, f"준 데미지: {st['dmg_dealt']}HP", (w//2 - 70, sy), (0, 255, 100), _KR_SM); sy += gap
        put_kr(frame, f"받은 피격: {st['hits_taken']}회", (w//2 - 70, sy), (0, 100, 255), _KR_SM); sy += gap
        put_kr(frame, f"방어 성공: {st['dodges']}회", (w//2 - 70, sy), (0, 200, 255), _KR_SM); sy += gap
        put_kr(frame, f"최고 콤보: {st['best_combo']}", (w//2 - 70, sy), (0, 255, 255), _KR_SM); sy += gap + 10
        put_kr(frame, f"HP: YOU {_player_hp}  vs  AI {_ai_hp}", (w//2 - 110, sy), (255, 255, 255), _KR_MD)
        sy += 50
        put_kr(frame, "R키: 다음 라운드  |  Q키: 종료", (w//2 - 140, sy), (140, 140, 140), _KR_SM)


def draw_decision_screen(frame):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    if _player_hp > _ai_hp:
        result_txt, result_col = "판정승! YOU WIN!", (0, 255, 100)
    elif _player_hp < _ai_hp:
        result_txt, result_col = "판정패... YOU LOSE", (0, 100, 255)
    else:
        result_txt, result_col = "무승부! DRAW", (0, 200, 255)
    put_kr(frame, "FINAL RESULT", (w//2 - 120, 40), (0, 200, 255), _KR_LG)
    put_kr(frame, result_txt, (w//2 - 140, 100), result_col, _KR_LG)
    put_kr(frame, f"최종 HP:  YOU {_player_hp}  vs  AI {_ai_hp}", (w//2 - 150, 160), (255, 255, 255), _KR_MD)
    sy = 220
    put_kr(frame, " R  | 적중 | 데미지 | 피격 | 방어 | 콤보", (60, sy), (200, 200, 200), _KR_SM)
    sy += 8
    cv2.line(frame, (60, sy), (w - 60, sy), (100, 100, 100), 1)
    sy += 18
    for st in _round_history:
        line = f"R{st['round']:2d} |  {st['landed']:3d}  |  {st['dmg_dealt']:4d}  |  {st['hits_taken']:2d}  |  {st['dodges']:2d}  |  {st['best_combo']:2d}"
        put_kr(frame, line, (60, sy), (0, 220, 255), _KR_SM)
        sy += 24
    sy += 20
    total_landed = sum(s["landed"] for s in _round_history)
    total_dodges = sum(s["dodges"] for s in _round_history)
    put_kr(frame, f"총 적중: {total_landed}  |  총 방어: {total_dodges}  |  최고 콤보: {_combo_best}",
           (60, sy), (0, 255, 100), _KR_SM)
    sy += 40
    put_kr(frame, "R: 재도전  |  Q: 종료", (w//2 - 110, sy), (140, 140, 140), _KR_SM)


# ══════════════════════════════════════════════════════════════════════════════
# 통계 관리
# ══════════════════════════════════════════════════════════════════════════════
def save_round_stats():
    _round_history.append({
        "round":      _round_num,
        "landed":     _rnd_punches_landed,
        "dmg_dealt":  _rnd_dmg_dealt,
        "hits_taken": _rnd_hits_taken,
        "dodges":     _rnd_dodges,
        "best_combo": _combo_best,
    })

def reset_round_stats():
    global _rnd_punches_landed, _rnd_dodges, _rnd_hits_taken, _rnd_dmg_dealt
    global _combo_count, _combo_best
    _rnd_punches_landed = 0
    _rnd_dodges = 0
    _rnd_hits_taken = 0
    _rnd_dmg_dealt = 0
    _combo_count = 0
    _combo_best = 0

def reset_game():
    global _player_hp, _ai_hp, _round_num, _ai_state, _ai_last_atk
    global _combo_count, _combo_best
    _player_hp = PLAYER_MAX_HP
    _ai_hp = AI_MAX_HP
    _round_num = 0
    _ai_state = "IDLE"
    _ai_last_atk = 0.0
    _combo_count = 0
    _combo_best = 0
    _round_history.clear()
    for k in _pstats:
        _pstats[k] = 0
    reset_round_stats()


# ══════════════════════════════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
frame_count = 0
print("BOXING GAME 2!")
print("R: 게임 시작  |  Q: 종료")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now   = time.time()
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms == 0:
            timestamp_ms = frame_count * 33

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # ── 상태 전환 ────────────────────────────────────────────────────────
        if _game_state == "COUNTDOWN":
            if now - _countdown_start >= 3.5:
                _game_state = "ROUND"
                _round_start = now
                _ai_last_atk = now

        elif _game_state == "ROUND":
            elapsed = now - _round_start
            if _player_hp <= 0:
                _ko_target = "player"
                _ko_time = now
                _game_state = "KO"
                save_round_stats()
            elif _ai_hp <= 0:
                _ko_target = "ai"
                _ko_time = now
                _game_state = "KO"
                save_round_stats()
            elif elapsed >= ROUND_DURATION:
                save_round_stats()
                if _round_num >= MAX_ROUNDS:
                    _game_state = "DECISION"
                else:
                    _game_state = "REST"
                    _rest_start = now
                    _player_hp = min(PLAYER_MAX_HP, _player_hp + HP_RECOVERY)
                    _ai_hp = min(AI_MAX_HP, _ai_hp + HP_RECOVERY)

        elif _game_state == "REST":
            if now - _rest_start >= REST_DURATION:
                _round_num += 1
                reset_round_stats()
                set_ai_difficulty(_round_num)
                _ai_state = "IDLE"
                _countdown_start = now
                _game_state = "COUNTDOWN"

        # ── 렌더링 ───────────────────────────────────────────────────────────
        if _game_state == "TITLE":
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                draw_skeleton(frame, results.pose_landmarks[0])
            draw_title_screen(frame)

        elif _game_state == "COUNTDOWN":
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                draw_skeleton(frame, results.pose_landmarks[0])
            draw_countdown(frame, now)

        elif _game_state == "ROUND":
            player_defense = None
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                lms = results.pose_landmarks[0]
                draw_skeleton(frame, lms)
                l_sh = lm(lms, 11); r_sh = lm(lms, 12)
                sw = max(float(np.linalg.norm(l_sh - r_sh)), 0.05)
                detect_punch(lms, now)
                player_defense = detect_defense(lms, sw)
            update_ai(now, player_defense)
            draw_hp_bars(frame)
            draw_round_timer(frame, now)
            draw_combo(frame)
            draw_punch_stats(frame)
            draw_ai_attack_warning(frame, now)
            draw_hit_effects(frame, now)
            draw_feedback(frame, now)

        elif _game_state == "REST":
            draw_rest_screen(frame, now)

        elif _game_state == "KO":
            draw_ko_screen(frame, now)

        elif _game_state == "DECISION":
            draw_decision_screen(frame)

        cv2.imshow('BOXING GAME 2', frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            if _game_state == "TITLE":
                reset_game()
                _round_num = 1
                set_ai_difficulty(1)
                _countdown_start = now
                _game_state = "COUNTDOWN"
            elif _game_state == "REST":
                _round_num += 1
                reset_round_stats()
                set_ai_difficulty(_round_num)
                _ai_state = "IDLE"
                _countdown_start = now
                _game_state = "COUNTDOWN"
            elif _game_state in ("KO", "DECISION"):
                reset_game()
                _round_num = 1
                set_ai_difficulty(1)
                _countdown_start = now
                _game_state = "COUNTDOWN"
        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()
