"""
label_tool.py — 수동 펀치 라벨링 도구
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용법:
  python label_tool.py LIM1.mp4
  python label_tool.py bivol.mp4 --out bivol_labels.csv
  python label_tool.py bivol.mp4 --speed 0.5

키 조작:
  Space        재생 / 일시정지
  ← / →        1프레임 이동 (또는 A / D)
  F / G         10프레임 이동 (또는 Shift+←/→)
  J            잽   라벨
  C            크로스 라벨
  H            훅   라벨
  U            어퍼컷 라벨
  Backspace    현재 프레임 ±8 내 가장 가까운 라벨 삭제
  S            저장
  Q / Esc      종료 (자동 저장)

출력 CSV:
  frame_number, punch_type
  → eval_from_csv.py 의 --gt_labels 옵션으로 GT로 사용 가능
"""

import cv2
import csv
import os
import sys
import argparse
from collections import OrderedDict
from PIL import ImageFont, ImageDraw, Image as PILImage
import numpy as np

# ─── 색상 (BGR) ───────────────────────────────────────────────────
PUNCH_BGR = {
    'jab':      (0,   220, 255),
    'cross':    (40,  160, 255),
    'hook':     (255,  50, 200),
    'uppercut': (50,  255, 100),
}
C_BG     = (18,  13,  11)
C_PANEL  = (28,  21,  19)
C_BORDER = (55,  48,  46)
C_TEXT   = (245, 244, 244)
C_DIM    = (120, 113, 113)
C_ACCENT = (87,   61, 255)
C_TL_BG  = (38,  33,  31)

# ─── 폰트 ─────────────────────────────────────────────────────────
def _font(size):
    for p in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc",
              "C:/Windows/Fonts/arialbd.ttf"]:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()

F_LG = _font(26); F_MD = _font(18); F_SM = _font(14)

# ─── PIL 헬퍼 ─────────────────────────────────────────────────────
def b2p(img): return PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def p2b(pil): return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
def rgb(bgr): return (bgr[2], bgr[1], bgr[0])

def put_text(img, text, pos, font, color):
    pil = b2p(img)
    ImageDraw.Draw(pil).text(pos, text, font=font, fill=rgb(color))
    img[:] = p2b(pil)

# ─── 레이아웃 ─────────────────────────────────────────────────────
WIN_W, WIN_H  = 1280, 760
VID_MAX_H     = 580
TL_Y          = VID_MAX_H + 6
TL_H          = 28
TL_X, TL_W   = 8, WIN_W - 16
LIST_Y        = TL_Y + TL_H + 8
HINT_Y        = WIN_H - 24

# ─── CSV I/O ──────────────────────────────────────────────────────
def load_labels(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            d[int(row['frame_number'])] = row['punch_type']
    return d

def save_labels(d, path):
    rows = sorted(d.items())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['frame_number', 'punch_type'])
        w.writeheader()
        w.writerows({'frame_number': fn, 'punch_type': pt} for fn, pt in rows)
    return len(rows)

def nearest_label(d, frame, win=8):
    best, best_d = None, win + 1
    for fn in d:
        dd = abs(fn - frame)
        if dd < best_d:
            best_d = dd; best = fn
    return best

# ─── 렌더링 ───────────────────────────────────────────────────────
def render(canvas, frame_img, labels, fidx, total, playing, speed, out_path, msg):
    canvas[:] = C_BG

    # 비디오 영역
    fh, fw = frame_img.shape[:2]
    scale  = min(WIN_W / fw, VID_MAX_H / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    ox     = (WIN_W - nw) // 2
    canvas[0:nh, ox:ox+nw] = cv2.resize(frame_img, (nw, nh))

    # 현재 라벨 오버레이 (화면 상단)
    if fidx in labels:
        pt  = labels[fidx]
        col = PUNCH_BGR[pt]
        ov  = canvas.copy()
        cv2.rectangle(ov, (ox, 0), (ox+nw, nh), col, 8)
        cv2.addWeighted(ov, 0.35, canvas, 0.65, 0, canvas)
        put_text(canvas, f"[ {pt.upper()} ]", (ox + nw//2 - 55, 8), F_LG, col)

    # 프레임 정보 바
    status = f"{'▶' if playing else '⏸'}  {fidx:04d} / {total-1:04d}   x{speed:.1f}   Labels: {len(labels)}"
    put_text(canvas, status, (10, VID_MAX_H - 26), F_MD, C_TEXT)

    # 메시지 (저장 등 피드백)
    if msg:
        put_text(canvas, msg, (WIN_W - 350, VID_MAX_H - 26), F_MD, C_ACCENT)

    # 타임라인
    cv2.rectangle(canvas, (TL_X, TL_Y), (TL_X+TL_W, TL_Y+TL_H), C_TL_BG, -1)
    cv2.rectangle(canvas, (TL_X, TL_Y), (TL_X+TL_W, TL_Y+TL_H), C_BORDER, 1)

    # 재생 진행 바
    prog_w = int(TL_W * fidx / max(total - 1, 1))
    cv2.rectangle(canvas, (TL_X, TL_Y), (TL_X+prog_w, TL_Y+TL_H), (45, 42, 40), -1)

    # 라벨 마커
    for fn, pt in labels.items():
        mx  = TL_X + int(TL_W * fn / max(total - 1, 1))
        col = PUNCH_BGR[pt]
        cv2.circle(canvas, (mx, TL_Y + TL_H//2), 5, col, -1)

    # 현재 위치 커서
    cx = TL_X + int(TL_W * fidx / max(total - 1, 1))
    cv2.line(canvas, (cx, TL_Y - 3), (cx, TL_Y + TL_H + 3), (255, 255, 255), 2)

    # 최근 라벨 목록 (왼쪽)
    recent = sorted(labels.items())[-12:]
    pil = b2p(canvas); d = ImageDraw.Draw(pil)
    d.text((10, LIST_Y), "LABELS", font=F_SM, fill=rgb(C_DIM))
    for i, (fn, pt) in enumerate(reversed(recent)):
        col = PUNCH_BGR[pt]
        marker = "▶ " if fn == fidx else "   "
        d.text((10, LIST_Y + 18 + i * 17), f"{marker}{fn:04d}  {pt}", font=F_SM, fill=rgb(col))

    # 펀치 카운트 (오른쪽)
    cx2 = WIN_W - 180
    d.text((cx2, LIST_Y), "COUNT", font=F_SM, fill=rgb(C_DIM))
    for i, pt in enumerate(['jab', 'cross', 'hook', 'uppercut']):
        cnt = sum(1 for v in labels.values() if v == pt)
        col = PUNCH_BGR[pt]
        d.text((cx2, LIST_Y + 18 + i * 17), f"{pt:10s} {cnt:3d}", font=F_SM, fill=rgb(col))

    # 파일명
    d.text((cx2, LIST_Y + 90), os.path.basename(out_path), font=F_SM, fill=rgb(C_DIM))

    # 키 힌트
    hints = [
        ("Space","재생/정지"), ("←→/AD","1프레임"), ("FG","10프레임"),
        ("J","잽"), ("C","크로스"), ("H","훅"), ("U","어퍼컷"),
        ("BS","삭제"), ("[/]","속도"), ("S","저장"), ("Q","종료"),
    ]
    hx = 10
    for key, label in hints:
        d.text((hx, HINT_Y), key, font=F_SM, fill=rgb(C_ACCENT))
        kw = int(d.textlength(key, font=F_SM))
        d.text((hx + kw + 2, HINT_Y), f":{label}  ", font=F_SM, fill=rgb(C_DIM))
        hx += kw + int(d.textlength(f":{label}  ", font=F_SM)) + 2

    canvas[:] = p2b(pil)


# ─── 타임라인 클릭 → 프레임 이동 ────────────────────────────────
_mouse_x = -1
_mouse_clicked = False

def _on_mouse(event, x, y, flags, param):
    global _mouse_x, _mouse_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if TL_Y <= y <= TL_Y + TL_H:
            _mouse_x = x
            _mouse_clicked = True


# ─── 메인 ─────────────────────────────────────────────────────────
def main():
    global _mouse_clicked, _mouse_x

    parser = argparse.ArgumentParser()
    parser.add_argument('video',          help='영상 파일 경로')
    parser.add_argument('--out',          default=None,  help='출력 CSV 경로')
    parser.add_argument('--speed',        default=1.0, type=float, help='재생 속도 (기본 1.0)')
    parser.add_argument('--fresh',        action='store_true',
                        help='?? ?? CSV? ???? ?? ? ???? ?? ??')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"파일 없음: {args.video}"); sys.exit(1)

    base     = os.path.splitext(args.video)[0]
    out_path = args.out or f"{base}_labels.csv"

    cap    = cv2.VideoCapture(args.video)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    labels = {} if args.fresh else load_labels(out_path)
    speed  = args.speed

    if args.fresh and os.path.exists(out_path):
        print(f"? ??? ??: ?? ??? ?? ??? ????? -> {out_path}")
    elif labels:
        print(f"?? ?? {len(labels)}? ??: {out_path}")
    print(f"영상: {args.video}  |  총 {total}프레임  |  {fps:.1f}fps")
    print("━" * 50)

    # 프레임 캐시 (최근 500프레임)
    cache: OrderedDict = OrderedDict()
    MAX_CACHE = 500

    def get_frame(idx):
        if idx in cache:
            return cache[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if not ret:
            return None
        if len(cache) >= MAX_CACHE:
            cache.popitem(last=False)
        cache[idx] = f.copy()
        return f

    canvas  = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    fidx    = 0
    playing = False
    msg     = ''
    msg_ttl = 0
    frame   = get_frame(0)

    cv2.namedWindow('Punch Labeler', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Punch Labeler', WIN_W, WIN_H)
    cv2.setMouseCallback('Punch Labeler', _on_mouse)

    while frame is not None:
        render(canvas, frame, labels, fidx, total, playing, speed, out_path,
               msg if msg_ttl > 0 else '')
        msg_ttl = max(0, msg_ttl - 1)
        cv2.imshow('Punch Labeler', canvas)

        delay = max(1, int(1000 / fps / speed)) if playing else 30
        key   = cv2.waitKey(delay)

        # 타임라인 클릭
        if _mouse_clicked:
            fidx    = int(((_mouse_x - TL_X) / TL_W) * (total - 1))
            fidx    = max(0, min(total - 1, fidx))
            frame   = get_frame(fidx)
            playing = False
            _mouse_clicked = False
            continue

        # 재생 자동 진행
        if playing:
            nxt = fidx + 1
            if nxt < total:
                fidx  = nxt
                frame = get_frame(fidx)
            else:
                playing = False
            if key == -1:
                continue

        if key == -1:
            continue

        k = key & 0xFF

        # ── 종료 ──────────────────────────────────────────────────
        if k in (ord('q'), 27):
            n = save_labels(labels, out_path)
            print(f"\n저장 완료: {n}개 → {out_path}")
            break

        # ── 재생/정지 ──────────────────────────────────────────────
        elif k == ord(' '):
            playing = not playing

        # ── 1프레임 이동 (← → 또는 A D) ───────────────────────────
        elif k in (ord('a'), 81) or key in (2424832,):  # left
            fidx  = max(0, fidx - 1)
            frame = get_frame(fidx); playing = False
        elif k in (ord('d'), 83) or key in (2555904,):  # right
            fidx  = min(total - 1, fidx + 1)
            frame = get_frame(fidx); playing = False

        # ── 10프레임 이동 (Q E) ─────────────────────────────────────
        elif k == ord('q') - 16:  # won't match (q used for quit)
            pass
        elif key == 2424832 + (1 << 16):  # Shift+Left (Windows)
            fidx  = max(0, fidx - 10)
            frame = get_frame(fidx); playing = False
        elif key == 2555904 + (1 << 16):  # Shift+Right (Windows)
            fidx  = min(total - 1, fidx + 10)
            frame = get_frame(fidx); playing = False
        elif k == ord('f'):   # F = 10프레임 뒤
            fidx  = max(0, fidx - 10)
            frame = get_frame(fidx); playing = False
        elif k == ord('g'):   # G = 10프레임 앞
            fidx  = min(total - 1, fidx + 10)
            frame = get_frame(fidx); playing = False

        # ── 라벨 입력 ─────────────────────────────────────────────
        elif k in (ord('j'), ord('c'), ord('h'), ord('u')):
            pt = {'j':'jab','c':'cross','h':'hook','u':'uppercut'}[chr(k)]
            labels[fidx] = pt
            msg = f"+ {pt} @ {fidx}"; msg_ttl = 60
            print(f"  [{fidx:04d}] {pt}")

        # ── 라벨 삭제 (Backspace) ──────────────────────────────────
        elif k == 8:
            nearest = nearest_label(labels, fidx)
            if nearest is not None:
                removed = labels.pop(nearest)
                msg = f"삭제: {removed} @ {nearest}"; msg_ttl = 60
                print(f"  [삭제] [{nearest:04d}] {removed}")
            else:
                msg = "근처 라벨 없음"; msg_ttl = 40

        # ── 저장 ──────────────────────────────────────────────────
        elif k == ord('s'):
            n   = save_labels(labels, out_path)
            msg = f"저장 완료 {n}개"; msg_ttl = 90
            print(f"  저장: {n}개 → {out_path}")

        # ── 속도 조절 ([ / ]) ──────────────────────────────────────
        elif k == ord('['):
            speed = max(0.25, speed - 0.25)
            msg = f"속도 x{speed:.2f}"; msg_ttl = 60
        elif k == ord(']'):
            speed = min(4.0, speed + 0.25)
            msg = f"속도 x{speed:.2f}"; msg_ttl = 60

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
