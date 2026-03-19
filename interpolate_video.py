import subprocess
import os

FFMPEG = r"C:\Users\parkh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

# ── 변환할 파일 목록 ──────────────────────────────────────────────
VIDEOS = [
    "bivol.mp4",
    "bivol2.mp4",
    "bivol3.mp4",
    "bivol4.mp4",
]
# ─────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

for input_file in VIDEOS:
    input_path = os.path.join(BASE_DIR, input_file)

    if not os.path.exists(input_path):
        print(f"[건너뜀] 파일 없음: {input_file}")
        continue

    name, ext = os.path.splitext(input_file)
    output_file = f"{name}_60fps{ext}"
    output_path = os.path.join(BASE_DIR, output_file)

    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"\n변환 중: {input_file} → {output_file}")

    cmd = [
        FFMPEG,
        "-i", input_path,
        "-vf", "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        output_path
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"완료! {output_file}  ({size_mb:.1f} MB)")
    else:
        print(f"[오류] {input_file} 변환 실패")

print("\n모든 변환 완료!")
