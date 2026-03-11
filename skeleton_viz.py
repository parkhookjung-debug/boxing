import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# CSV 로드
df = pd.read_csv('bivol_full_data4.csv')

# 관절 이름 순서 (mediapipe 표준)
LANDMARKS = [
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

# 뼈대 연결선 정의 (이름 기준)
CONNECTIONS = [
    # 얼굴
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    # 몸통
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    # 왼팔
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('left_wrist', 'left_index'), ('left_wrist', 'left_pinky'), ('left_wrist', 'left_thumb'),
    # 오른팔
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('right_wrist', 'right_index'), ('right_wrist', 'right_pinky'), ('right_wrist', 'right_thumb'),
    # 왼다리
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('left_ankle', 'left_heel'), ('left_ankle', 'left_foot_index'),
    # 오른다리
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ('right_ankle', 'right_heel'), ('right_ankle', 'right_foot_index'),
]

# 파트별 색상
COLORS = {
    'face':   '#FFD700',
    'torso':  '#00BFFF',
    'left':   '#FF6B6B',
    'right':  '#98FB98',
    'foot':   '#DDA0DD',
}

def get_color(a, b):
    face_joints = {'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'}
    if a in face_joints or b in face_joints:
        return COLORS['face']
    if 'torso' in a or a in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'):
        return COLORS['torso']
    if 'left' in a:
        return COLORS['left']
    if 'right' in a:
        return COLORS['right']
    return COLORS['torso']

# 플롯 설정
fig, ax = plt.subplots(figsize=(6, 9), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)  # y축 반전 (이미지 좌표계)
ax.axis('off')
title = ax.set_title('', color='white', fontsize=12)

# 애니메이션 업데이트 함수
def update(frame_idx):
    ax.cla()
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.axis('off')

    row = df.iloc[frame_idx]
    frame_num = int(row['frame_number'])

    # 관절 좌표 추출
    coords = {}
    for name in LANDMARKS:
        x = row.get(f'{name}_x')
        y = row.get(f'{name}_y')
        v = row.get(f'{name}_v', 1.0)
        if pd.notna(x) and pd.notna(y):
            coords[name] = (x, y, v)

    # 연결선 그리기
    for a, b in CONNECTIONS:
        if a in coords and b in coords:
            x1, y1, v1 = coords[a]
            x2, y2, v2 = coords[b]
            alpha = min(v1, v2) * 0.9 + 0.1
            color = get_color(a, b)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.5, alpha=alpha)

    # 관절 점 그리기
    for name, (x, y, v) in coords.items():
        alpha = v * 0.9 + 0.1
        ax.scatter(x, y, s=25, color='white', alpha=alpha, zorder=5)

    ax.set_title(f'Frame {frame_num} / {int(df["frame_number"].max())}',
                 color='white', fontsize=12, pad=10)

ani = animation.FuncAnimation(
    fig, update,
    frames=len(df),
    interval=33,   # ~30fps
    repeat=True
)

plt.tight_layout()
plt.show()
