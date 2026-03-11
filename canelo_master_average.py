import pandas as pd
import glob
import os

# 1. 카넬로 CSV 파일 전부 불러오기
csv_files = sorted(glob.glob('canelo_full_data*.csv'))
print(f"발견된 파일 ({len(csv_files)}개):")
for f in csv_files:
    df_tmp = pd.read_csv(f)
    print(f"  {f}  →  {len(df_tmp)} 프레임")

# 2. 전부 합치기 (frame_number 열 제외)
all_dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    df = df.drop(columns=['frame_number'])
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
print(f"\n합산 총 프레임: {len(combined)}")

# 3. 각 관절별 평균(Mean)과 표준편차(Std) 계산
mean = combined.mean()
std  = combined.std()

profile = pd.DataFrame({
    'Average (Mean)': mean,
    'Tolerance (Std)': std
})

# 4. 저장
OUTPUT = 'canelo_master_average_profile.csv'
profile.to_csv(OUTPUT)
print(f"\n저장 완료: {OUTPUT}")

# 5. 핵심 수치 출력 (코치 임계값 설정에 필요한 값들)
sw = abs(combined['left_shoulder_x'].mean() - combined['right_shoulder_x'].mean())
nose_y   = combined['nose_y'].mean()
l_wr_y   = combined['left_wrist_y'].mean()
r_wr_y   = combined['right_wrist_y'].mean()
l_ank_x  = combined['left_ankle_x'].mean()
r_ank_x  = combined['right_ankle_x'].mean()
l_sh_y   = combined['left_shoulder_y'].mean()
r_sh_y   = combined['right_shoulder_y'].mean()
l_sh_x   = combined['left_shoulder_x'].mean()
r_sh_x   = combined['right_shoulder_x'].mean()

stance_raw = abs(l_ank_x - r_ank_x)
guard_l_raw = l_wr_y - nose_y
guard_r_raw = r_wr_y - nose_y
head_h_raw  = ((l_sh_y + r_sh_y) / 2) - nose_y

print("\n━━━ 카넬로 핵심 정규화 수치 (어깨너비 기준) ━━━")
print(f"  어깨너비 (raw):          {sw:.4f}")
print(f"  가드 비율 (왼손):         {guard_l_raw / sw:.3f}  (raw {guard_l_raw:.4f})")
print(f"  가드 비율 (오른손):        {guard_r_raw / sw:.3f}  (raw {guard_r_raw:.4f})")
print(f"  스탠스 비율:              {stance_raw / sw:.3f}  (raw {stance_raw:.4f})")
print(f"  헤드 높이 비율:            {head_h_raw / sw:.3f}  (raw {head_h_raw:.4f})")
print()

# wrist y std도 출력
lw_std = combined['left_wrist_y'].std()
rw_std = combined['right_wrist_y'].std()
print(f"  가드 Std (왼손 정규화):   {lw_std / sw:.3f}")
print(f"  가드 Std (오른손 정규화): {rw_std / sw:.3f}")
