import pandas as pd
import numpy as np
import random
from skbio import DistanceMatrix
from skbio.stats.distance import anosim

# 固定亂數種子（可重現性）
np.random.seed(42)
random.seed(42)

# 讀入距離矩陣
file_path = r"C:\Users\youmin\Desktop\Working\P-value\braycurtis.tsv"
dist_df = pd.read_csv(file_path, sep="\t", index_col=0)

# 分組邏輯：HE + HES = OLD；ST + NST = YOUNG
def age_group(sample_id):
    sid = sample_id.strip().upper()
    if "HE" in sid or "HES" in sid:
        return "OLD"
    elif "ST" in sid or "NST" in sid:
        return "YOUNG"
    else:
        return None

# 建立樣本對應
sample_ids = dist_df.index.tolist()
group_dict = {sid: age_group(sid) for sid in sample_ids}
selected_ids = [sid for sid in sample_ids if group_dict[sid] in ["OLD", "YOUNG"]]

# 過濾距離矩陣與群組資訊
filtered_df = dist_df.loc[selected_ids, selected_ids]
groups = [group_dict[sid] for sid in selected_ids]
dm = DistanceMatrix(filtered_df.values, ids=selected_ids)

# 執行 ANOSIM
result = anosim(dm, groups, permutations=999)

# 輸出結果
print("🔬 ANOSIM 結果：老年人 (HE+HES) vs 年輕人 (ST+NST)")
print(result)

# 匯出成 CSV
pd.DataFrame([{
    "Comparison": "OLD (HE+HES) vs YOUNG (ST+NST)",
    "R": round(result['test statistic'], 4),
    "p-value": round(result['p-value'], 6),
    "Permutations": 999,
    "Seed": 42
}]).to_csv("anosim_old_vs_young_fixed.csv", index=False)
