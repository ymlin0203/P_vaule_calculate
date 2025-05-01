import pandas as pd
import numpy as np
from skbio import DistanceMatrix
from skbio.stats.distance import anosim

# 固定種子（讓結果比較穩定）
np.random.seed(42)

# 讀取檔案
braycurtis_df = pd.read_csv("C:/Users/youmin/Desktop/Working/P-value/braycurtis.tsv", sep="\t", index_col=0)
metadata_df = pd.read_csv("C:/Users/youmin/Desktop/Working/P-value/25_0422_TRI_Non-TRI.csv")
metadata_df.columns = metadata_df.columns.str.strip()

# 檢查分組與樣本
print(metadata_df["Group"].value_counts())

# 提取樣本與分組資訊
samples = metadata_df["Sample"].values
grouping = metadata_df["Group"].values

# 對齊距離矩陣順序
filtered_df = braycurtis_df.loc[samples, samples]
bc_dm = DistanceMatrix(filtered_df.values, ids=samples)

# 執行 ANOSIM（你的版本不支援 random_state）
result = anosim(distance_matrix=bc_dm, grouping=grouping, permutations=999)
print(result)
