
import streamlit as st
import pandas as pd
import numpy as np
from skbio import DistanceMatrix
from skbio.stats.distance import anosim

st.title("🔬 ANOSIM 分析工具：自動依 metadata 分組")

st.markdown("請依序上傳：")
uploaded_bc = st.file_uploader("📂 上傳 Bray-Curtis 距離矩陣（.tsv）", type=["tsv"])
uploaded_meta = st.file_uploader("📂 上傳 Metadata 表（需包含 'Sample' 與 'Group' 欄位）", type=["csv"])

if uploaded_bc is not None and uploaded_meta is not None:
    np.random.seed(42)

    # 讀入資料
    braycurtis_df = pd.read_csv(uploaded_bc, sep="\t", index_col=0)
    metadata_df = pd.read_csv(uploaded_meta)
    metadata_df.columns = metadata_df.columns.str.strip()

    st.subheader("✅ 分組資訊統計")
    st.write(metadata_df["Group"].value_counts())

    # 提取樣本與分組
    samples = metadata_df["Sample"].values
    grouping = metadata_df["Group"].values

    try:
        filtered_df = braycurtis_df.loc[samples, samples]
        bc_dm = DistanceMatrix(filtered_df.values, ids=samples)

        # 執行 ANOSIM
        result = anosim(distance_matrix=bc_dm, grouping=grouping, permutations=999)

        st.subheader("📊 ANOSIM 結果")
        st.write(result)

        # 匯出結果
        result_df = pd.DataFrame([{
            "Comparison": "Auto-grouped from Metadata",
            "R": round(result['test statistic'], 4),
            "p-value": round(result['p-value'], 6),
            "Permutations": 999,
            "Seed": 42
        }])
        st.download_button("📥 下載結果 CSV", result_df.to_csv(index=False), file_name="anosim_metadata_grouping.csv")

    except Exception as e:
        st.error(f"資料處理時發生錯誤：{e}")
else:
    st.info("請同時上傳距離矩陣與 metadata 檔案以執行分析")
