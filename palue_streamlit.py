import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from skbio import DistanceMatrix
from skbio.stats.distance import anosim
from scipy.stats import mannwhitneyu
import itertools

st.set_page_config(page_title="Diversity Analysis Tool", layout="centered")
st.title("Alpha & Beta Diversity P-Value Calculator")

analysis_type = st.selectbox("Select Analysis Type", ['Alpha Diversity (Shannon & Observed Features & Simpsom)', "Beta Diversity (PERMANOVA/ANOSIM)"])

if analysis_type == "Beta Diversity (PERMANOVA/ANOSIM)":
    sample_sheet = st.file_uploader("Upload Sample Sheet (.csv)", type=["csv"])
    braycurtis_file = st.file_uploader("Upload Bray-Curtis Distance Matrix (.tsv)", type=["tsv"])

    method = st.radio("Select Beta Diversity Method:", ["PERMANOVA", "ANOSIM"])

    with st.expander("⚙️ Advanced Settings"):
        sample_col = st.text_input("Sample Column Name", value="Sample")
        group_col = st.text_input("Group Column Name", value="Group")
        permutations = st.slider("Number of Permutations", min_value=100, max_value=9999, value=999, step=100)
        random_seed = st.number_input("Random Seed (Set 0 for no fixed seed)", value=42)

    if sample_sheet and braycurtis_file:
        if random_seed != 0:
            np.random.seed(int(random_seed))

        metadata_df = pd.read_csv(sample_sheet)
        metadata_df.columns = metadata_df.columns.str.strip()
        bc_df = pd.read_csv(braycurtis_file, sep="\t", index_col=0)

        samples = metadata_df[sample_col].values
        groups = metadata_df[group_col].values

        st.subheader("Detected Groups")
        group_counts = metadata_df[group_col].value_counts()
        for grp, count in group_counts.items():
            st.markdown(f"- **{grp}**: {count} samples")

        bc_filtered = bc_df.loc[samples, samples]

        if method == "PERMANOVA":
            dist_array = squareform(bc_filtered.values)

            def get_group_dists(labels, dists):
                n = len(labels)
                within, between = [], []
                k = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if labels[i] == labels[j]:
                            within.append(dists[k])
                        else:
                            between.append(dists[k])
                        k += 1
                return np.array(within), np.array(between)

            obs_w, obs_b = get_group_dists(groups, dist_array)
            obs_F = (np.mean(obs_b) - np.mean(obs_w)) / np.mean(obs_w)

            perm_F = []
            for _ in range(permutations):
                perm = np.random.permutation(groups)
                w, b = get_group_dists(perm, dist_array)
                if len(w) == 0 or len(b) == 0:
                    continue
                perm_F.append((np.mean(b) - np.mean(w)) / np.mean(w))

            p_value = np.mean(np.array(perm_F) >= obs_F)
            st.success(f"PERMANOVA pseudo-F = {obs_F:.4f}, p-value = {p_value:.4f}")

        elif method == "ANOSIM":
            bc_dm = DistanceMatrix(bc_filtered.values, ids=samples)
            result = anosim(bc_dm, grouping=groups, permutations=permutations)
            st.success(f"ANOSIM R = {result['test statistic']:.4f}, p-value = {result['p-value']:.4f}")

elif analysis_type == "Alpha Diversity (Shannon & Observed Features)":
    alpha_file = st.file_uploader("Upload Alpha Diversity CSV", type="csv")

    if alpha_file:
        alpha_df = pd.read_csv(alpha_file)
        alpha_df.columns = alpha_df.columns.str.strip()

        group_col = "Group"
        st.subheader("Detected Groups")
        group_counts = alpha_df[group_col].value_counts()
        for grp, count in group_counts.items():
            st.markdown(f"- **{grp}**: {count} samples")

        group_values = alpha_df[group_col].unique()
        if len(group_values) >= 2:
            st.subheader("Pairwise Mann-Whitney U Test Results")
            for g1, g2 in itertools.combinations(group_values, 2):
                obs1 = alpha_df[alpha_df[group_col] == g1]["observed_features"]
                obs2 = alpha_df[alpha_df[group_col] == g2]["observed_features"]
                shan1 = alpha_df[alpha_df[group_col] == g1]["shannon_entropy"]
                shan2 = alpha_df[alpha_df[group_col] == g2]["shannon_entropy"]
                sip1 = alpha_df[alpha_df[group_col] == g1]["simpson"]
                sip2 = alpha_df[alpha_df[group_col] == g2]["simpson"]

                p_obs = mannwhitneyu(obs1, obs2).pvalue
                p_shan = mannwhitneyu(shan1, shan2).pvalue
                p_sip = mannwhitneyu(sip1, sip2).pvalue

                st.markdown(f"**{g1} vs {g2}**")
                st.markdown(f"- Observed Features p-value: `{p_obs:.4f}`")
                st.markdown(f"- Shannon Entropy p-value: `{p_shan:.4f}`")
                st.markdown(f'- Simpson p-value: `{p_sip:.4f}`')
        else:
            st.warning("At least 2 groups required for comparison.")
