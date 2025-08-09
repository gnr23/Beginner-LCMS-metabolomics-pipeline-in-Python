import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# 1. Load dataset
data = pd.read_csv("demo_metabolomics_data.csv", index_col=0)

# Define groups
control_cols = [c for c in data.columns if c.startswith("Control")]
treatment_cols = [c for c in data.columns if c.startswith("Treatment")]

# 2. Preprocessing
# Drop features with >20% missing (none here, but included for completeness)
data = data.dropna(thresh=int(0.8 * data.shape[1]))

# Fill remaining missing with median
data = data.fillna(data.median())

# Normalize total intensity per sample
data_norm = data.div(data.sum(axis=0), axis=1) * 1e6

# Scale features (mean=0, std=1)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_norm.T)  # samples as rows

# 3. PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(data_scaled)

plt.figure(figsize=(6,4))
plt.scatter(pcs[:len(control_cols), 0], pcs[:len(control_cols), 1], label='Control')
plt.scatter(pcs[len(control_cols):, 0], pcs[len(control_cols):, 1], label='Treatment')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of LC-MS/MS metabolomics demo data')
plt.legend()
plt.tight_layout()
plt.savefig("PCA_plot.png")
plt.show()

# 4. Statistical test (t-test)
results = []
for metabolite in data.index:
    control_vals = data.loc[metabolite, control_cols]
    treatment_vals = data.loc[metabolite, treatment_cols]
    stat, pval = ttest_ind(control_vals, treatment_vals)
    results.append({"Metabolite": metabolite, "p-value": pval})

results_df = pd.DataFrame(results)
results_df['adj_pval'] = results_df['p-value'] * len(results_df)  # Bonferroni correction
results_df.to_csv("ttest_results.csv", index=False)

print("Pipeline complete. Outputs: PCA_plot.png, ttest_results.csv")
