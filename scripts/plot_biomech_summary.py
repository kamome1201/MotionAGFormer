import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("biomech_eval/biomech_summary_all.csv")

labels = [
    "base",
    "lc001",
    "ls001",
    "lc001+ls001",
]

x = range(len(df))

metrics = [
    ("global_temporal_std_mean", "Temporal std"),
    ("global_temporal_cv_mean", "Temporal CV"),
    ("global_sym_abs_diff_mean", "Symmetry abs diff"),
    ("global_bone_abs_error_mean", "Bone abs error"),
]

for col, ylabel in metrics:
    plt.figure(figsize=(6, 4), dpi=300)
    plt.bar(labels, df[col].values)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.tight_layout()
    out = f"biomech_eval/{col}.png"
    plt.savefig(out)
    plt.close()
    print(f"saved: {out}")