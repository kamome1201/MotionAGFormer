import json
from pathlib import Path
import pandas as pd

files = [
    "biomech_eval/base/h36m243_light_base_summary.json",
    "biomech_eval/lc001/h36m243_light_lc001_summary.json",
    "biomech_eval/ls001/h36m243_light_ls001_summary.json",
    "biomech_eval/lc001_ls001/h36m243_light_lc001_ls001_summary.json",
]

rows = []
for f in files:
    with open(f, "r") as fh:
        rows.append(json.load(fh))

df = pd.DataFrame(rows)

cols = [
    "tag",
    "global_temporal_std_mean",
    "global_temporal_cv_mean",
    "global_sym_abs_diff_mean",
    "global_sym_abs_diff_std",
    "global_bone_abs_error_mean",
    "global_bone_abs_error_std",
    "gt_global_temporal_std_mean",
    "gt_global_temporal_cv_mean",
    "gt_global_sym_abs_diff_mean",
    "gt_global_sym_abs_diff_std",
]

df = df[cols]
print(df)
df.to_csv("biomech_eval/biomech_summary_all.csv", index=False)