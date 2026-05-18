import re
import os
import pandas as pd

runs = [
    "h36m243_light_base",
    "h36m243_light_lc001",
    "h36m243_light_ls001",
    "h36m243_light_lc001_ls001",
]

base_dir = "checkpoint"

mpjpe_pattern = r"Protocol #1 Error \(MPJPE\): ([0-9\.]+)"
pmpjpe_pattern = r"Protocol #2 Error \(P-MPJPE\): ([0-9\.]+)"

rows = []

for run in runs:
    log_path = os.path.join(base_dir, run, "log.txt")

    if not os.path.exists(log_path):
        print(f"[WARN] {log_path} not found")
        continue

    with open(log_path, "r") as f:
        text = f.read()

    mpjpe = re.findall(mpjpe_pattern, text)
    pmpjpe = re.findall(pmpjpe_pattern, text)

    mpjpe = float(mpjpe[-1]) if mpjpe else None
    pmpjpe = float(pmpjpe[-1]) if pmpjpe else None

    rows.append({
        "run": run,
        "MPJPE": mpjpe,
        "P-MPJPE": pmpjpe
    })

df = pd.DataFrame(rows)
print(df)

df.to_csv("mpjpe_summary.csv", index=False)