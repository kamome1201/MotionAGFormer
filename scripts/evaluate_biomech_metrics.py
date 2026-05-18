import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Human3.6M / MotionAGFormer の 17-joint 用 limb 定義
LIMBS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

LIMB_NAMES = [
    "root_rhip", "rhip_rknee", "rknee_rankle",
    "root_lhip", "lhip_lknee", "lknee_lankle",
    "root_spine", "spine_thorax", "thorax_neck", "neck_head",
    "thorax_lshoulder", "lshoulder_lelbow", "lelbow_lwrist",
    "thorax_rshoulder", "rshoulder_relbow", "relbow_rwrist",
]

# 左右対応骨の index ペア
SYM_PAIRS = [
    (0, 3),
    (1, 4),
    (2, 5),
    (10, 13),
    (11, 14),
    (12, 15),
]


def load_array(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[-1] != 3 or arr.shape[2] != 17:
        raise ValueError(
            f"{path} has shape {arr.shape}, expected (N, T, 17, 3)"
        )
    return arr


def get_limb_lengths(x: np.ndarray) -> np.ndarray:
    """
    x: (N, T, 17, 3)
    return: (N, T, 16)
    """
    out = []
    for i, j in LIMBS:
        d = x[:, :, i, :] - x[:, :, j, :]
        l = np.linalg.norm(d, axis=-1)
        out.append(l)
    return np.stack(out, axis=-1)


def summarize_limb_lengths(lengths: np.ndarray) -> pd.DataFrame:
    """
    lengths: (N, T, 16)
    """
    rows = []
    eps = 1e-8

    # 全サンプル全時刻でまとめた統計
    flat = lengths.reshape(-1, lengths.shape[-1])  # (N*T, 16)

    for idx, name in enumerate(LIMB_NAMES):
        vals = flat[:, idx]
        mean_ = float(np.mean(vals))
        std_ = float(np.std(vals))
        cv_ = float(std_ / (mean_ + eps))
        min_ = float(np.min(vals))
        max_ = float(np.max(vals))

        rows.append({
            "limb_idx": idx,
            "limb_name": name,
            "mean": mean_,
            "std": std_,
            "cv": cv_,
            "min": min_,
            "max": max_,
        })

    return pd.DataFrame(rows)


def summarize_temporal_variation(lengths: np.ndarray) -> pd.DataFrame:
    """
    各サンプルごとに時間方向 std を出し、その平均を limb ごとに集計
    lengths: (N, T, 16)
    """
    eps = 1e-8
    temporal_std = np.std(lengths, axis=1)   # (N, 16)
    temporal_mean = np.mean(lengths, axis=1) # (N, 16)
    temporal_cv = temporal_std / (temporal_mean + eps)

    rows = []
    for idx, name in enumerate(LIMB_NAMES):
        rows.append({
            "limb_idx": idx,
            "limb_name": name,
            "temporal_std_mean": float(np.mean(temporal_std[:, idx])),
            "temporal_std_std": float(np.std(temporal_std[:, idx])),
            "temporal_cv_mean": float(np.mean(temporal_cv[:, idx])),
            "temporal_cv_std": float(np.std(temporal_cv[:, idx])),
        })
    return pd.DataFrame(rows)


def summarize_symmetry(lengths: np.ndarray) -> pd.DataFrame:
    """
    左右骨長差 |L-R| を集計
    lengths: (N, T, 16)
    """
    rows = []
    for a, b in SYM_PAIRS:
        diff = np.abs(lengths[:, :, a] - lengths[:, :, b]).reshape(-1)
        rows.append({
            "left_limb_idx": a,
            "left_limb_name": LIMB_NAMES[a],
            "right_limb_idx": b,
            "right_limb_name": LIMB_NAMES[b],
            "abs_diff_mean": float(np.mean(diff)),
            "abs_diff_std": float(np.std(diff)),
            "abs_diff_median": float(np.median(diff)),
            "abs_diff_max": float(np.max(diff)),
        })
    return pd.DataFrame(rows)


def compare_pred_gt(pred_lengths: np.ndarray, gt_lengths: np.ndarray) -> pd.DataFrame:
    """
    骨長そのものの誤差を limb ごとに集計
    """
    if pred_lengths.shape != gt_lengths.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_lengths.shape}, gt {gt_lengths.shape}"
        )

    diff = np.abs(pred_lengths - gt_lengths)  # (N, T, 16)
    flat = diff.reshape(-1, diff.shape[-1])

    rows = []
    for idx, name in enumerate(LIMB_NAMES):
        vals = flat[:, idx]
        rows.append({
            "limb_idx": idx,
            "limb_name": name,
            "bone_abs_error_mean": float(np.mean(vals)),
            "bone_abs_error_std": float(np.std(vals)),
            "bone_abs_error_median": float(np.median(vals)),
            "bone_abs_error_max": float(np.max(vals)),
        })
    return pd.DataFrame(rows)


def global_summary(lengths: np.ndarray) -> dict:
    eps = 1e-8
    temporal_std = np.std(lengths, axis=1)   # (N, 16)
    temporal_mean = np.mean(lengths, axis=1) # (N, 16)
    temporal_cv = temporal_std / (temporal_mean + eps)

    sym_diffs = []
    for a, b in SYM_PAIRS:
        sym_diffs.append(np.abs(lengths[:, :, a] - lengths[:, :, b]))
    sym_diffs = np.stack(sym_diffs, axis=-1)  # (N, T, 6)

    return {
        "global_temporal_std_mean": float(np.mean(temporal_std)),
        "global_temporal_cv_mean": float(np.mean(temporal_cv)),
        "global_sym_abs_diff_mean": float(np.mean(sym_diffs)),
        "global_sym_abs_diff_std": float(np.std(sym_diffs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to pred.npy, shape (N,T,17,3)")
    parser.add_argument("--gt", default=None, help="Optional path to gt.npy, shape (N,T,17,3)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--tag", default="run", help="Tag name for output files")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred = load_array(args.pred)
    pred_lengths = get_limb_lengths(pred)

    pred_overall = summarize_limb_lengths(pred_lengths)
    pred_temporal = summarize_temporal_variation(pred_lengths)
    pred_sym = summarize_symmetry(pred_lengths)
    pred_global = global_summary(pred_lengths)

    pred_overall.to_csv(outdir / f"{args.tag}_pred_limb_overall.csv", index=False)
    pred_temporal.to_csv(outdir / f"{args.tag}_pred_limb_temporal.csv", index=False)
    pred_sym.to_csv(outdir / f"{args.tag}_pred_symmetry.csv", index=False)

    summary = {
        "tag": args.tag,
        **pred_global,
    }

    if args.gt is not None:
        gt = load_array(args.gt)
        if gt.shape != pred.shape:
            raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")

        gt_lengths = get_limb_lengths(gt)

        gt_overall = summarize_limb_lengths(gt_lengths)
        gt_temporal = summarize_temporal_variation(gt_lengths)
        gt_sym = summarize_symmetry(gt_lengths)
        bone_err = compare_pred_gt(pred_lengths, gt_lengths)
        gt_global = global_summary(gt_lengths)

        gt_overall.to_csv(outdir / f"{args.tag}_gt_limb_overall.csv", index=False)
        gt_temporal.to_csv(outdir / f"{args.tag}_gt_limb_temporal.csv", index=False)
        gt_sym.to_csv(outdir / f"{args.tag}_gt_symmetry.csv", index=False)
        bone_err.to_csv(outdir / f"{args.tag}_bone_error.csv", index=False)

        summary.update({
            "gt_global_temporal_std_mean": gt_global["global_temporal_std_mean"],
            "gt_global_temporal_cv_mean": gt_global["global_temporal_cv_mean"],
            "gt_global_sym_abs_diff_mean": gt_global["global_sym_abs_diff_mean"],
            "gt_global_sym_abs_diff_std": gt_global["global_sym_abs_diff_std"],
            "global_bone_abs_error_mean": float(np.mean(np.abs(pred_lengths - gt_lengths))),
            "global_bone_abs_error_std": float(np.std(np.abs(pred_lengths - gt_lengths))),
        })

    with open(outdir / f"{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] saved:")
    print(outdir / f"{args.tag}_pred_limb_overall.csv")
    print(outdir / f"{args.tag}_pred_limb_temporal.csv")
    print(outdir / f"{args.tag}_pred_symmetry.csv")
    if args.gt is not None:
        print(outdir / f"{args.tag}_bone_error.csv")
    print(outdir / f"{args.tag}_summary.json")

    print("\n[INFO] summary")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()