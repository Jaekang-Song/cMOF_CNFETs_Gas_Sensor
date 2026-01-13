import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
import os

# ============================================================
# Density-weighted jitter for dot plot (log-axis friendly)
# ============================================================
## This is to show distribution simialr to what violin plots do.

def density_weighted_jitter(y, max_width=0.28, bw="scott", seed=0):
    # y: 1D array
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(y)
    y2 = y[mask]
    if y2.size < 3:
        return np.zeros(y.size)

    kde = gaussian_kde(y2, bw_method=bw)
    d = kde(y2)
    d = d / d.max()

    jit = np.zeros(y.size)
    jit[mask] = (rng.random(mask.sum()) - 0.5) * (max_width * d)
    return jit

# ============================================================
# Build long-form dataframe at a fixed (Vg, time index)
# ============================================================
def build_long(
    packed_npz_prefix,          # e.g., r"Sample_Dataset\Chemical_gas_dataset"
    gas_order,
    series_order,
    vg_target,
    t_idx,
    n_chunks=32
):

    # Load chunk 0 for labels/voltage and to get mapping
    d0 = np.load(f"{packed_npz_prefix}0.npz", allow_pickle=False)

    # Load all chunks (np.load is lazy; reading arrays happens when indexed)
    chunks = [np.load(f"{packed_npz_prefix}{i}.npz", allow_pickle=False) for i in range(n_chunks)]

    rows = []
    for gas in gas_order:
        labels = [str(x) for x in d0[f"labels_{gas}"].tolist()]
        V = d0[f"voltage_{gas}"] if f"voltage_{gas}" in d0.files else d0["voltage"]
        vi = int(np.argmin(np.abs(V - vg_target)))

        label_to_idx = {lab: i for i, lab in enumerate(labels)}

        for series in series_order:
            di = label_to_idx[f"{gas}_{series}"]

            # Collect per-chunk (chunk_len,) and concatenate -> (2048,)
            resp_parts = [c[f"X_{gas}"][di, vi, :, t_idx] for c in chunks]
            resp = np.concatenate(resp_parts, axis=0)

            for val in resp:
                rows.append((gas, series, float(val)))

    return pd.DataFrame(rows, columns=["gas", "series", "value"])


# ============================================================
# Plot: Fig2(h)-style (log y) + bars(mean/std) + density dots
# ============================================================
## This is to compare the enhancement from cMOF functionalization.
def plot_dist(data_long, gas_order, series_order, var_names, colors,
              y_floor=0.01, y_top=80000, out_path=None):
    df = data_long.copy()

    # Convert to numeric and scale to %
    df["value"] = 100.0 * pd.to_numeric(df["value"], errors="coerce") ## change units

    # Log-safe filtering
    df = df[np.isfinite(df["value"])]
    df = df[df["value"] > 0]

    df["gas"] = pd.Categorical(df["gas"], categories=list(gas_order), ordered=True)
    df["series"] = pd.Categorical(df["series"], categories=list(series_order), ordered=True)

    summary = (
        df.groupby(["gas", "series"])["value"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )

    series_list = [s for s in series_order if s in summary["series"].astype(str).unique().tolist()]
    palette = {series_list[i]: colors[i] for i in range(len(series_list))}

    # Bar placement (same pattern you used)
    bar_width = 0.5 / (len(series_list) - 1) if len(series_list) > 1 else 0.3
    X_axis = np.arange(len(gas_order))

    fig, ax = plt.subplots(figsize=(8, 4), dpi=400)
    ax.set_yscale("log")

    # Bars + error bars (start bars at y_floor for log-axis)
    for i, series in enumerate(series_list):
        for j, gas in enumerate(gas_order):
            row = summary[(summary["series"] == series) & (summary["gas"] == gas)]
            if row.empty:
                continue

            x = X_axis[j] - 0.25 + bar_width * i
            y = row["mean"].values[0]
            err = row["std"].values[0]

            y_top_bar = max(y, y_floor)
            ax.bar(x, y_top_bar - y_floor, bar_width, bottom=y_floor,
                   color=palette[series], alpha=0.4)

            ax.errorbar(x, max(y, y_floor), yerr=err, fmt="none",
                        color="black", capsize=2, linewidth=1)

    # Density-weighted dots
    for (gas, series), g in df.groupby(["gas", "series"], sort=False):
        gas = str(gas)
        series = str(series)
        if series not in series_list:
            continue

        j = list(gas_order).index(gas)
        i = series_list.index(series)

        x_center = X_axis[j] - 0.25 + bar_width * i
        y = g["value"].values
        n = y.size

        jitter = np.zeros(n) if n <= 5 else density_weighted_jitter(y, max_width=bar_width * 0.9, seed=0)
        ax.scatter(x_center + jitter, y,
                   s=12 if n <= 5 else 10,
                   alpha=1.0 if n <= 5 else 0.35,
                   color=palette[series],
                   linewidths=0)

    # Legend
    legend_handles = [Patch(facecolor=colors[i], label=var_names[i]) for i in range(len(series_list))]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=13)

    # Axes style
    ax.set_ylim([y_floor, y_top])
    ax.set_xlabel("")
    ax.set_ylabel("Response (%)", fontsize=14)
    ax.tick_params(axis="x", direction="in", length=4, labelsize=12)
    ax.tick_params(axis="y", direction="in", length=4, labelsize=12)
    ax.set_xticks(X_axis)
    ax.set_xticklabels(list(gas_order), fontsize=14)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=1200)
    plt.show()
    return fig, ax, summary

# ============================================================
# Main (edit these)
# ============================================================

if __name__ == "__main__":
    path=os.getcwd()
    packed_npz_path = path+"\Sample_Dataset\Chemical_gas_dataset"

    gas_order = ("ND", "NH", "HS")
    series_order = ("NiHHTP", "NiHITP", "Ref")

    # Display names (legend order must match series_order)
    var_names = ("NiHHTP-CNFET", "NiHITP-CNFET", "CNFET")
    colors = ("#34678a", "darkorange", "dimgray")

    vg_target = -0.5
    t_idx = 79  # 1 step = 1 min

    data_long = build_long(packed_npz_path, gas_order, series_order, vg_target, t_idx)
    plot_dist(data_long, gas_order, series_order, var_names, colors,
              y_floor=0.01, y_top=80000, out_path=None)
