from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_blue_bypass_vs_split(bypass_root: str, split_root: str) -> None:
    """
    bypass_root, split_root: paths to 'results' folders for
    beads_bypass_blue and beads_split_blue, respectively.
    """
    bypass_root = Path(bypass_root)
    split_root = Path(split_root)

    fig_root = bypass_root.parent.parent / "figures"
    fig_root.mkdir(exist_ok=True)

    # --- 1. Load data ---
    per_pos_bypass = pd.read_csv(bypass_root / "config_per_pos.csv")
    per_pos_split = pd.read_csv(split_root / "config_per_pos.csv")
    per_aoi_bypass = pd.read_csv(bypass_root / "config_per_AOI.csv")
    per_aoi_split = pd.read_csv(split_root / "config_per_AOI.csv")

   # Filter for blue channel if column exists
    for df in [per_pos_bypass, per_pos_split, per_aoi_bypass, per_aoi_split]:
        if "channel" in df.columns:
            df.query("channel == 'blue'", inplace=True)
    
    # --- 2. Mean SNR per position ---
    per_pos = per_pos_bypass.merge(
        per_pos_split[["pos", "mean_SNR"]].rename(
            columns={"mean_SNR": "mean_SNR_split"}
        ),
        on="pos",
        how="inner",
    )
    per_pos = per_pos.rename(columns={"mean_SNR": "mean_SNR_bypass"})
    x = per_pos["pos"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, per_pos["mean_SNR_bypass"], marker="o", label="bypass blue")
    plt.plot(x, per_pos["mean_SNR_split"], marker="s", label="split blue")
    plt.xlabel("Position")
    plt.ylabel("Mean SNR")
    plt.title("Mean SNR per position: bypass vs split (blue)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_root / "blue_mean_snr_per_position.png", dpi=300)
    plt.close()

    # --- 3. SNR efficiency score (ratio) ---
    per_pos["efficiency"] = (
        per_pos["mean_SNR_split"] / per_pos["mean_SNR_bypass"]
    )

    plt.figure(figsize=(8, 4))
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.plot(x, per_pos["efficiency"], marker="o", color="blue")
    plt.xlabel("Position")
    plt.ylabel("SNR(split) / SNR(bypass)")
    plt.title("SNR efficiency ratio per position (blue)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_root / "blue_snr_efficiency_ratio.png", dpi=300)
    plt.close()

    # --- 4. Spatial Efficiency Map ---
    aoi_data = per_aoi_bypass.merge(
        per_aoi_split[["pos", "AOI", "max_SNR", "best_z"]].rename(
            columns={"max_SNR": "max_SNR_split", "best_z": "best_z_split"}
        ),
        on=["pos", "AOI"]
    )
    aoi_data["efficiency"] = aoi_data["max_SNR_split"] / aoi_data["max_SNR"]

    coords_path = bypass_root / "positions.npy"
    if coords_path.exists():
        coords = np.load(coords_path)
        x_c = [coords[int(r["AOI"])-1][0] for _, r in aoi_data.iterrows()]
        y_c = [coords[int(r["AOI"])-1][1] for _, r in aoi_data.iterrows()]

        plt.figure(figsize=(10, 7))
        sc = plt.scatter(x_c, y_c, c=aoi_data["efficiency"], cmap="viridis", s=150, edgecolor="black")
        plt.colorbar(sc, label="Efficiency (Split / Bypass)")
        plt.title("Spatial Efficiency Map (blue)")
        plt.savefig(fig_root / "blue_spatial_efficiency_map.png", dpi=300)
        plt.close()

    # --- 5. Focus Offset Plot ---
    plt.figure(figsize=(6, 6))
    # Identity line (where points would be if there was zero focus shift)
    max_z = max(aoi_data["best_z"].max(), aoi_data["best_z_split"].max())
    plt.plot([0, max_z], [0, max_z], color="gray", linestyle="--", label="Zero Shift")
    
    plt.scatter(aoi_data["best_z"], aoi_data["best_z_split"], 
                color="blue", alpha=0.6, s=100, edgecolor="white", label="AOIs")
    
    plt.xlabel("Best Z-slice (Bypass)")
    plt.ylabel("Best Z-slice (Split)")
    plt.title("Focal Shift Benchmark: Bypass vs Split (blue)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate mean shift
    z_shift = (aoi_data["best_z_split"] - aoi_data["best_z"]).mean()
    plt.annotate(f"Mean Z-Shift: {z_shift:.2f} slices", 
                 xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(fig_root / "blue_focus_offset_plot.png", dpi=300)
    plt.close()
    print(f"Focus offset plot saved to {fig_root}")

        # --- 6. Overall SNR Comparison (Global Statistics) ---
    # Pooling all AOI-level max SNRs for global stats
    global_mean_bypass = aoi_data["max_SNR"].mean()
    global_sd_bypass = aoi_data["max_SNR"].std()
    
    global_mean_split = aoi_data["max_SNR_split"].mean()
    global_sd_split = aoi_data["max_SNR_split"].std()
    
    retention = (global_mean_split / global_mean_bypass) * 100
    loss = 100 - retention

    plt.figure(figsize=(6, 7))
    labels = ["Bypass", "Split"]
    means = [global_mean_bypass, global_mean_split]
    stds = [global_sd_bypass, global_sd_split]
    
    # Create bar plot with error bars (yerr)
    bars = plt.bar(labels, means, yerr=stds, capsize=10,
                   color=["#3498db", "#2980b9"], edgecolor="black", alpha=0.8)
    
    plt.ylabel("Mean AOI Peak SNR")
    plt.title("Global Blue Channel SNR Comparison (All Positions)")
    
    # Annotate bars with "Mean ± SD"
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 1,
                 f'{means[i]:.1f} ± {stds[i]:.1f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add text box with the retention/loss stats
    stats_text = (f"Global Retention: {retention:.1f}%\n"
                  f"Estimated Loss: {loss:.1f}%")
    plt.annotate(stats_text, xy=(0.5, 0.4), xycoords='axes fraction',
                 ha='center', bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Ensure the y-axis accommodates the error bars
    plt.ylim(0, max([m + s for m, s in zip(means, stds)]) * 1.15)
    
    plt.tight_layout()
    plt.savefig(fig_root / "blue_global_snr_comparison.png", dpi=300)
    plt.close()

    print(f"All blue channel comparison plots saved to {fig_root}")