import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def assess_crosstalk(results_root: str) -> None:
    """
    Assess crosstalk in dual-channel SNR data.

    For each AOI, determine the 'dominant' channel (higher SNR) and compute:
    - Crosstalk fraction: SNR_wrong / SNR_right
    - Plot histograms of crosstalk per AOI and bar plots per position.

    :param results_root: Path to results folder (e.g., 'D:\\tu_projects\\benchmarking\\data\\beads_dual_red_blue\\results')
    """
    results_root = Path(results_root)
    fig_root = results_root.parent.parent / "figures"
    fig_root.mkdir(exist_ok=True)

    # 2. Load per-AOI data
    per_aoi_path = results_root / "config_per_AOI.csv"
    if not per_aoi_path.exists():
        print(f"Error: {per_aoi_path} not found.")
        return

    df = pd.read_csv(per_aoi_path)

    # Pivot to get blue and red SNR per AOI/position
    df_pivot = df.pivot_table(index=["pos", "AOI"], columns="channel", values="max_SNR", aggfunc="first").reset_index()
    df_pivot = df_pivot.dropna()  # Drop AOIs missing data

    # Compute crosstalk per AOI
    crosstalk_list = []
    found_any_coords = False

    for _, row in df_pivot.iterrows():
        snr_blue = row["blue"]
        snr_red = row["red"]
        pos = row["pos"]
        aoi_num = int(row["AOI"])

        # Calculate crosstalk fraction
        if snr_blue > snr_red:
            crosstalk_frac = snr_red / snr_blue if snr_blue > 0 else 0
            crosstalk_type = "red_from_blue"
        else:
            crosstalk_frac = snr_blue / snr_red if snr_red > 0 else 0
            crosstalk_type = "blue_from_red"

        # --- NEW: DYNAMIC SEARCH FOR POSITIONS.NPY ---
        # We check the 'blue' folder for this specific position
        coords_path = results_root / pos / "blue" / "positions.npy"
        
        x, y = None, None
        if coords_path.exists():
            aoi_coords = np.load(coords_path)
            # AOI in CSV is 1-based, index in npy is 0-based
            if len(aoi_coords) >= aoi_num:
                x, y = aoi_coords[aoi_num - 1]
                found_any_coords = True
        else:
            print(f"Warning: Could not find coords for {pos} at {coords_path}")

        crosstalk_list.append({
            "x": x, "y": y, 
            "pos": pos, 
            "AOI": aoi_num, 
            "crosstalk_frac": crosstalk_frac, 
            "type": crosstalk_type
        })

    crosstalk_df = pd.DataFrame(crosstalk_list)


     # 3. Plot Spatial Crosstalk Map
    if found_any_coords:
        plt.figure(figsize=(10, 7))
        # Filter out rows where we didn't find coordinates
        plot_df = crosstalk_df.dropna(subset=["x", "y"])
        
        scatter = plt.scatter(plot_df["x"], plot_df["y"], c=plot_df["crosstalk_frac"], 
                             cmap="viridis", s=200, edgecolor="white", linewidth=1.5)
        
        plt.colorbar(scatter, label="Crosstalk Fraction (SNR_leak / SNR_signal)")
        plt.xlabel("Sensor X Coordinate (px)")
        plt.ylabel("Sensor Y Coordinate (px)")
        plt.title("Spatial Crosstalk Map: Dual Channel Performance")
        plt.grid(True, linestyle="--", alpha=0.6)
        
        # Annotate with Position and AOI number
        for _, row in plot_df.iterrows():
            plt.annotate(f"{row['pos']}_AOI{int(row['AOI'])}", 
                         (row['x']+8, row['y']+8), fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.savefig(fig_root / "spatial_crosstalk_map.png", dpi=300)
        plt.close()
        print(f"Spatial crosstalk map saved to {fig_root}")
    else:
        print("Warning: No coordinates found in subfolders. Skipping spatial map.")

    # Histogram of crosstalk fractions per AOI
    plt.figure(figsize=(8, 6))
    plt.hist(crosstalk_df["crosstalk_frac"], bins=10, alpha=0.7, edgecolor="black")
    plt.xlabel("Crosstalk Fraction (SNR_wrong / SNR_right)")
    plt.ylabel("Number of AOIs")
    plt.title("Crosstalk Histogram per AOI (Dual Channel)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_root / "crosstalk_histogram_per_aoi.png", dpi=300)
    plt.close()

    # Bar plot of mean crosstalk per position
    per_pos_crosstalk = crosstalk_df.groupby("pos").agg(
        mean_crosstalk=("crosstalk_frac", "mean"),
        std_crosstalk=("crosstalk_frac", "std"),
        count=("crosstalk_frac", "size")
    ).reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(per_pos_crosstalk["pos"], per_pos_crosstalk["mean_crosstalk"], yerr=per_pos_crosstalk["std_crosstalk"], capsize=5)
    plt.xlabel("Position")
    plt.ylabel("Mean Crosstalk Fraction")
    plt.title("Mean Crosstalk per Position (Dual Channel)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_root / "crosstalk_barplot_per_position.png", dpi=300)
    plt.close()

    # Print summary stats
    overall_mean = crosstalk_df["crosstalk_frac"].mean()
    overall_std = crosstalk_df["crosstalk_frac"].std()
    low_crosstalk_count = (crosstalk_df["crosstalk_frac"] < 0.1).sum()  # e.g., <10% crosstalk
    total_aois = len(crosstalk_df)

    print("=== Crosstalk Assessment Summary ===")
    print(f"Overall Mean Crosstalk: {overall_mean:.3f} ± {overall_std:.3f}")
    print(f"AOIs with Low Crosstalk (<10%): {low_crosstalk_count}/{total_aois} ({100 * low_crosstalk_count / total_aois:.1f}%)")
    print(f"Plots saved to {fig_root}")

    # Optional: Save detailed crosstalk data
    crosstalk_df.to_csv(results_root / "crosstalk_analysis.csv", index=False)
    print(f"Detailed crosstalk data saved to {results_root / 'crosstalk_analysis.csv'}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python assess_crosstalk.py <results_root>")
        sys.exit(1)
    assess_crosstalk(sys.argv[1])