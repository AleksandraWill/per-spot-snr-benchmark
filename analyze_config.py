import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def quality_label(max_snr: float) -> str:
    if max_snr > 50:
        return "Excellent"
    elif max_snr > 10:
        return "Good"
    elif max_snr > 0:
        return "Moderate"
    else:
        return "Poor"

def _load_snr_one_channel(
    pos_or_channel_dir: Path,
    pos_name: str,
    channel: str | None,
    tag: str,
) -> pd.DataFrame:
    # construct path based on tag
    if tag == "real":
        snr_file = pos_or_channel_dir / "snr.npy"
    elif tag == "ideal":
        snr_file = pos_or_channel_dir / "ideal_SNR" / "snr.npy"
    else:
        raise ValueError(f"Invalid tag: {tag}. Must be 'real' or 'ideal'.")
    
    if not snr_file.is_file():
        raise FileNotFoundError(f"Could not find snr.npy at {snr_file}")

    snr_data = np.load(snr_file, allow_pickle=True)

    # Handle different saved formats:
    # 1) dict with "snr_table" (current format)
    # 2) bare numpy array / float of SNR values (older/simple format)
    if isinstance(snr_data, np.ndarray) and snr_data.dtype == "O":
        # likely a dict saved with allow_pickle=True
        snr_data = snr_data.item()

    if isinstance(snr_data, dict) and "snr_table" in snr_data:
        df = pd.DataFrame(snr_data["snr_table"])
    else:
        # Fallback: treat snr_data itself as 1D SNR array over z
        arr = np.array(snr_data).ravel()
        df = pd.DataFrame({"SNR": arr})
        # If you have z information elsewhere, you can add it here;
        # otherwise, use simple index as z:
        df["z"] = np.arange(len(df))

    df["pos"] = pos_name
    df["type"] = tag  # "real" or "ideal"
    if channel is not None:
        df["channel"] = channel
    return df

def load_snr_from_pos(pos_dir: Path, tag: str) -> pd.DataFrame:
    """
    Load SNR data for a position.

    Supports two layouts:
    1) Single config:
         pos_dir/snr.npy (for real) or pos_dir/ideal_SNR/snr.npy (for ideal)
    2) Dual config:
         pos_dir/blue/snr.npy and pos_dir/red/snr.npy (for real)
         pos_dir/blue/ideal_SNR/snr.npy and pos_dir/red/ideal_SNR/snr.npy (for ideal)

    tag: "real" or "ideal"
    """
    blue_dir = pos_dir / "blue"
    red_dir = pos_dir / "red"

    pos_name = pos_dir.name

    # Dual configuration
    if blue_dir.is_dir() and red_dir.is_dir():
        dfs = []
        # Check based on tag
        if tag == "real":
            snr_path_blue = blue_dir / "snr.npy"
            snr_path_red = red_dir / "snr.npy"
        else:  # tag == "ideal"
            snr_path_blue = blue_dir / "ideal_SNR" / "snr.npy"
            snr_path_red = red_dir / "ideal_SNR" / "snr.npy"
        
        if snr_path_blue.is_file():
            dfs.append(_load_snr_one_channel(blue_dir, pos_name, "blue", tag))
        if snr_path_red.is_file():
            dfs.append(_load_snr_one_channel(red_dir, pos_name, "red", tag))

        if not dfs:
            raise FileNotFoundError(
                f"No snr.npy found under {blue_dir} or {red_dir} for tag '{tag}'"
            )
        return pd.concat(dfs, ignore_index=True)

    # Single configuration
    return _load_snr_one_channel(pos_dir, pos_name, None, tag)

def analyze_configuration(root: Path, manual_channel: str | None = None) -> None:
    # """
    # root: configuration results directory, e.g.
    #   D:\\...\\beads_bypass_blue\\results
    # Expects subfolders pos1, pos2, ...
    # """
    pos_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    all_rows = []
    for pos in pos_dirs:
        # Real beads
        df_real = load_snr_from_pos(pos, tag="real")
        if not df_real.empty:
            # If no channel was found by folder structure, use the manual tag
            if "channel" not in df_real.columns and manual_channel:
                df_real["channel"] = manual_channel
            all_rows.append(df_real)

        # Ideal Gaussian SNR (if present)
        df_ideal = load_snr_from_pos(pos, tag="ideal")
        if not df_ideal.empty:
            # If no channel was found by folder structure, use the manual tag
            if "channel" not in df_ideal.columns and manual_channel:
                df_ideal["channel"] = manual_channel
            all_rows.append(df_ideal)

    if not all_rows:
        print(f"No SNR data found under {root}")
        return

    snr_all = pd.concat(all_rows, ignore_index=True)

    # --- Per AOI / per position stats (real beads only) ---
    real = snr_all[snr_all["type"] == "real"]

    # Decide grouping keys depending on whether channel is present
    if "channel" in real.columns:
        aoi_group_keys = ["channel", "pos", "AOI"]
        pos_group_keys = ["channel", "pos"]
    else:
        aoi_group_keys = ["pos", "AOI"]
        pos_group_keys = ["pos"]

    if "AOI" in real.columns:
        # AOI-resolved data available
        per_aoi = (
            real.groupby(aoi_group_keys, as_index=False)
            .agg(
                mean_SNR=("SNR", "mean"),
                sd_SNR=("SNR", lambda x: x.std(ddof=1)),
                max_SNR=("SNR", "max"),
                min_SNR=("SNR", "min"),
            )
        )

        # Best z per AOI
        best_z = (
            real
            .loc[
                lambda d: d.groupby(aoi_group_keys)["SNR"].idxmax(),
                aoi_group_keys + ["z"],
            ]
            .rename(columns={"z": "best_z"})
        )
        per_aoi = per_aoi.merge(best_z, on=aoi_group_keys, how="left")

        # Quality labels
        per_aoi["quality"] = per_aoi["max_SNR"].apply(quality_label)

        # === NEW: keep only AOIs with max_SNR > 50 ===
        per_aoi = per_aoi[per_aoi["max_SNR"] > 50].reset_index(drop=True)

        # Per position from AOI stats
        per_pos = (
            per_aoi.groupby(pos_group_keys, as_index=False)
            .agg(
                mean_SNR=("mean_SNR", "mean"),
                sd_SNR=("mean_SNR", lambda x: x.std(ddof=1)),
                max_SNR=("max_SNR", "max"),
                min_SNR=("min_SNR", "min"),
            )
        )
    else:
        # No AOI column
        per_aoi = real.copy()
        per_pos = (
            real.groupby(pos_group_keys, as_index=False)
            .agg(
                mean_SNR=("SNR", "mean"),
                sd_SNR=("SNR", lambda x: x.std(ddof=1)),
                max_SNR=("SNR", "max"),
                min_SNR=("SNR", "min"),
            )
        )


    # --- Global configuration-level stats (real beads only) ---
    real = snr_all[snr_all["type"] == "real"]

    if "channel" in real.columns:
        # overall per channel
        overall_df = (
            real.groupby("channel", as_index=False)
            .agg(
                overall_mean_SNR=("SNR", "mean"),
                overall_sd_SNR=("SNR", lambda x: x.std(ddof=1)),
                overall_max_SNR=("SNR", "max"),
                overall_min_SNR=("SNR", "min"),
                total_points=("SNR", "size"),
            )
        )
    else:
        overall = {
            "overall_mean_SNR": real["SNR"].mean(),
            "overall_sd_SNR": real["SNR"].std(ddof=1),
            "overall_max_SNR": real["SNR"].max(),
            "overall_min_SNR": real["SNR"].min(),
            "total_points": len(real),
        }
        overall_df = pd.DataFrame([overall])

    # --- Ideal vs real comparison (if ideal present) ---
    if "channel" in snr_all.columns:
        group_cols = ["channel", "type"]
    else:
        group_cols = ["type"]

    by_type = (
        snr_all.groupby(group_cols, as_index=False)
        .agg(
            mean_SNR=("SNR", "mean"),
            sd_SNR=("SNR", lambda x: x.std(ddof=1)),
            max_SNR=("SNR", "max"),
            min_SNR=("SNR", "min"),
            n_points=("SNR", "size"),
        )
    )

    # --- Save CSV outputs ---
    (root / "config_per_AOI.csv").write_text(per_aoi.to_csv(index=False))
    (root / "config_per_pos.csv").write_text(per_pos.to_csv(index=False))
    (root / "config_overall.csv").write_text(overall_df.to_csv(index=False))
    (root / "config_real_vs_ideal.csv").write_text(by_type.to_csv(index=False))
    (root / "config_all_SNR_long.csv").write_text(snr_all.to_csv(index=False))

    print(f"Saved per-AOI stats to {root / 'config_per_AOI.csv'}")
    print(f"Saved per-position stats to {root / 'config_per_pos.csv'}")
    print(f"Saved overall stats to {root / 'config_overall.csv'}")
    print(f"Saved type comparison to {root / 'config_real_vs_ideal.csv'}")
    print(f"Saved long SNR table to {root / 'config_all_SNR_long.csv'}")


def analyze_configuration_from_path_string(path_str: str, manual_channel: str | None = None) -> None:
    """
    Thin wrapper to be called from main.py.
    Receives a path string, validates it, then runs analyze_configuration().
    """
    root = Path(path_str)
    if not root.is_dir():
        print(f"{root} is not a directory.")
        return
    analyze_configuration(root, manual_channel=manual_channel)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate SNR statistics for a full configuration "
                    "(e.g. bypass_blue) across positions."
    )
    parser.add_argument(
        "results_root",
        help="Path to configuration results folder, e.g. "
             "D:\\\\...\\\\beads_bypass_blue\\\\results",
    )
    args = parser.parse_args()
    analyze_configuration_from_path_string(args.results_root)