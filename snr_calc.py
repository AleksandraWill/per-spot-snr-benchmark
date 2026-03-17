# Adapted from tapqir/utils/stats.py and enhanced for per-spot SNR
import math
import torch
from gaussian_spot_model import gaussian_spots, generate_gaussian_spot
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

def gaussian_2d(params, x, y):
    """2D Gaussian function for fitting."""
    amp, x0, y0, sigma_x, sigma_y = params
    # Clamp sigmas to avoid division by zero or negative values
    sigma_x = max(sigma_x, 0.1)
    sigma_y = max(sigma_y, 0.1)
    return amp * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))

def fit_gaussian(aoi, debug=False):
    """Simple Gaussian fit to detect spot parameters in an AOI."""
    y, x = np.indices(aoi.shape)
    initial_guess = [aoi.max(), aoi.shape[1] // 2, aoi.shape[0] // 2, 1.0, 1.0]
    try:
        result = minimize(lambda p: np.sum((gaussian_2d(p, x, y) - aoi)**2), initial_guess, method='L-BFGS-B')
        if debug:
            print(f"Fit success: {result.success}, params: {result.x}")
        return result.x  # [height, x_center, y_center, width_x, width_y]
    except Exception as e:
        if debug:
            print(f"Fit failed: {e}, using initial guess")
        return np.array(initial_guess)

def snr_per_spot(
    data: torch.Tensor,  # (N, Z, P, P)
    background_mean: float,
    background_var: float,
    gain: float = 1.0,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:  # Return SNR and heights
    """
    
    Compute per-spot SNR using the Ordabayev/Tapqir-style formula:
      S = sum_ij w_ij (d_ij - b)
      N = sqrt(sigma_b^2 + gain * b)   
      SNR = S / N
    
    """
    N, Z, P, _ = data.shape
    snr_results = np.zeros((N, Z))
    heights = np.zeros((N, Z))  # New array for heights

    # Precompute noise denominator (independent of AOI if you use global b,o)
    noise_var = background_var + gain * background_mean
    noise = math.sqrt(max(noise_var, 1e-12))

    for n in range(N):
        for z in range(Z):
            aoi = data[n, z].numpy()

            # Fit Gaussian to get spot params
            params = fit_gaussian(aoi, debug=debug)
            height, x0, y0, width_x, width_y = params
            heights[n, z] = height  # Save height here
            width = (width_x + width_y) / 2.0  # Average width

            # Target locs: center of AOI
            target_locs = torch.tensor([[[P // 2, P // 2]]], dtype=torch.float32)  # (1,1,1,2)

            # Generate ideal Gaussian
            height_t = torch.tensor([[[[height]]]], dtype=torch.float32)
            width_t = torch.tensor([[[[width]]]], dtype=torch.float32)
            x_rel = x0 - P // 2  # Relative to center
            y_rel = y0 - P // 2
            x_t = torch.tensor([[[[x_rel]]]], dtype=torch.float32)
            y_t = torch.tensor([[[[y_rel]]]], dtype=torch.float32)
            gaussians = gaussian_spots(height_t, width_t, x_t, y_t, target_locs, P)

            psf = gaussians.squeeze()  # Shape: (P,P)

            # Normalized weights w_ij
            weights = psf / (psf.sum() + 1e-12)

            # Data and corrections
            d = data[n, z].float()
            # Use scalar background and offset (local or global means)
            S = ((d - background_mean) * weights).sum().item()

            snr_results[n, z] = S / noise if noise > 0 else 0.0

            # # Compute weights and signal
            # weights = gaussians / height_t  # Shape: (1,1,1,1,P,P)
            # weights = weights.squeeze()  # Shape: (P,P)
            # background = torch.full_like(data[n, z], offset_mean, dtype=torch.float32)
            # signal = ((data[n, z] - background - offset_mean) * weights).sum()
            # noise = (offset_var + background.mean() * gain + 1e-6).sqrt()  # Add epsilon to avoid /0
            # snr_results[n, z] = (signal / noise).item() if noise > 0 else 0

            if debug:
                print(f"AOI {n+1}, Z {z}: height={height:.1f}, width={width:.2f}, SNR={snr_results[n, z]:.2f}")
                print(f"Debug: signal={S:.2f}, noise={noise:.2f}, background_mean={background_mean}")
    
    return snr_results, heights  # Return both

def simulate_noisy_gaussian_bead(
    height: float,
    width: float,
    P: int,
    background_mean: float,
    background_var: float,
    gain: float = 1.0,
    seed: int = 0,
) -> torch.Tensor:
    """
    Generate a single synthetic AOI with a Gaussian bead plus realistic noise.

    The noise statistics (background_mean, background_var, gain) are taken
    from the experimental data, so the resulting SNR should be in the same
    10–100 range as real beads if 'height' is chosen similarly.
    """
    rng = np.random.default_rng(seed)

    # Ideal Gaussian spot (no noise, no background)
    # Centered in the AOI: x=y=0 relative to target at center
    target_xy = P // 2
    ideal_spot = generate_gaussian_spot(
        height=height,
        width=width,
        x=0.0,
        y=0.0,
        target_x=target_xy,
        target_y=target_xy,
        P=P,
    )  # shape (P, P), numpy

    # Add constant background level
    img = ideal_spot + background_mean

    # Add background / readout noise ~ N(0, sigma_b)
    sigma_b = np.sqrt(max(background_var, 1e-12))
    img += rng.normal(loc=0.0, scale=sigma_b, size=img.shape)

    # Add simple photon noise: variance ~ gain * signal
    # Here we approximate with Gaussian noise for speed
    photon_sigma = np.sqrt(np.clip(gain * np.maximum(img - background_mean, 0.0), 0, None))
    img += rng.normal(loc=0.0, scale=photon_sigma, size=img.shape)

    # Clip to non-negative and convert to torch tensor
    img = np.clip(img, 0, None)
    return torch.tensor(img, dtype=torch.float32)

def snr_from_known_psf(
    aoi: torch.Tensor,     # (P, P)
    height: float,
    width: float,
    background_mean: float,
    background_var: float,
    gain: float = 1.0,
) -> float:
    """
    Compute SNR for a single AOI using known PSF parameters (no fitting).

    S = sum_ij w_ij (d_ij - b)
    N = sqrt(sigma_b^2 + gain * b)
    SNR = S / N
    """
    P = aoi.shape[0]
    assert aoi.shape[0] == aoi.shape[1], "AOI must be square"

    # Target locs: center of AOI
    target_locs = torch.tensor([[[P // 2, P // 2]]], dtype=torch.float32)

    # Known PSF parameters
    height_t = torch.tensor([[[[height]]]], dtype=torch.float32)
    width_t = torch.tensor([[[[width]]]], dtype=torch.float32)
    x_t = torch.tensor([[[[0.0]]]], dtype=torch.float32)  # centered
    y_t = torch.tensor([[[[0.0]]]], dtype=torch.float32)

    gaussians = gaussian_spots(height_t, width_t, x_t, y_t, target_locs, P)
    psf = gaussians.squeeze()             # (P, P)
    weights = psf / (psf.sum() + 1e-12)   # normalized w_ij

    d = aoi.float()
    S = ((d - background_mean) * weights).sum().item()

    noise_var = background_var + gain * background_mean
    noise = math.sqrt(max(noise_var, 1e-12))

    return S / noise if noise > 0 else 0.0


def analyze_and_save_snr_summary(snr: np.ndarray, data_path: Path) -> None:
    """
    Analyze SNR results, print summary to console, and save to files.
    """
    # Automatic Analysis and Summary
    print("\n=== SNR Analysis Summary ===")
    N, Z = snr.shape  # N AOIs, Z z-slices

    # Używamy errstate, aby zignorować ostrzeżenia o dzieleniu przez zero
    with np.errstate(divide='ignore', invalid='ignore'):
        for n in range(N):
            aoi_snr = snr[n]
            avg_snr = np.mean(aoi_snr)
            # FIX: Check if there is more than 1 slice before calculating SD
            if len(aoi_snr) > 1:
                sd_snr = np.std(aoi_snr, ddof=1)
            else:
                sd_snr = 0.0 # Or float('nan')

            if np.isnan(sd_snr): sd_snr = 0.0

            max_snr = np.max(aoi_snr)
            min_snr = np.min(aoi_snr)
            best_z = np.argmax(aoi_snr)  # Z-slice with highest SNR
            positive_slices = np.sum(aoi_snr > 0)  # Count positive SNR slices
            
            print(f"\nAOI {n+1} (Bead {n+1}):")  # Start from 1
            print(f"  - Average SNR: {avg_snr:.2f}")
            print(f"  - SD of SNR: {sd_snr:.2f}")
            print(f"  - Max SNR: {max_snr:.2f} (at Z-slice {best_z})")
            print(f"  - Min SNR: {min_snr:.2f}")
            print(f"  - Positive SNR slices: {positive_slices}/{Z} ({100 * positive_slices / Z:.1f}%)")
            
            if max_snr > 50:
                print("  - Quality: Excellent (bright, well-focused bead)")
            elif max_snr > 10:
                print("  - Quality: Good")
            elif max_snr > 0:
                print("  - Quality: Moderate")
            else:
                print("  - Quality: Poor (check bead position or focus)")

        # Overall stats
        overall_avg = np.mean(snr)
        overall_max = np.max(snr)
        overall_min = np.min(snr)
        # Check total number of points across all AOIs and Z-slices
        total_points = snr.size 
        if total_points > 1:
            overall_sd = np.std(snr, ddof=1)
        else:
            overall_sd = 0.0

        if np.isnan(overall_sd): overall_sd = 0.0

    ideal_mask = snr.max(axis=1) > 50
    ideal_indices = np.where(ideal_mask)[0]  # zero-based

    if ideal_indices.size > 0:
        print("\nAOIs ideal for benchmarking (max SNR > 50):",
                ", ".join(str(i+1) for i in ideal_indices))
    else:
        print("\nNo AOIs reached the ideal threshold (max SNR > 50).") 

    total_positive = np.sum(snr > 0)
    total_slices = N * Z

    print(f"\nOverall (All AOIs and Z-slices):")
    print(f"  - Average SNR: {overall_avg:.2f}")
    print(f"  - Max SNR: {overall_max:.2f}")
    print(f"  - Min SNR: {overall_min:.2f}")
    print(f"  - Overall SD of SNR: {overall_sd:.2f}")
    print(f"  - Positive SNR slices: {total_positive}/{total_slices} ({100 * total_positive / total_slices:.1f}%)")

    # Recommendations
    best_aoi = np.argmax(snr.max(axis=1))
    print(f"\nRecommendation: Best bead is AOI {best_aoi + 1} (use for benchmarking). Focus on Z-slices with SNR >10 for reliable data.")  # Start from 1

    # Save summary to file
    summary_file = data_path / "snr_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("SNR Analysis Summary\n")
        f.write(f"Generated on: {np.datetime64('now')}\n\n")
        for n in range(N):
            aoi_snr = snr[n]

            if aoi_snr.size > 1:
                current_sd = np.std(aoi_snr, ddof=1)
            else:
                current_sd = 0.0
            
            if np.isnan(current_sd): current_sd = 0.0

            f.write(f"AOI {n+1}:\n")  # Start from 1
            f.write(f"  Average SNR: {np.mean(aoi_snr):.2f}\n")
            f.write(f"  SD SNR: {np.std(aoi_snr, ddof=1):.2f}\n")
            f.write(f"  Max SNR: {np.max(aoi_snr):.2f} (Z-slice {np.argmax(aoi_snr)})\n")
            f.write(f"  Positive slices: {np.sum(aoi_snr > 0)}/{Z}\n\n")
        
        f.write(f"Overall Average: {overall_avg:.2f}\n")
        f.write(f"Overall SD: {overall_sd:.2f}\n")
        f.write(f"Best AOI: {best_aoi + 1}\n")  # Start from 1
        
        ideal_indices = np.where(snr.max(axis=1) > 50)[0]
        f.write(f"Ideal AOIs (max SNR > 50): "
        + ', '.join(str(i+1) for i in ideal_indices)
        + "\n"
        )
    
    print(f"\nFull summary saved to {summary_file}")

    # Save raw data (CSV uses the array)
    np.savetxt(data_path / "snr.csv", snr, delimiter=",", header=",".join([f"AOI{n+1}_Z{z}" for n in range(N) for z in range(Z)]), comments='')  # Start from 1
    print(f"Raw SNR saved to {data_path / 'snr_raw.npy'} and {data_path / 'snr.csv'}")
    print(f"Structured SNR data saved to {data_path / 'snr.npy'} (for aggregation)")

def compute_snr_for_aois(data_path, offset_samples, offset_weights, P=50, gain=1.0, debug=True):
    """
    Compute per-spot SNR for extracted AOIs using Gaussian fitting.
    """
    data = torch.tensor(np.load(data_path / "data.npy"), dtype=torch.float32)
    
    # background = histogram over offset patch (includes camera offset)
    samples_t = torch.tensor(offset_samples, dtype=torch.float32)
    weights_t = torch.tensor(offset_weights, dtype=torch.float32)
    weights_t = weights_t / weights_t.sum()

    background_mean = (samples_t * weights_t).sum().item()
    background_sq_mean = (samples_t**2 * weights_t).sum().item()
    background_var = max(background_sq_mean - background_mean**2, 1e-12)

    snr, heights = snr_per_spot(
        data,
        background_mean=background_mean,
        background_var=background_var,
        gain=gain,
        debug=debug,
    )

    np.save(data_path / "heights.npy", heights)
    print(f"Heights saved to {data_path / 'heights.npy'}")

    # Create a DataFrame for snr_table with AOI indices
    N, Z = snr.shape
    snr_df = pd.DataFrame()
    for n in range(N):
        for z in range(Z):
            snr_df = pd.concat([snr_df, pd.DataFrame({"AOI": [n+1], "z": [z], "SNR": [snr[n, z]]})], ignore_index=True)

    # Save as dict to match expected format in analyze_config.py
    snr_data = {"snr_table": snr_df}
    np.save(data_path / "snr.npy", snr_data, allow_pickle=True)

    # Also save the raw array for backward compatibility (optional, but keeps existing behavior)
    np.save(data_path / "snr_raw.npy", snr)

    # Now handle analysis and saving
    analyze_and_save_snr_summary(snr, data_path)
    
    return snr