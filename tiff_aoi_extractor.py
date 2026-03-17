# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile
from matplotlib.patches import Rectangle

import matplotlib
matplotlib.use('Agg')  # Set headless backend for plotting

# logger
logger = logging.getLogger(__name__)


def bin_hist(
    samples: torch.Tensor, weights: torch.Tensor, s: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, r = divmod(len(samples) - 1, s)
    n = 1 + q + int(bool(r))
    new_samples = torch.zeros(n, dtype=torch.int)
    new_weights = torch.zeros(n)
    new_samples[0] = samples[0]
    new_weights[0] = weights[0]
    new_samples[1 : 1 + q] = samples[1 + s // 2 : 1 + q * s : s]
    for i in range(s):
        new_weights[1 : 1 + q] += weights[1 + i : 1 + q * s : s]
    if r:
        new_samples[-1] = samples[1 + q * s + r // 2]
        new_weights[-1] = weights[1 + q * s :].sum()
    return new_samples, new_weights


class TiffStackDataset:
    """
    TiffStackDataset processes TIFF z-stack files and extracts AOIs around specified positions.

    :param tiff_path: Path to the TIFF stack file.
    :param positions: List of (x, y) positions for AOI centers (0-based indexing).
    :param offset_x: X-coordinate of background offset region.
    :param offset_y: Y-coordinate of background offset region.
    :param offset_P: Size of background offset region.
    :param z_start: Starting z-slice (0-based).
    :param z_end: Ending z-slice (inclusive, 0-based).
    """

    def __init__(self, **kwargs):
        self.tiff_path = Path(kwargs["tiff_path"])
        self.positions = kwargs["positions"]  # list of (x, y) tuples
        self.offset_x = kwargs["offset_x"]
        self.offset_y = kwargs["offset_y"]
        self.offset_P = kwargs["offset_P"]
        self.z_start = kwargs.get("z_start", 0)
        self.z_end = kwargs.get("z_end", None)

        # Read the TIFF stack
        self.stack = tifffile.imread(self.tiff_path)
        if self.stack.ndim == 3:  # (Z, Y, X)
            self.Z, self.height, self.width = self.stack.shape
        elif self.stack.ndim == 2:  # Single slice
            self.stack = self.stack[np.newaxis, ...]
            self.Z, self.height, self.width = self.stack.shape
        else:
            raise ValueError("Unsupported TIFF dimensions")

        if self.z_end is None:
            self.z_end = self.Z - 1
        self.stack = self.stack[self.z_start : self.z_end + 1]
        self.Z = self.stack.shape[0]

        self.N = len(self.positions)

    @property
    def F(self) -> int:
        return self.Z  # Treat z-slices as frames

    def __getitem__(self, key):
        if isinstance(key, slice):
            imgs = []
            for z in range(
                key.start, key.stop, 1 if key.step is None else key.step
            ):
                imgs.append(self[z])
            return np.stack(imgs, 0)
        z = key
        return self.stack[z]

    def plot(
        self,
        P: int,
        n: int = None,
        z: int = 0,
        save: bool = False,
        path: Path = None,
        ax=None,
        item: dict = {},
        title: str = None,
    ) -> None:
        """
        Plot AOIs in the field of view.

        :param P: AOI size.
        :param path: Path where to save plots.
        :param save: Save plots.
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 10 * self.height / self.width))
            ax = fig.add_subplot(1, 1, 1)

        img = self[z]
        if "fov" in item:
            item["fov"].set_data(img)
        else:
            vmin = np.percentile(img, 1)
            vmax = np.percentile(img, 99)
            item["fov"] = ax.imshow(img, vmin=vmin, vmax=vmax, cmap="gray")

        for i, (x, y) in enumerate(self.positions):
            # AOI rectangle
            x_pos = x - 0.5 * (P - 1) - 0.5
            y_pos = y - 0.5 * (P - 1) - 0.5
            if f"aoi_n{i}" in item:
                item[f"aoi_n{i}"].set_xy((x_pos, y_pos))
            else:
                item[f"aoi_n{i}"] = ax.add_patch(
                    Rectangle(
                        (x_pos, y_pos),
                        P,
                        P,
                        edgecolor="#AA3377",
                        lw=1,
                        facecolor="none",
                    )
                )
            if n == i:
                item[f"aoi_n{i}"].set_edgecolor("red")
                item[f"aoi_n{i}"].set(zorder=2)

        # Offset region
        ax.add_patch(
            Rectangle(
                (self.offset_x, self.offset_y),
                self.offset_P,
                self.offset_P,
                edgecolor="#CCBB44",
                lw=1,
                facecolor="none",
            )
        )

        if title is None:
            title = f"AOI {n}, Z-slice {z}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        if save:
            plt.savefig(path / f"aois_z{z}.png", dpi=300)


def read_tiff_stack(path, progress_bar, **kwargs):
    """
    Preprocess TIFF z-stack and extract AOIs around Tetraspeck bead positions.
    """
    P = 50  # Fixed crop size
    tiff_path = kwargs.pop("tiff_path")
    positions = kwargs.pop("positions")
    
    bin_size = kwargs.pop("bin_size", 10)
    name = kwargs.pop("dataset", "tetraspeck")

    offsets = defaultdict(int)
    offset_medians = []

    logger.info("Processing TIFF stack")
    tiff_ds = TiffStackDataset(tiff_path=tiff_path, positions=positions, **kwargs)
    offset_P = kwargs.pop("offset_P")
    N = tiff_ds.N
    Z = tiff_ds.Z
    data = np.zeros((N, Z, P, P), dtype="int")
    target_xy = np.zeros((N, Z, 2))

    # Plot initial positions
    tiff_ds.plot(P, path=path, save=True, item={}, title="AOI locations")

    # Loop through each z-slice
    for z in progress_bar(range(Z)):
        img = tiff_ds[z]

        # Sample offset
        offset_img = img[
            tiff_ds.offset_y : tiff_ds.offset_y + offset_P,
            tiff_ds.offset_x : tiff_ds.offset_x + offset_P,
        ]
        offset_medians.append(np.median(offset_img))
        values, counts = np.unique(offset_img, return_counts=True)
        for value, count in zip(values, counts):
            offsets[value] += count

        # Extract AOIs
        for n, (x, y) in enumerate(positions):
            shiftx = round(x - 0.5 * (P - 1))
            shifty = round(y - 0.5 * (P - 1))
            data[n, z, :, :] = img[shifty : shifty + P, shiftx : shiftx + P]
            target_xy[n, z, 0] = x - shiftx
            target_xy[n, z, 1] = y - shifty

    # Assert target positions are within AOI
    assert (target_xy >= 0).all() and (target_xy < P).all()

    logger.info("Processing extracted AOIs ...")

    # Convert to tensors
    data = torch.tensor(data)
    target_xy = torch.tensor(target_xy)
    is_ontarget = torch.ones(N, dtype=torch.bool)  # All are on-target for beads

    # Process offset data
    offsets = OrderedDict(sorted(offsets.items()))
    offset_samples = np.array(list(offsets.keys()))
    offset_weights = np.array(list(offsets.values()))
    
    min_data = data.min()
    if min_data < offset_samples[0]:
        offset_samples = np.insert(offset_samples, 0, min_data - 1)
        offset_weights = np.insert(offset_weights, 0, 1)
    
    offset_weights = offset_weights / offset_weights.sum()
    high_mask = offset_weights.cumsum() > 0.995
    high_weights = offset_weights[high_mask].sum()
    offset_samples = offset_samples[~high_mask]
    offset_weights = offset_weights[~high_mask]
    
    if offset_weights.size > 0:
        offset_weights[-1] += high_weights
    else:
        offset_samples = np.array([0], dtype=int)
        offset_weights = np.array([1.0])
    
    offset_samples = torch.tensor(offset_samples, dtype=torch.int)
    offset_weights = torch.tensor(offset_weights, dtype=torch.float32)

    offset_samples, offset_weights = bin_hist(offset_samples, offset_weights, bin_size)

    logger.info(
        f"Dataset: N={N} AOIs, "
        f"F={Z} z-slices, "
        f"Px={P} pixels, "
        f"Py={P} pixels"
    )

    # Save data manually
    np.save(path / "data.npy", data.numpy())
    np.save(path / "target_xy.npy", target_xy.numpy())
    np.save(path / "is_ontarget.npy", is_ontarget.numpy())
    np.save(path / "offset_samples.npy", offset_samples.numpy())
    np.save(path / "offset_weights.npy", offset_weights.numpy())

    logger.info("- saving images")

    # Plot offset distribution
    plt.figure(figsize=(3, 3))
    plt.hist(offset_img.flatten(), bins=50, alpha=0.5, label="Offset")
    data_flat = data.numpy().flatten()
    plt.hist(data_flat, bins=50, alpha=0.5, label="Data")
    plt.title("Empirical Distribution", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xlabel("Intensity", fontsize=12)
    plt.legend()
    plt.yscale('log')  # Log scale for better visibility
    plt.tight_layout()
    plt.savefig(path / "offset-distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.plot(offset_medians, 'o-', label="Offset Median")  # Add markers
    plt.title("Offset across Z", fontsize=12)
    plt.ylabel("Intensity", fontsize=12)
    plt.xlabel("Z-slice", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "offset-medians.png", dpi=300)
    plt.close()
