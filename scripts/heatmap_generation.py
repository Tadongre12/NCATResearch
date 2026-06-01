"""
Cancer Cell Migration Heatmap Generator
Replicates Figure 3D from Huang et al. 2023 (Cell Reports Physical Science)

Pipeline:
  recognized image  →  binary cell mask  →  Gaussian density  →  3D surface + 2D inset

Usage (in VS Code terminal, from the folder containing your images):
  python heatmap_generation.py

Install dependencies once:
  pip install opencv-python numpy matplotlib scipy pillow
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – safe for VS Code run terminal
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.ndimage import gaussian_filter
from PIL import Image
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  –  edit these paths and parameters
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_PAIRS = [
    {
        "recognized": "raw_data/microscope_images/recognized_0h.png",
        "original":   "raw_data/microscope_images/original_0h.png",
        "timepoint":  "0 h",
        "sigma": 12,
    },
    {
        "recognized": "raw_data/microscope_images/recognized_96h.png",
        "original":   "raw_data/microscope_images/original_96h.png",
        "timepoint":  "96 h",
        "sigma": 6,
    },
]

OUTPUT_DIR = "outputs/heatmaps"   # folder where PNGs will be saved

GRID_SIZE = 100    # density map resolution (paper uses 100×100)

# Red-mask detection thresholds – tune if your cell overlay uses different colours
RED_R_MIN = 140    # minimum R value for a "cell" pixel
RED_G_MAX = 120    # maximum G value
RED_B_MAX = 120    # maximum B value

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def extract_cell_mask(recognized_path: str) -> np.ndarray:
    """
    Extract binary cell mask from a 'recognized' image where detected cells
    are coloured red by the upstream AI/segmentation algorithm.
    Returns float32 array: 1.0 = cell present, 0.0 = background.
    """
    img = np.array(Image.open(recognized_path).convert("RGB"))
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = (r > RED_R_MIN) & (g < RED_G_MAX) & (b < RED_B_MAX)
    return mask.astype(np.float32)


def mask_to_density(mask: np.ndarray,
                    grid_size: int = GRID_SIZE,
                    sigma: float = 10) -> np.ndarray:
    """
    Resize binary mask → grid_size × grid_size, apply Gaussian smoothing,
    normalise to [0, 1].
    This is the probability-density map described in the paper (Fig 3D).
    """
    small = cv2.resize(mask, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    density = gaussian_filter(small, sigma=sigma)
    dmax = density.max()
    if dmax > 0:
        density /= dmax
    return density


def make_heatmap_figure(density: np.ndarray,
                        timepoint: str,
                        out_path: str):
    """
    Produce a publication-style figure matching Figure 3D of Huang 2023:
      • Large 3D surface (jet colormap, matching paper)
      • 2D top-down inset (top-right corner)
      • Colorbars for both panels
    """
    n = density.shape[0]
    x = np.linspace(0, n, n)
    y = np.linspace(0, n, n)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(7, 6), dpi=150)
    fig.patch.set_facecolor("white")

    # ── 3D surface ────────────────────────────────────────────────────────────
    ax3d = fig.add_axes([0.02, 0.02, 0.72, 0.92], projection="3d")
    cmap = cm.jet

    surf = ax3d.plot_surface(
        X, Y, density,
        cmap=cmap, linewidth=0, antialiased=True,
        rcount=100, ccount=100,
    )

    ax3d.set_xlim(0, n); ax3d.set_ylim(0, n); ax3d.set_zlim(0, 1)
    ax3d.set_xticks([0, n // 2, n])
    ax3d.set_yticks([0, n // 2, n])
    ax3d.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3d.tick_params(labelsize=7)

    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("lightgrey")
    ax3d.grid(True, color="lightgrey", linewidth=0.4)
    ax3d.view_init(elev=28, azim=-55)   # camera angle matching the paper

    cbar3d = fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=12,
                          pad=0.0, location="left")
    cbar3d.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar3d.ax.tick_params(labelsize=7)

    ax3d.text2D(0.02, 0.95, timepoint, transform=ax3d.transAxes,
                fontsize=11, fontweight="bold", va="top")

    # ── 2D inset (top-right) ──────────────────────────────────────────────────
    ax2d = fig.add_axes([0.70, 0.52, 0.26, 0.40])

    im2d = ax2d.imshow(density, cmap=cmap, origin="upper",
                       vmin=0, vmax=1, extent=[0, n, n, 0])
    ax2d.set_xlim(0, n); ax2d.set_ylim(n, 0)
    ax2d.set_xticks([0, n]); ax2d.set_yticks([0, n])
    ax2d.tick_params(labelsize=6)
    for spine in ax2d.spines.values():
        spine.set_linewidth(0.6)

    cbar2d = fig.colorbar(im2d, ax=ax2d, shrink=0.9, aspect=10, pad=0.04)
    cbar2d.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar2d.ax.tick_params(labelsize=6)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Cancer Cell Migration Heatmap Generator")
    print("  Huang et al. 2023 (Cell Reports) style")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for entry in IMAGE_PAIRS:
        rec_path = entry["recognized"]
        tp       = entry["timepoint"]
        sigma    = entry.get("sigma", 10)
        tag      = tp.replace(" ", "")

        print(f"\n[{tp}]")

        if not os.path.exists(rec_path):
            print(f"  WARNING: {rec_path} not found – skipping.")
            continue

        # Step 1: extract binary cell mask
        mask = extract_cell_mask(rec_path)
        n_cell_px = int(mask.sum())
        print(f"  Cell pixels detected : {n_cell_px}")

        if n_cell_px == 0:
            print("  WARNING: no red pixels found. Check RED_R_MIN / RED_G_MAX / RED_B_MAX thresholds.")
            continue

        # Step 2: convert to Gaussian density map
        density = mask_to_density(mask, grid_size=GRID_SIZE, sigma=sigma)
        print(f"  Density range        : [{density.min():.3f}, {density.max():.3f}]")
        print(f"  Gaussian sigma       : {sigma}")

        # Step 3: generate figure
        out_path = os.path.join(OUTPUT_DIR, f"heatmap_{tag}.png")
        make_heatmap_figure(density, tp, out_path)

    print(f"\nDone. Outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
