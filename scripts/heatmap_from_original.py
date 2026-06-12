"""
Cancer Cell Migration Heatmap Generator
Works directly from raw microscope images — no pre-recognized images needed.

Pipeline:
  original image → grayscale → Otsu threshold → morphological cleanup
  → hole fill → Gaussian density → 3D surface + 2D inset

Usage (run from your Cancer_Research folder):
  python scripts/heatmap_from_original.py

Install dependencies once:
  pip install opencv-python numpy matplotlib scipy pillow
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.ndimage import gaussian_filter
from PIL import Image
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_PAIRS = [
    {
        "original":  "raw_data/microscope_images/original_0h.png",
        "timepoint": "0 h",
        "sigma": 12,      # smoothing: larger = broader peak
    },
    {
        "original":  "raw_data/microscope_images/original_96h.png",
        "timepoint": "96 h",
        "sigma": 6,
    },
    # Add more timepoints here:
    # { "original": "raw_data/microscope_images/original_24h.png", "timepoint": "24 h", "sigma": 10 },
    # { "original": "raw_data/microscope_images/original_48h.png", "timepoint": "48 h", "sigma": 8  },
]

OUTPUT_DIR = "outputs/heatmaps"
GRID_SIZE  = 100     # density map resolution (paper uses 100×100)
MIN_CELL_AREA = 100  # minimum pixel area to count as a cell (filters noise)

# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def segment_cells(image_path: str) -> np.ndarray:
    """
    Extract binary cell mask from a raw brightfield microscope image.
    Cells appear darker than the background in brightfield.
    Returns float32 array: 1.0 = cell, 0.0 = background.
    """
    # Load as grayscale
    img = np.array(Image.open(image_path).convert("L"))

    # Step 1: Gaussian blur — removes camera noise before thresholding
    blurred = cv2.GaussianBlur(img, (7, 7), 2)

    # Step 2: Otsu thresholding — automatically finds the best threshold
    # THRESH_BINARY_INV: dark pixels (cells) become white (1), background becomes black (0)
    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 3: Morphological closing — closes small gaps between cell borders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,  kernel, iterations=3)
    dilated = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel, iterations=2)

    # Step 4: Flood-fill from corner to find true background,
    # then invert to fill holes inside cell clusters
    filled = dilated.copy()
    h, w   = filled.shape
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, ffmask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    result     = dilated | filled_inv

    # Step 5: Keep only contours larger than MIN_CELL_AREA (removes dust/noise)
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(result)
    kept  = 0
    for c in contours:
        if cv2.contourArea(c) > MIN_CELL_AREA:
            cv2.drawContours(final, [c], -1, 255, -1)
            kept += 1

    print(f"  Contours detected: {len(contours)}, kept after size filter: {kept}")
    print(f"  Cell pixels: {(final > 0).sum()}")

    return (final > 0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DENSITY MAP
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_density(mask: np.ndarray,
                    grid_size: int = GRID_SIZE,
                    sigma: float = 10) -> np.ndarray:
    """
    Resize binary mask → grid_size × grid_size,
    apply Gaussian smoothing → continuous probability-density surface,
    normalise to [0, 1].
    """
    small   = cv2.resize(mask, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    density = gaussian_filter(small, sigma=sigma)
    dmax    = density.max()
    if dmax > 0:
        density /= dmax
    return density


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def make_heatmap_figure(density: np.ndarray,
                        timepoint: str,
                        out_path: str):
    """
    Publication-style figure matching Figure 3D of Huang et al. 2023:
      • 3D surface plot (jet colormap)
      • 2D top-down inset (top-right)
      • Colorbars for both
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
    ax3d.view_init(elev=28, azim=-55)

    cbar3d = fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=12,
                          pad=0.0, location="left")
    cbar3d.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar3d.ax.tick_params(labelsize=7)

    ax3d.text2D(0.02, 0.95, timepoint, transform=ax3d.transAxes,
                fontsize=11, fontweight="bold", va="top")

    # ── 2D inset ──────────────────────────────────────────────────────────────
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Cancer Cell Heatmap — from raw microscope images")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for entry in IMAGE_PAIRS:
        ori_path = entry["original"]
        tp       = entry["timepoint"]
        sigma    = entry.get("sigma", 10)
        tag      = tp.replace(" ", "")

        print(f"\n[{tp}]")

        if not os.path.exists(ori_path):
            print(f"  WARNING: {ori_path} not found — skipping.")
            continue

        # Segment cells from raw image
        mask = segment_cells(ori_path)

        if mask.sum() == 0:
            print("  WARNING: no cells detected. Try lowering MIN_CELL_AREA.")
            continue

        # Convert to density map
        density = mask_to_density(mask, grid_size=GRID_SIZE, sigma=sigma)
        print(f"  Gaussian sigma: {sigma}")

        # Generate figure
        out_path = os.path.join(OUTPUT_DIR, f"heatmap_{tag}.png")
        make_heatmap_figure(density, tp, out_path)

    print(f"\nDone. Outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()