"""
3D density heatmap for Picture1.png.
X and Y axes show actual image pixel coordinates.
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

IMAGE_PATH = "Picture1.png"
OUTPUT_PATH = "results/heatmap_picture1_3d.png"
SIGMA = 25          # Gaussian smoothing radius in pixels
CELL_THRESHOLD = 150  # pixels darker than this are cells; brighter = background


def load_density(image_path: str) -> tuple[np.ndarray, int, int]:
    """
    Hard-threshold: only pixels darker than CELL_THRESHOLD count as cells.
    Background noise (gray ~170-200) is zeroed out entirely.
    """
    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    h, w = img.shape
    print(f"  Image size: {w} x {h} px")

    # Cell mask: 1 where pixel is darker than threshold, 0 elsewhere
    cell_mask = (img < CELL_THRESHOLD).astype(np.float32)
    # Weight by how dark the pixel is within the cell region
    darkness = np.where(cell_mask, CELL_THRESHOLD - img, 0.0)
    darkness /= darkness.max()
    print(f"  Cell pixels: {int(cell_mask.sum())} / {w*h} ({100*cell_mask.mean():.1f}%)")
    return darkness, w, h


def make_3d_heatmap(raw_density: np.ndarray, img_w: int, img_h: int, out_path: str):
    density = gaussian_filter(raw_density, sigma=SIGMA)
    dmax = density.max()
    if dmax > 0:
        density /= dmax

    h, w = density.shape
    # Downsample for speed while keeping aspect ratio
    step = max(1, min(h, w) // 200)
    ds = density[::step, ::step]
    dh, dw = ds.shape

    # Pixel-coordinate axes
    x = np.arange(dw) * step   # columns → X
    y = np.arange(dh) * step   # rows    → Y
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 7), dpi=150)
    fig.patch.set_facecolor("white")

    ax3d = fig.add_axes([0.02, 0.02, 0.72, 0.94], projection="3d")
    cmap = cm.jet

    surf = ax3d.plot_surface(
        X, Y, ds,
        cmap=cmap, linewidth=0, antialiased=True,
        rcount=150, ccount=150,
    )

    ax3d.set_xlim(0, img_w); ax3d.set_ylim(0, img_h); ax3d.set_zlim(0, 1)
    ax3d.set_xlabel("X (pixels)", fontsize=9, labelpad=8)
    ax3d.set_ylabel("Y (pixels)", fontsize=9, labelpad=8)
    ax3d.set_zlabel("Density", fontsize=9, labelpad=6)

    # Tick marks at round pixel values
    xticks = np.linspace(0, img_w, 5, dtype=int)
    yticks = np.linspace(0, img_h, 5, dtype=int)
    ax3d.set_xticks(xticks); ax3d.set_xticklabels(xticks)
    ax3d.set_yticks(yticks); ax3d.set_yticklabels(yticks)
    ax3d.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
    ax3d.tick_params(labelsize=7)

    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("lightgrey")
    ax3d.grid(True, color="lightgrey", linewidth=0.4)
    ax3d.view_init(elev=28, azim=-55)

    cbar = fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=12,
                        pad=0.0, location="left")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7)

    ax3d.set_title("Cell Density — Picture1", fontsize=12, fontweight="bold", pad=12)

    # 2D top-down inset
    ax2d = fig.add_axes([0.70, 0.52, 0.26, 0.40])
    im2d = ax2d.imshow(density, cmap=cmap, origin="upper",
                       vmin=0, vmax=1, extent=[0, img_w, img_h, 0])
    ax2d.set_xlabel("X (px)", fontsize=7)
    ax2d.set_ylabel("Y (px)", fontsize=7)
    xt2 = np.linspace(0, img_w, 3, dtype=int)
    yt2 = np.linspace(0, img_h, 3, dtype=int)
    ax2d.set_xticks(xt2); ax2d.set_yticks(yt2)
    ax2d.tick_params(labelsize=6)
    fig.colorbar(im2d, ax=ax2d, shrink=0.9, aspect=10, pad=0.04).ax.tick_params(labelsize=6)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    print("Generating 3D density heatmap for Picture1.png …")
    raw_density, img_w, img_h = load_density(IMAGE_PATH)
    make_3d_heatmap(raw_density, img_w, img_h, OUTPUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
