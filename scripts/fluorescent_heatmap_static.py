"""
Static 3D density heatmap for the fluorescent microscope image.

This follows the same style as the earlier normal heatmaps, but uses green
fluorescence intensity as the density signal instead of grayscale darkness.
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_FILE = BASE_DIR / "raw_data" / "fluorescent_integrin_net.png"
OUTPUT_FILE = BASE_DIR / "results" / "fluorescent_density_heatmap.png"

GRID_CELLS = 150
SMOOTHING_SIGMA = 2.0

# Approximate inner microscope frame, excluding the screenshot border.
ROI_X0, ROI_Y0 = 32, 25
ROI_X1, ROI_Y1 = 826, 756

# Ignore the baked-in annotation labels so text is not counted as density.
ANNOTATION_ZONES = [
    (250, 585, 620, 640),
    (615, 650, 820, 755),
    (610, 495, 775, 545),
]


def detect_fluorescent_signal(image_path):
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    img_height, img_width = image.shape[:2]

    roi = np.zeros((img_height, img_width), dtype=bool)
    roi[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1] = True
    for x0, y0, x1, y1 in ANNOTATION_ZONES:
        roi[y0:y1, x0:x1] = False

    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    signal = green - (0.35 * red) - (0.35 * blue)
    signal = np.where((signal > 18) & roi, signal, 0)

    ys, xs = np.where(signal > 0)
    weights = signal[ys, xs]

    print(f"Detected {len(xs)} fluorescent signal pixels")
    print(f"Image dimensions: {img_width}w x {img_height}h pixels")
    return xs, ys, weights, (img_width, img_height)


def compute_density_grid(xs, ys, weights, img_dims):
    img_width, img_height = img_dims
    x_edges = np.linspace(0, img_width, GRID_CELLS + 1)
    y_edges = np.linspace(0, img_height, GRID_CELLS + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    density, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=weights)
    density = density.T
    density = gaussian_filter(density, sigma=SMOOTHING_SIGMA)

    if density.max() > 0:
        density = density / density.max()

    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    print(
        f"Grid: {GRID_CELLS}x{GRID_CELLS} squares | "
        f"Density range: {density.min():.3f} to {density.max():.3f}"
    )
    return grid_x, grid_y, density


def plot_density(grid_x, grid_y, density, img_dims):
    img_width, img_height = img_dims

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        grid_x,
        grid_y,
        density,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )

    ax.set_title("3D Fluorescent Density Heatmap", fontsize=17, pad=18)
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X (pixels)", labelpad=10)
    ax.set_ylabel("Y (pixels)", labelpad=10)
    ax.set_zlabel("Normalized Density", labelpad=12)

    ax.set_xticks([0, img_width // 2, img_width])
    ax.set_yticks([0, img_height // 2, img_height])
    ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.view_init(elev=28, azim=-130)

    cbar = fig.colorbar(surface, ax=ax, shrink=0.55, pad=0.03)
    cbar.set_label("Normalized Fluorescent Density", labelpad=12)

    inset = fig.add_axes([0.80, 0.74, 0.14, 0.14])
    inset.imshow(
        density,
        origin="lower",
        extent=[0, img_width, 0, img_height],
        cmap="jet",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    inset.set_xticks([0, img_width])
    inset.set_yticks([0, img_height])
    inset.tick_params(labelsize=7)
    inset.set_title("Top-down", fontsize=7, pad=2)

    fig.subplots_adjust(left=0.06, right=0.80, top=0.93, bottom=0.07)
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved density heatmap to: {OUTPUT_FILE}")


def main():
    print(f"Generating density heatmap from: {IMAGE_FILE.name}")
    xs, ys, weights, img_dims = detect_fluorescent_signal(IMAGE_FILE)
    grid_x, grid_y, density = compute_density_grid(xs, ys, weights, img_dims)
    plot_density(grid_x, grid_y, density, img_dims)
    print("Done.")


if __name__ == "__main__":
    main()
