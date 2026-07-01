import os
import subprocess
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FILE = BASE_DIR / "fig1.png"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / "density_map.png"

GRID_CELLS = 100
SMOOTHING_SIGMA = 1.5


def detect_cells(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load {image_path}")
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = gray.shape

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    filled = binary_fill_holes(binary).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cell_mask = np.zeros_like(gray)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.drawContours(cell_mask, [contour], -1, 255, thickness=cv2.FILLED)

    ys, xs = np.where(cell_mask > 0)
    if len(xs) == 0:
        print("No cells detected.")
        return None, None, None

    print(f"Detected {len(xs)} cell pixels")
    print(f"Image dimensions: {img_width}w x {img_height}h pixels")
    return xs, ys, (img_width, img_height)


def compute_density(xs, ys, img_dims):
    img_width, img_height = img_dims

    x_edges = np.linspace(0, img_width, GRID_CELLS + 1)
    y_edges = np.linspace(0, img_height, GRID_CELLS + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    counts, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    density = counts.T
    density = gaussian_filter(density, sigma=SMOOTHING_SIGMA)

    if density.max() > 0:
        density = density / density.max()

    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    print(
        f"Grid: {GRID_CELLS}x{GRID_CELLS} squares  |  "
        f"Cell size: {img_width / GRID_CELLS:.1f}x{img_height / GRID_CELLS:.1f} px  |  "
        f"Density range: {density.min():.3f} to {density.max():.3f}"
    )
    return grid_x, grid_y, density


def plot_density(grid_x, grid_y, density, img_dims):
    img_width, img_height = img_dims

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        grid_x,
        grid_y,
        density,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )

    ax.set_title("3D Cell Density Heatmap", fontsize=17, pad=18)
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X (pixels)", labelpad=10)
    ax.set_ylabel("Y (pixels)", labelpad=10)
    ax.set_zlabel("Normalized Cell Density", labelpad=12)
    ax.set_xticks([0, img_width // 2, img_width])
    ax.set_yticks([0, img_height // 2, img_height])
    ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.view_init(elev=28, azim=-130)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.03)
    cbar.set_label("Normalized Cell Density", labelpad=12)

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

    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved density map to: {OUTPUT_FILE}")


def open_output():
    result = subprocess.run(["open", str(OUTPUT_FILE)], check=False)
    if result.returncode != 0:
        subprocess.run(["open", "-R", str(OUTPUT_FILE)], check=False)
        print("Could not open the image automatically, so it was revealed in Finder.")


def main():
    print(f"Creating density map from: {IMAGE_FILE.name}")

    xs, ys, img_dims = detect_cells(IMAGE_FILE)
    if xs is None:
        print("Could not create density map.")
        return

    grid_x, grid_y, density = compute_density(xs, ys, img_dims)
    plot_density(grid_x, grid_y, density, img_dims)
    open_output()
    print("Done.")


if __name__ == "__main__":
    main()
