import subprocess
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FILE = BASE_DIR / "fig1.png"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / "displacement_map_vscode.png"

GRID_CELLS = 100


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


def compute_displacement(xs, ys, img_dims):
    img_width, img_height = img_dims
    center_x = img_width / 2.0
    center_y = img_height / 2.0

    x_edges = np.linspace(0, img_width, GRID_CELLS + 1)
    y_edges = np.linspace(0, img_height, GRID_CELLS + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    counts, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)

    distance_from_center = np.sqrt(
        (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    )
    has_cells = counts.T > 0
    displacement = np.where(has_cells, distance_from_center, np.nan)

    print(
        f"Grid: {GRID_CELLS}x{GRID_CELLS} squares  |  "
        f"Cell size: {img_width / GRID_CELLS:.1f}x{img_height / GRID_CELLS:.1f} px  |  "
        f"Image center: ({center_x:.0f}, {center_y:.0f}) px"
    )
    return grid_x, grid_y, displacement, center_x, center_y


def plot_displacement(grid_x, grid_y, displacement, img_dims, center_x, center_y):
    img_width, img_height = img_dims

    z_plot = np.nan_to_num(displacement, nan=0.0)
    max_displacement = (
        np.nanmax(displacement) if not np.all(np.isnan(displacement)) else 1.0
    )

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        grid_x,
        grid_y,
        z_plot,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )

    ax.scatter(
        [center_x],
        [center_y],
        [0],
        color="lime",
        s=80,
        zorder=5,
        label=f"Center ({center_x:.0f}, {center_y:.0f})",
    )

    ax.set_title("3D Cell Displacement from Image Center", fontsize=17, pad=18)
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.set_zlim(0, max_displacement * 1.05)
    ax.set_xlabel("X (pixels)", labelpad=10)
    ax.set_ylabel("Y (pixels)", labelpad=10)
    ax.set_zlabel("Displacement from Center (px)", labelpad=12)

    ax.set_xticks([0, img_width // 2, img_width])
    ax.set_yticks([0, img_height // 2, img_height])
    ax.set_zticks(np.round(np.linspace(0, max_displacement, 5), 0))
    ax.view_init(elev=28, azim=-130)
    ax.legend(loc="upper left", fontsize=9)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.03)
    cbar.set_label("Distance from Center (pixels)", labelpad=12)

    inset = fig.add_axes([0.80, 0.74, 0.14, 0.14])
    inset.imshow(
        displacement,
        origin="lower",
        extent=[0, img_width, 0, img_height],
        cmap="jet",
        vmin=0,
        vmax=max_displacement,
        aspect="auto",
    )
    inset.plot(center_x, center_y, "g+", markersize=8, markeredgewidth=1.5)
    inset.set_xticks([0, img_width])
    inset.set_yticks([0, img_height])
    inset.tick_params(labelsize=7)
    inset.set_title("Top-down", fontsize=7, pad=2)

    fig.subplots_adjust(left=0.06, right=0.80, top=0.93, bottom=0.07)

    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved displacement map to: {OUTPUT_FILE}")


def open_output():
    result = subprocess.run(["open", str(OUTPUT_FILE)], check=False)
    if result.returncode != 0:
        subprocess.run(["open", "-R", str(OUTPUT_FILE)], check=False)
        print("Could not open the image automatically, so it was revealed in Finder.")


def main():
    print(f"Creating displacement map from: {IMAGE_FILE.name}")

    xs, ys, img_dims = detect_cells(IMAGE_FILE)
    if xs is None:
        print("Could not create displacement map.")
        return

    grid_x, grid_y, displacement, center_x, center_y = compute_displacement(
        xs, ys, img_dims
    )
    plot_displacement(grid_x, grid_y, displacement, img_dims, center_x, center_y)
    open_output()
    print("Done.")


if __name__ == "__main__":
    main()
