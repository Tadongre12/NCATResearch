import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, gaussian_filter

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_FILE = "fig1.png"  # replace with your actual filename

# ── GRID RESOLUTION ────────────────────────────────────────────────────────────
# Each grid cell represents a square region in the image.
# A smaller GRID_CELLS value = coarser grid (fewer, larger squares).
# A larger GRID_CELLS value = finer grid (more, smaller squares).
# Tune this so each square is roughly one cell diameter.
GRID_CELLS = 100  # number of grid divisions along each axis


def detect_cells(image_path):
    """
    Load image, threshold, fill holes, filter noise, return pixel coords.
    Returns (xs, ys, (img_width, img_height)) or (None, None, None) on failure.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load {image_path}")
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = gray.shape

    # Otsu thresholding — cells (dark blobs) → white in binary mask
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Fill any holes inside detected regions
    filled = binary_fill_holes(binary).astype(np.uint8) * 255

    # Find and filter contours (remove noise < 100 px²)
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


def compute_grid_displacement(xs, ys, img_dims, grid_cells=GRID_CELLS):
    """
    Divide the image into a grid_cells × grid_cells grid.
    For every cell (grid square) that contains at least one detected pixel,
    compute:
        Z = Euclidean distance from that grid square's center to the image center
    Returns:
        grid_x, grid_y  — 2-D coordinate arrays (pixel coords of grid centers)
        Z_disp          — 2-D displacement array (same shape as grid)
        cx, cy          — image center in pixels
    """
    img_width, img_height = img_dims

    # Image center point (pixels)
    cx = img_width / 2.0
    cy = img_height / 2.0

    # Build the grid: edges and center coordinates for each bin
    x_edges = np.linspace(0, img_width, grid_cells + 1)
    y_edges = np.linspace(0, img_height, grid_cells + 1)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2   # shape: (grid_cells,)
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2   # shape: (grid_cells,)

    # Count cell pixels per grid square
    counts, _, _ = np.histogram2d(
        xs, ys,
        bins=[x_edges, y_edges]
    )
    # counts shape: (grid_cells, grid_cells) — axis 0 = X, axis 1 = Y

    # 2-D coordinate meshgrid (pixel coords)
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    # grid_x / grid_y shape: (grid_cells, grid_cells)

    # Distance from each grid center to the image center
    dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)

    # Z = displacement only where cells were detected; elsewhere = NaN
    # (transpose counts to align with meshgrid orientation: rows=Y, cols=X)
    has_cells = counts.T > 0
    Z_disp = np.where(has_cells, dist, np.nan)

    print(f"Grid: {grid_cells}×{grid_cells} squares  |  "
          f"Cell size: {img_width/grid_cells:.1f}×{img_height/grid_cells:.1f} px  |  "
          f"Image center: ({cx:.0f}, {cy:.0f}) px")

    return grid_x, grid_y, Z_disp, cx, cy


def plot_displacement(grid_x, grid_y, Z_disp, img_dims, cx, cy):
    """
    3-D surface of displacement-from-center + 2-D inset top-down view.
    Saves to OUTPUT_DIR/displacement_map.png.
    """
    img_width, img_height = img_dims

    # Replace NaN with 0 for surface plotting (gaps appear as flat floor)
    Z_plot = np.nan_to_num(Z_disp, nan=0.0)
    max_disp = np.nanmax(Z_disp) if not np.all(np.isnan(Z_disp)) else 1.0

    fig = plt.figure(figsize=(13, 8))

    # ── 3-D surface ────────────────────────────────────────────────────────────
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        grid_x, grid_y, Z_plot,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        alpha=0.92
    )

    # Mark the center point on the floor
    ax.scatter([cx], [cy], [0],
               color="lime", s=80, zorder=5, label=f"Center ({cx:.0f}, {cy:.0f})")

    ax.set_title("3D Cell Displacement from Image Center", fontsize=17, pad=18)
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.set_zlim(0, max_disp * 1.05)
    ax.set_xlabel("X (pixels)", labelpad=10)
    ax.set_ylabel("Y (pixels)", labelpad=10)
    ax.set_zlabel("Displacement from Center (px)", labelpad=12)

    ax.set_xticks([0, img_width // 2, img_width])
    ax.set_yticks([0, img_height // 2, img_height])
    z_ticks = np.linspace(0, max_disp, 5)
    ax.set_zticks(np.round(z_ticks, 0))

    ax.view_init(elev=28, azim=-130)
    ax.legend(loc="upper left", fontsize=9)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.03)
    cbar.set_label("Distance from Center (pixels)", labelpad=12)

    # ── 2-D inset: top-down view ───────────────────────────────────────────────
    inset = fig.add_axes([0.80, 0.74, 0.14, 0.14])
    im = inset.imshow(
        Z_disp,
        origin="lower",
        extent=[0, img_width, 0, img_height],
        cmap="jet",
        vmin=0,
        vmax=max_disp,
        aspect="auto"
    )
    # Mark center on inset
    inset.plot(cx, cy, "g+", markersize=8, markeredgewidth=1.5)
    inset.set_xticks([0, img_width])
    inset.set_yticks([0, img_height])
    inset.tick_params(labelsize=7)
    inset.set_title("Top-down", fontsize=7, pad=2)

    fig.subplots_adjust(left=0.06, right=0.80, top=0.93, bottom=0.07)

    save_path = f"{OUTPUT_DIR}/displacement_map.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved displacement map to: {save_path}")


def main():
    print(f"Analyzing: {IMAGE_FILE}")

    xs, ys, img_dims = detect_cells(IMAGE_FILE)
    if xs is None:
        print("Could not create displacement map.")
        return

    grid_x, grid_y, Z_disp, cx, cy = compute_grid_displacement(
        xs, ys, img_dims, grid_cells=GRID_CELLS
    )

    plot_displacement(grid_x, grid_y, Z_disp, img_dims, cx, cy)
    print("Done.")


if __name__ == "__main__":
    main()