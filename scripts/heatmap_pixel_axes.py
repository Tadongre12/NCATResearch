import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, gaussian_filter

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_FILE = "Picture1.png"  # replace with your actual filename


def detect_cells(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load {image_path}")
        return None, None, None

    # Convert to grayscale (correct approach for a B&W microscope image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_height, img_width = gray.shape

    # Otsu thresholding: automatically finds the best threshold value
    # THRESH_BINARY_INV makes cells (dark blobs) become white (255) in the mask
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Fill holes inside detected cell regions
    filled = binary_fill_holes(binary).astype(np.uint8) * 255

    # Find contours of each detected blob
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out tiny blobs (noise) — only keep blobs larger than 100 pixels
    cell_mask = np.zeros_like(gray)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.drawContours(cell_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Get the pixel coordinates of all detected cell pixels
    ys, xs = np.where(cell_mask > 0)

    if len(xs) == 0:
        print("No cells detected.")
        return None, None, None

    print(f"Detected {len(xs)} cell pixels")
    print(f"Image dimensions: {img_width}w x {img_height}h pixels")

    return xs, ys, (img_width, img_height)


def create_heatmap(xs, ys, img_dims):
    img_width, img_height = img_dims

    # 200x200 bins keeps axes in real pixel coordinates
    # without creating 2 million spiky 1-pixel bins
    BINS = 200

    heatmap, _, _ = np.histogram2d(
        xs,
        ys,
        bins=[BINS, BINS],
        range=[[0, img_width], [0, img_height]]
    )

    # Mild smoothing to make the surface readable
    heatmap = gaussian_filter(heatmap, sigma=3)

    # Normalize density to 0-1
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Coordinate grids still use real pixel coordinates on the axes
    X, Y = np.meshgrid(
        np.linspace(0, img_width, BINS),
        np.linspace(0, img_height, BINS)
    )

    Z = heatmap.T  # transpose to align with meshgrid orientation

    # --- 3D surface plot ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        X, Y, Z,
        cmap="jet",
        linewidth=0,
        antialiased=True
    )

    ax.set_title("3D Cell Distribution Heatmap", fontsize=18, pad=18)

    # Axes in actual pixel coordinates
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.set_zlim(0, 1)

    ax.set_xlabel("X (pixels)", labelpad=10)
    ax.set_ylabel("Y (pixels)", labelpad=10)
    ax.set_zlabel("Normalized Density", labelpad=12)

    # Tick marks spaced across the real pixel range
    ax.set_xticks([0, img_width // 2, img_width])
    ax.set_yticks([0, img_height // 2, img_height])
    ax.set_zticks([0, 0.5, 1.0])

    ax.view_init(elev=25, azim=-135)

    cbar = fig.colorbar(surface, ax=ax, shrink=0.60, pad=0.03)
    cbar.set_label("Normalized Density", labelpad=12)

    # --- 2D inset (top-down view) ---
    inset = fig.add_axes([0.82, 0.76, 0.12, 0.12])
    inset.imshow(
        Z,
        origin="lower",
        extent=[0, img_width, 0, img_height],
        cmap="jet",
        vmin=0,
        vmax=1
    )
    inset.set_xticks([0, img_width])
    inset.set_yticks([0, img_height])
    inset.set_xlim(0, img_width)
    inset.set_ylim(0, img_height)
    inset.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.08)

    save_path = f"{OUTPUT_DIR}/heatmap_pixel_axes.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved heatmap to: {save_path}")


def main():
    print(f"Analyzing: {IMAGE_FILE}")
    xs, ys, img_dims = detect_cells(IMAGE_FILE)
    if xs is None:
        print("Could not create heatmap.")
        return
    create_heatmap(xs, ys, img_dims)
    print("Done.")


if __name__ == "__main__":
    main()