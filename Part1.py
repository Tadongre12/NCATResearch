import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_FILE = "fig1.png"


def detect_red_region(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not load {image_path}")
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    red = img_rgb[:, :, 0]
    green = img_rgb[:, :, 1]
    blue = img_rgb[:, :, 2]

    mask = (
        (red > 120) &
        (green < 150) &
        (blue < 150)
    )

    ys, xs = np.where(mask)

    if len(xs) == 0:
        print("No red cell region detected.")
        return None, None

    return xs, ys


def create_paper_style_heatmap(xs, ys):
    bins = 100

    x_norm = 100 * (xs - xs.min()) / (xs.max() - xs.min())
    y_norm = 100 * (ys - ys.min()) / (ys.max() - ys.min())

    heatmap, _, _ = np.histogram2d(
        x_norm,
        y_norm,
        bins=bins,
        range=[[0, 100], [0, 100]]
    )

    heatmap = gaussian_filter(heatmap, sigma=5)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    X, Y = np.meshgrid(
        np.linspace(0, 100, bins),
        np.linspace(0, 100, bins)
    )

    Z = heatmap.T

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="jet",
        linewidth=0,
        antialiased=True
    )

    ax.set_title("3D Cell Migration Heat Map", fontsize=18, pad=18)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 1)

    ax.set_xlabel("X Position", labelpad=10)
    ax.set_ylabel("Y Position", labelpad=10)
    ax.set_zlabel("Normalized Density", labelpad=12)

    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_zticks([0, 0.5, 1.0])

    ax.view_init(elev=25, azim=-135)

    cbar = fig.colorbar(
        surface,
        ax=ax,
        shrink=0.60,
        pad=0.03
    )

    cbar.set_label("Normalized Density", labelpad=12)

    inset = fig.add_axes([
        0.82,
        0.76,
        0.12,
        0.12
    ])

    inset.imshow(
        Z,
        origin="lower",
        extent=[0, 100, 0, 100],
        cmap="jet",
        vmin=0,
        vmax=1
    )

    inset.set_xticks([0, 100])
    inset.set_yticks([0, 100])
    inset.set_xlim(0, 100)
    inset.set_ylim(0, 100)
    inset.tick_params(labelsize=8)

    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        top=0.92,
        bottom=0.08
    )

    save_path = f"{OUTPUT_DIR}/paper_style_3d_heatmap.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved heat map to: {save_path}")


def main():
    print(f"Analyzing: {IMAGE_FILE}")

    xs, ys = detect_red_region(IMAGE_FILE)

    if xs is None:
        print("Could not create heat map.")
        return

    create_paper_style_heatmap(xs, ys)

    print("Done.")


if __name__ == "__main__":
    main()