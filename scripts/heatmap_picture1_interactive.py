"""
Interactive 3D density heatmap for Picture1.png.
Outputs an HTML file — open in any browser to rotate, zoom, and pan.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import os

IMAGE_PATH   = "Picture1.png"
OUTPUT_PATH  = "results/heatmap_picture1_interactive.html"
SIGMA        = 25
CELL_THRESHOLD = 150   # pixels darker than this are cells


def load_density(image_path):
    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    h, w = img.shape
    print(f"  Image size: {w} x {h} px")

    cell_mask = (img < CELL_THRESHOLD).astype(np.float32)
    darkness  = np.where(cell_mask, CELL_THRESHOLD - img, 0.0)
    darkness /= darkness.max()
    print(f"  Cell pixels: {int(cell_mask.sum())} ({100*cell_mask.mean():.1f}%)")
    return darkness, w, h


def main():
    print("Generating interactive 3D density heatmap …")
    raw, img_w, img_h = load_density(IMAGE_PATH)

    density = gaussian_filter(raw, sigma=SIGMA)
    density /= density.max()

    # Downsample so the browser stays responsive (~300 pts per side max)
    step = max(1, min(density.shape) // 300)
    ds   = density[::step, ::step]
    dh, dw = ds.shape

    x = np.arange(dw) * step   # pixel x-coords
    y = np.arange(dh) * step   # pixel y-coords

    fig = go.Figure(data=[
        go.Surface(
            z=ds,
            x=x,
            y=y,
            colorscale="Jet",
            cmin=0, cmax=1,
            colorbar=dict(
                title=dict(text="Density", side="right"),
                thickness=18,
                len=0.6,
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
            ),
        )
    ])

    fig.update_layout(
        title=dict(text="Cell Density — Picture1", x=0.5, font=dict(size=18)),
        scene=dict(
            xaxis=dict(title="X (pixels)", range=[0, img_w]),
            yaxis=dict(title="Y (pixels)", range=[0, img_h]),
            zaxis=dict(title="Density",    range=[0, 1]),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
            aspectmode="manual",
            aspectratio=dict(x=img_w/img_h, y=1, z=0.5),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        width=1100,
        height=750,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.write_html(OUTPUT_PATH, include_plotlyjs="cdn")
    print(f"  Saved → {OUTPUT_PATH}")
    print("  Open the HTML file in your browser — drag to rotate, scroll to zoom.")


if __name__ == "__main__":
    main()
