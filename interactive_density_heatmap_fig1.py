import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FILE = BASE_DIR / "fig1.png"
OUTPUT_FILE = BASE_DIR / "results" / "interactive_density_heatmap_fig1.html"

GRID_CELLS = 120
SMOOTHING_SIGMA = 1.5


def detect_cells(image_path):
    img = cv2.imread(image_path)
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


def build_density(xs, ys, img_dims):
    img_width, img_height = img_dims
    x_edges = np.linspace(0, img_width, GRID_CELLS + 1)
    y_edges = np.linspace(0, img_height, GRID_CELLS + 1)
    x_centers = ((x_edges[:-1] + x_edges[1:]) / 2).round(2)
    y_centers = ((y_edges[:-1] + y_edges[1:]) / 2).round(2)

    counts, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    density = counts.T
    density = gaussian_filter(density, sigma=SMOOTHING_SIGMA)

    if density.max() > 0:
        density = density / density.max()

    print(
        f"Grid: {GRID_CELLS}x{GRID_CELLS} squares  |  "
        f"Density range: {density.min():.3f} to {density.max():.3f}"
    )
    return x_centers.tolist(), y_centers.tolist(), density.round(4).tolist()


def write_html(x_values, y_values, z_values, img_dims):
    img_width, img_height = img_dims

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interactive Density Heatmap - fig1.png</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      font-family: Arial, sans-serif;
      background: #f7f7f7;
    }}
    #plot {{
      width: 100vw;
      height: 100vh;
    }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const xValues = {json.dumps(x_values)};
    const yValues = {json.dumps(y_values)};
    const zValues = {json.dumps(z_values)};

    const data = [{{
      type: "surface",
      x: xValues,
      y: yValues,
      z: zValues,
      colorscale: "Jet",
      cmin: 0,
      cmax: 1,
      colorbar: {{
        title: "Density",
        thickness: 18,
        len: 0.72
      }},
      contours: {{
        z: {{
          show: true,
          usecolormap: true,
          highlightcolor: "white",
          project: {{ z: true }}
        }}
      }}
    }}];

    const layout = {{
      title: {{
        text: "Interactive Cell Density Heatmap - fig1.png",
        x: 0.5
      }},
      margin: {{ l: 0, r: 0, t: 52, b: 0 }},
      scene: {{
        xaxis: {{ title: "X (pixels)", range: [0, {img_width}] }},
        yaxis: {{ title: "Y (pixels)", range: [0, {img_height}] }},
        zaxis: {{ title: "Normalized Cell Density", range: [0, 1] }},
        camera: {{
          eye: {{ x: 1.45, y: -1.75, z: 1.05 }}
        }},
        aspectmode: "manual",
        aspectratio: {{ x: {img_width / img_height:.4f}, y: 1, z: 0.45 }}
      }}
    }};

    const config = {{
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"]
    }};

    Plotly.newPlot("plot", data, layout, config);
  </script>
</body>
</html>
"""

    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as html_file:
        html_file.write(html)
    output_path = OUTPUT_FILE.resolve()
    print(f"Saved interactive heatmap to: {output_path}")
    return output_path


def open_output(output_path):
    browser_targets = [
        "Google Chrome",
        "Safari",
        "/Applications/Google Chrome.app",
        "/Applications/Safari.app",
    ]
    for app_name in browser_targets:
        result = subprocess.run(
            ["open", "-a", app_name, str(output_path)],
            check=False,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            print(f"Opened in {app_name}.")
            return True

    subprocess.run(["open", "-R", str(output_path)], check=False)
    print("Could not open a browser automatically, so the file was revealed in Finder.")
    return False


def main():
    print(f"Generating interactive density heatmap from: {IMAGE_FILE.name}")
    xs, ys, img_dims = detect_cells(IMAGE_FILE)
    if xs is None:
        print("Could not create interactive density heatmap.")
        return

    x_values, y_values, z_values = build_density(xs, ys, img_dims)
    output_path = write_html(x_values, y_values, z_values, img_dims)
    did_open = open_output(output_path)
    if did_open:
        print("Done. The heatmap opened automatically.")
    else:
        print("Done. Open the HTML file above to view the heatmap.")
    print("Drag to rotate, scroll to zoom, and shift-drag to pan.")


if __name__ == "__main__":
    main()
