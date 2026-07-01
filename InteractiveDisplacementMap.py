import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FILE = BASE_DIR / "fig1.png"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / "interactive_displacement_map.html"

GRID_CELLS = 120


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


def build_displacement(xs, ys, img_dims):
    img_width, img_height = img_dims
    center_x = img_width / 2.0
    center_y = img_height / 2.0

    x_edges = np.linspace(0, img_width, GRID_CELLS + 1)
    y_edges = np.linspace(0, img_height, GRID_CELLS + 1)
    x_centers = ((x_edges[:-1] + x_edges[1:]) / 2).round(2)
    y_centers = ((y_edges[:-1] + y_edges[1:]) / 2).round(2)

    counts, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)

    displacement = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
    displacement = np.where(counts.T > 0, displacement, 0)

    max_displacement = displacement.max()
    print(
        f"Grid: {GRID_CELLS}x{GRID_CELLS} squares  |  "
        f"Image center: ({center_x:.0f}, {center_y:.0f}) px  |  "
        f"Max displacement: {max_displacement:.1f} px"
    )
    return (
        x_centers.tolist(),
        y_centers.tolist(),
        displacement.round(2).tolist(),
        center_x,
        center_y,
        max_displacement,
    )


def write_html(x_values, y_values, z_values, img_dims, center_x, center_y, max_z):
    img_width, img_height = img_dims

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interactive Displacement Map - fig1.png</title>
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

    const surface = {{
      type: "surface",
      x: xValues,
      y: yValues,
      z: zValues,
      colorscale: "Jet",
      cmin: 0,
      cmax: {max_z:.2f},
      colorbar: {{
        title: "Displacement (px)",
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
    }};

    const centerPoint = {{
      type: "scatter3d",
      mode: "markers+text",
      x: [{center_x:.2f}],
      y: [{center_y:.2f}],
      z: [0],
      text: ["Center"],
      textposition: "top center",
      marker: {{ size: 6, color: "lime" }},
      showlegend: false
    }};

    const layout = {{
      title: {{
        text: "Interactive Cell Displacement from Image Center - fig1.png",
        x: 0.5
      }},
      margin: {{ l: 0, r: 0, t: 52, b: 0 }},
      scene: {{
        xaxis: {{ title: "X (pixels)", range: [0, {img_width}] }},
        yaxis: {{ title: "Y (pixels)", range: [0, {img_height}] }},
        zaxis: {{ title: "Displacement from Center (px)", range: [0, {max_z * 1.05:.2f}] }},
        camera: {{
          eye: {{ x: 1.45, y: -1.75, z: 1.05 }}
        }},
        aspectmode: "manual",
        aspectratio: {{ x: {img_width / img_height:.4f}, y: 1, z: 0.55 }}
      }}
    }};

    const config = {{
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"]
    }};

    Plotly.newPlot("plot", [surface, centerPoint], layout, config);
  </script>
</body>
</html>
"""

    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Saved interactive displacement map to: {OUTPUT_FILE}")
    return OUTPUT_FILE


def open_output(output_path):
    for app_name in [
        "Google Chrome",
        "Safari",
        "/Applications/Google Chrome.app",
        "/Applications/Safari.app",
    ]:
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
    print(f"Creating interactive displacement map from: {IMAGE_FILE.name}")

    xs, ys, img_dims = detect_cells(IMAGE_FILE)
    if xs is None:
        print("Could not create interactive displacement map.")
        return

    x_values, y_values, z_values, center_x, center_y, max_z = build_displacement(
        xs, ys, img_dims
    )
    output_path = write_html(
        x_values, y_values, z_values, img_dims, center_x, center_y, max_z
    )
    did_open = open_output(output_path)
    if did_open:
        print("Done. The interactive displacement map opened automatically.")
    else:
        print("Done. Open the HTML file above to view the interactive map.")
    print("Drag to rotate, scroll to zoom, and shift-drag to pan.")


if __name__ == "__main__":
    main()
