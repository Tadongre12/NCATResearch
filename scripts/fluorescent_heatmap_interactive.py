"""
Interactive fluorescent density heatmap.

Green fluorescence is converted into the z-axis density surface. Small
pink/red fluorescent puncta are detected separately and displayed as a
spread/NET-like overlay on top of the surface.
"""

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter


BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_FILE = BASE_DIR / "raw_data" / "fluorescent_integrin_net.png"
OUTPUT_FILE = BASE_DIR / "results" / "interactive_fluorescent_density_heatmap.html"
QA_FILE = BASE_DIR / "results" / "fluorescent_pink_detection_overlay.png"

GRID_CELLS = 170
DENSITY_SIGMA = 3.0
PINK_SIGMA = 2.2

# Approximate microscope frame inside the screenshot.
ROI_X0, ROI_Y0 = 32, 25
ROI_X1, ROI_Y1 = 826, 756

# Baked-in annotation text zones that should not be counted as signal.
ANNOTATION_ZONES = [
    (250, 585, 620, 640),  # magenta label text
    (615, 650, 820, 755),  # red label text
    (610, 495, 775, 545),  # green leader-cell label
]


def in_annotation_zone(cx, cy):
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in ANNOTATION_ZONES)


def load_image():
    image = np.array(Image.open(IMAGE_FILE).convert("RGB"))
    height, width = image.shape[:2]

    roi = np.zeros((height, width), dtype=bool)
    roi[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1] = True

    for x0, y0, x1, y1 in ANNOTATION_ZONES:
        roi[y0:y1, x0:x1] = False

    return image, roi, width, height


def build_green_density(image, roi):
    red = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    blue = image[:, :, 2].astype(np.float32)

    # Green fluorescence should be bright and green-dominant. Subtracting part
    # of the red/blue channels suppresses white labels, arrows, and border lines.
    signal = green - (0.35 * red) - (0.35 * blue)
    signal = np.where((signal > 18) & roi, signal, 0)
    signal = gaussian_filter(signal, sigma=5)

    if signal.max() > 0:
        signal = signal / signal.max()

    return signal


def detect_pink_spread(image, roi, density):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)

    red_or_magenta = ((hue < 35) | (hue > 158) | ((hue > 135) & (hue < 170)))
    mask = (red_or_magenta & (sat > 70) & (val > 55) & roi).astype(np.uint8) * 255

    total_labels, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros(mask.shape, dtype=np.uint8)
    spots = []

    for label_id in range(1, total_labels):
        x, y, w, h, area = stats[label_id]
        cx, cy = centers[label_id]

        if area < 1 or area > 260:
            continue
        if w > 30 or h > 30:
            continue
        if in_annotation_zone(cx, cy):
            continue

        kept[labels == label_id] = 255
        px = int(round(cx))
        py = int(round(cy))
        px = min(max(px, 0), density.shape[1] - 1)
        py = min(max(py, 0), density.shape[0] - 1)
        spots.append(
            {
                "x": round(float(cx), 1),
                "y": round(float(cy), 1),
                "area": int(area),
                "density": round(float(density[py, px]), 3),
            }
        )

    pink_field = gaussian_filter(kept.astype(np.float32), sigma=6)
    if pink_field.max() > 0:
        pink_field = pink_field / pink_field.max()

    return kept, pink_field, spots


def grid_average(values, width, height):
    y_edges = np.linspace(0, height, GRID_CELLS + 1)
    x_edges = np.linspace(0, width, GRID_CELLS + 1)

    yy, xx = np.indices(values.shape)
    weighted, _, _ = np.histogram2d(
        yy.ravel(), xx.ravel(), bins=[y_edges, x_edges], weights=values.ravel()
    )
    counts, _, _ = np.histogram2d(yy.ravel(), xx.ravel(), bins=[y_edges, x_edges])
    grid = np.divide(weighted, counts, out=np.zeros_like(weighted), where=counts > 0)

    return grid


def create_qa_overlay(image, pink_mask):
    overlay = Image.fromarray(image).convert("RGBA")
    highlight = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(highlight)

    ys, xs = np.where(pink_mask > 0)
    for x, y in zip(xs, ys):
        draw.point((int(x), int(y)), fill=(255, 0, 190, 220))

    combined = Image.alpha_composite(overlay, highlight)
    combined.save(QA_FILE)


def encode_source_image():
    with open(IMAGE_FILE, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")


def write_html(x_values, y_values, density_grid, pink_grid, spots, width, height):
    spots = sorted(spots, key=lambda spot: spot["density"], reverse=True)
    spot_x = [spot["x"] for spot in spots]
    spot_y = [spot["y"] for spot in spots]
    spot_area = [spot["area"] for spot in spots]
    spot_density = [spot["density"] for spot in spots]
    spot_z = [
        min(1.06, max(0.06, spot["density"] + 0.07))
        for spot in spots
    ]

    source_b64 = encode_source_image()
    mean_density = round(float(np.mean(spot_density)), 3) if spot_density else 0
    first_x = spot_x[0] if spot_x else width / 2
    first_y = spot_y[0] if spot_y else height / 2
    first_z = spot_z[0] if spot_z else 0.5

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interactive Fluorescent Density Heatmap</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      font-family: Arial, sans-serif;
      background: #ffffff;
      color: #1f2933;
    }}
    #plot {{
      position: fixed;
      inset: 0;
    }}
    .panel {{
      position: fixed;
      right: 16px;
      top: 68px;
      width: min(330px, calc(100vw - 32px));
      padding: 12px 14px;
      border: 1px solid rgba(31,41,51,0.18);
      background: rgba(255,255,255,0.92);
      box-shadow: 0 8px 28px rgba(31,41,51,0.16);
      font-size: 13px;
      line-height: 1.35;
    }}
    .panel b {{
      color: #c2188f;
    }}
    .controls {{
      display: grid;
      gap: 8px;
      margin: 10px 0;
    }}
    button {{
      width: 100%;
      min-height: 34px;
      border: 1px solid rgba(31,41,51,0.28);
      background: #ffffff;
      color: #1f2933;
      font-size: 13px;
    }}
    button {{
      cursor: pointer;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .readout {{
      border-top: 1px solid rgba(31,41,51,0.18);
      border-bottom: 1px solid rgba(31,41,51,0.18);
      padding: 8px 0;
      margin: 8px 0;
    }}
    .thumb {{
      width: 100%;
      margin-top: 8px;
      border: 1px solid rgba(31,41,51,0.18);
      display: block;
    }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <div class="panel">
    <div><b>Pink spot finder:</b> {len(spots)} pink/red spots detected.</div>
    <div>Use the buttons below. The selected spot becomes a large yellow marker on the heatmap.</div>
    <div class="controls">
      <div class="row">
        <button id="focusButton" type="button">Strongest spot</button>
        <button id="nextButton" type="button">Next spot</button>
      </div>
      <div class="row">
        <button id="pinkButton" type="button">Pink only</button>
        <button id="allButton" type="button">All</button>
      </div>
      <div class="row">
        <button id="topButton" type="button">Top-down</button>
        <button id="resetButton" type="button">Reset view</button>
      </div>
    </div>
    <div class="readout" id="spotReadout"></div>
    <div>Mean local density around pink signal: {mean_density}</div>
    <img class="thumb" alt="source fluorescence image" src="data:image/png;base64,{source_b64}">
  </div>
  <script>
    const xValues = {json.dumps(x_values)};
    const yValues = {json.dumps(y_values)};
    const densityZ = {json.dumps(density_grid)};
    const pinkField = {json.dumps(pink_grid)};
    const spots = {json.dumps(spots)};
    const spotX = {json.dumps(spot_x)};
    const spotY = {json.dumps(spot_y)};
    const spotZ = {json.dumps(spot_z)};
    const spotArea = {json.dumps(spot_area)};
    const spotDensity = {json.dumps(spot_density)};

    const pinkFloor = pinkField.map(row => row.map(value => value > 0.08 ? 0.025 : null));

    const data = [
      {{
        type: "surface",
        name: "Green fluorescence density",
        x: xValues,
        y: yValues,
        z: densityZ,
        colorscale: "Jet",
        cmin: 0,
        cmax: 1,
        colorbar: {{ title: "Density", thickness: 18, len: 0.65 }},
        contours: {{
          z: {{ show: true, usecolormap: true, highlightcolor: "white", project: {{ z: true }} }}
        }},
        hovertemplate: "x=%{{x:.0f}} px<br>y=%{{y:.0f}} px<br>density=%{{z:.3f}}<extra></extra>"
      }},
      {{
        type: "surface",
        name: "Pink spread footprint",
        x: xValues,
        y: yValues,
        z: pinkFloor,
        surfacecolor: pinkField,
        colorscale: [
          [0.00, "rgba(0,0,0,0)"],
          [0.30, "#5e0047"],
          [0.68, "#ff2bbb"],
          [1.00, "#ffd1f3"]
        ],
        showscale: false,
        opacity: 0.72,
        hoverinfo: "skip"
      }},
      {{
        type: "scatter3d",
        mode: "markers",
        name: "Pink / NET-like puncta",
        x: spotX,
        y: spotY,
        z: spotZ,
        customdata: spotArea.map((area, i) => [area, spotDensity[i]]),
        marker: {{
          color: "#ff37cf",
          size: 4,
          symbol: "circle",
          line: {{ color: "#ffffff", width: 1 }},
          opacity: 0.95
        }},
        hovertemplate: "pink signal<br>x=%{{x:.1f}} px<br>y=%{{y:.1f}} px<br>local density=%{{customdata[1]:.3f}}<extra></extra>"
      }},
      {{
        type: "scatter3d",
        mode: "markers+text",
        name: "Selected pink spot",
        x: [{first_x}],
        y: [{first_y}],
        z: [{first_z}],
        text: ["selected pink spot"],
        textposition: "top center",
        marker: {{
          color: "#ffe600",
          size: 11,
          symbol: "diamond",
          line: {{ color: "#000000", width: 2 }},
          opacity: 1
        }},
        hovertemplate: "selected pink spot<br>x=%{{x:.1f}} px<br>y=%{{y:.1f}} px<extra></extra>"
      }}
    ];

    const layout = {{
      title: {{ text: "Interactive Fluorescent Density Heatmap with Pink Spread Overlay", x: 0.5 }},
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {{ color: "#1f2933" }},
      margin: {{ l: 0, r: 0, t: 54, b: 0 }},
      legend: {{ x: 0.02, y: 0.98, bgcolor: "rgba(255,255,255,0.75)" }},
      scene: {{
        xaxis: {{ title: "X (pixels)", range: [0, {width}], backgroundcolor: "#f6f8fa", gridcolor: "#d0d7de", zerolinecolor: "#8c959f" }},
        yaxis: {{ title: "Y (pixels)", range: [{height}, 0], backgroundcolor: "#f6f8fa", gridcolor: "#d0d7de", zerolinecolor: "#8c959f" }},
        zaxis: {{ title: "Density", range: [0, 1.08], backgroundcolor: "#f6f8fa", gridcolor: "#d0d7de", zerolinecolor: "#8c959f" }},
        camera: {{ eye: {{ x: 1.45, y: -1.65, z: 1.05 }} }},
        aspectmode: "manual",
        aspectratio: {{ x: {width / height:.4f}, y: 1, z: 0.46 }}
      }},
      updatemenus: [{{
        type: "buttons",
        direction: "right",
        x: 0.5,
        y: 1.03,
        xanchor: "center",
        buttons: [
          {{ label: "All overlays", method: "restyle", args: [{{ visible: [true, true, true, true] }}] }},
          {{ label: "Density only", method: "restyle", args: [{{ visible: [true, false, false, false] }}] }},
          {{ label: "Pink only", method: "restyle", args: [{{ visible: [false, true, true, true] }}] }}
        ]
      }}]
    }};

    const config = {{
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"]
    }};

    const spotReadout = document.getElementById("spotReadout");
    let selectedIndex = 0;

    function readoutText(index) {{
      const spot = spots[index];
      if (!spot) return "No pink spot selected.";
      return `Selected #${{index + 1}}: x=${{spot.x}} px, y=${{spot.y}} px, local density=${{spot.density}}.`;
    }}

    function focusSpot(index) {{
      const spot = spots[index];
      if (!spot) return;
      selectedIndex = index;
      const z = Math.min(1.06, Math.max(0.06, spot.density + 0.07));
      Plotly.restyle("plot", {{
        x: [[spot.x]],
        y: [[spot.y]],
        z: [[z]],
        text: [[`spot #${{index + 1}}`]]
      }}, [3]);
      spotReadout.textContent = readoutText(index);
      Plotly.restyle("plot", {{ visible: [true, true, true, true] }});
      Plotly.relayout("plot", {{
        "scene.camera.eye": {{ x: 1.15, y: -1.45, z: 0.86 }},
        "scene.camera.center": {{
          x: (spot.x - {width / 2}) / {width},
          y: ({height / 2} - spot.y) / {height},
          z: 0
        }}
      }});
    }}

    function nextSpot() {{
      if (!spots.length) return;
      focusSpot((selectedIndex + 1) % spots.length);
    }}

    function showAll() {{
      Plotly.restyle("plot", {{ visible: [true, true, true, true] }});
    }}

    function showPinkOnly() {{
      Plotly.restyle("plot", {{ visible: [false, true, true, true] }});
    }}

    function topDown() {{
      Plotly.relayout("plot", {{
        "scene.camera.eye": {{ x: 0, y: 0, z: 2.25 }},
        "scene.camera.up": {{ x: 0, y: 1, z: 0 }},
        "scene.camera.center": {{ x: 0, y: 0, z: 0 }}
      }});
    }}

    function resetView() {{
      Plotly.relayout("plot", {{
        "scene.camera.eye": {{ x: 1.45, y: -1.65, z: 1.05 }},
        "scene.camera.up": {{ x: 0, y: 0, z: 1 }},
        "scene.camera.center": {{ x: 0, y: 0, z: 0 }}
      }});
    }}

    document.getElementById("focusButton").addEventListener("click", () => focusSpot(0));
    document.getElementById("nextButton").addEventListener("click", nextSpot);
    document.getElementById("pinkButton").addEventListener("click", showPinkOnly);
    document.getElementById("allButton").addEventListener("click", showAll);
    document.getElementById("topButton").addEventListener("click", topDown);
    document.getElementById("resetButton").addEventListener("click", resetView);

    Plotly.newPlot("plot", data, layout, config).then(() => focusSpot(0));
  </script>
</body>
</html>
"""

    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    OUTPUT_FILE.write_text(html, encoding="utf-8")


def main():
    print(f"Generating fluorescent heatmap from {IMAGE_FILE}")
    image, roi, width, height = load_image()

    density = build_green_density(image, roi)
    pink_mask, pink_field, spots = detect_pink_spread(image, roi, density)

    density_grid = gaussian_filter(grid_average(density, width, height), sigma=DENSITY_SIGMA)
    if density_grid.max() > 0:
        density_grid = density_grid / density_grid.max()

    pink_grid = gaussian_filter(grid_average(pink_field, width, height), sigma=PINK_SIGMA)
    if pink_grid.max() > 0:
        pink_grid = pink_grid / pink_grid.max()

    x_edges = np.linspace(0, width, GRID_CELLS + 1)
    y_edges = np.linspace(0, height, GRID_CELLS + 1)
    x_values = ((x_edges[:-1] + x_edges[1:]) / 2).round(2).tolist()
    y_values = ((y_edges[:-1] + y_edges[1:]) / 2).round(2).tolist()

    create_qa_overlay(image, pink_mask)
    write_html(
        x_values,
        y_values,
        density_grid.round(4).tolist(),
        pink_grid.round(4).tolist(),
        spots,
        width,
        height,
    )

    print(f"Detected {len(spots)} pink/red puncta.")
    print(f"Saved interactive heatmap: {OUTPUT_FILE}")
    print(f"Saved pink detection QA overlay: {QA_FILE}")


if __name__ == "__main__":
    main()
