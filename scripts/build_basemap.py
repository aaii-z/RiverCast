"""Stitch OpenStreetMap tiles into light and dark basemaps aligned with the web map."""

import argparse
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
WIDTH, HEIGHT, SCALE, ZOOM, TILE = 640, 380, 2, 12, 256
LIGHT_BG, DARK_BG = np.array([0xF4, 0xF7, 0xF6]) / 255, np.array([0x0D, 0x16, 0x16]) / 255


def view_lonlat(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat of every output pixel, using the same projection as web/index.html."""
    nodes = [[n["lon"], n["lat"]] for n in data["nodes"]]
    pts = [p for line in data["rivers"] for p in line] + nodes
    kx = math.cos(math.radians(52.6))
    xs, ys = [p[0] * kx for p in pts], [p[1] for p in pts]
    x0, y1 = min(xs), max(ys)
    s = min(600 / (max(xs) - x0), 340 / (y1 - min(ys)))
    px = (np.arange(WIDTH * SCALE) + 0.5) / SCALE
    py = (np.arange(HEIGHT * SCALE) + 0.5) / SCALE
    X, Y = np.meshgrid(px, py)
    return ((X - 20) / s + x0) / kx, y1 - (Y - 20) / s


def tile(x: int, y: int, cache: Path) -> Image.Image:
    path = cache / f"{ZOOM}_{x}_{y}.png"
    if not path.exists():
        r = requests.get(TILE_URL.format(z=ZOOM, x=x, y=y), timeout=30,
                         headers={"User-Agent": "RiverCast/0.1 (static research basemap)"})
        r.raise_for_status()
        path.write_bytes(r.content)
        time.sleep(0.1)
    return Image.open(io.BytesIO(path.read_bytes())).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="web/forecast_data.json")
    parser.add_argument("--out", default="web")
    parser.add_argument("--cache", default="data/raw/tiles")
    args = parser.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    lon, lat = view_lonlat(json.loads(Path(args.data).read_text()))
    n = 2**ZOOM
    tx = (lon + 180) / 360 * n
    ty = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / math.pi) / 2 * n
    xs = range(int(tx.min()), int(tx.max()) + 1)
    ys = range(int(ty.min()), int(ty.max()) + 1)

    mosaic = Image.new("RGB", (len(xs) * TILE, len(ys) * TILE))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            mosaic.paste(tile(x, y, cache), (i * TILE, j * TILE))
    m = np.asarray(mosaic)
    rows = np.clip(((ty - ys[0]) * TILE).astype(int), 0, m.shape[0] - 1)
    cols = np.clip(((tx - xs[0]) * TILE).astype(int), 0, m.shape[1] - 1)
    rgb = m[rows, cols] / 255

    soft = rgb * 0.55 + (rgb @ [0.299, 0.587, 0.114])[..., None] * 0.45  # muted so data stands out
    inverted = 1 - soft
    images = {
        "light": soft * 0.85 + LIGHT_BG * 0.15,
        "dark": DARK_BG + (inverted - inverted.min()) * 0.55 * np.array([0.85, 1.0, 1.0]),
    }
    for name, img in images.items():
        path = Path(args.out) / f"basemap_{name}.jpg"
        pixels = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(path, quality=82, optimize=True)
        print(path)


if __name__ == "__main__":
    main()
