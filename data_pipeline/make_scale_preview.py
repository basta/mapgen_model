"""
Generate scale comparison images for docs/scale_comparison.md.

Answers the question: "Is 30m/pixel the right scale for fantasy maps?"

Each 256×256 model patch covers a different real-world area depending on
the effective pixel resolution. This script samples the same tile centre
at four scales by extracting progressively larger crops and downsampling
to 256×256, then shows the heightmap + derived labels at each scale.

Scales compared
  30m/px   →  7.7 km  (current: extract 256px, no downsample)
  90m/px   →  23 km   (extract 768px crop, 3× downsample)
  300m/px  →  77 km   (extract 2 560px crop, 10× downsample)
  900m/px  → 230 km   (extract full ~3 600px tile centre, ~14× downsample)

Usage:
    uv run python data_pipeline/make_scale_preview.py
    uv run python data_pipeline/make_scale_preview.py --tile swiss_alps
"""

import argparse
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path

# Keep in sync with preprocess.py
ELEV_MIN       = -500.0
ELEV_MAX       =  5000.0
OCEAN_MAX_M    =    10.0
MOUNTAIN_MIN_M =  1800.0
FOREST_LOW_M   =   100.0
FOREST_HIGH_M  =  1600.0

PATCH_PX = 256           # target output size (matches model input)
RAW_DIR  = Path("data/raw")
OUT_DIR  = Path("docs/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Scales: (label, metres_per_pixel, crop_px_before_downsample)
# crop_px is rounded to a multiple of PATCH_PX so the downsample ratio is clean.
SCALES = [
    ("30 m/px\n7.7 km",  30,   256),    # 1:1 — no downsample
    ("90 m/px\n23 km",   90,   768),    # 3× downsample
    ("300 m/px\n77 km",  300, 2560),    # 10× downsample
    ("900 m/px\n230 km", 900, None),    # use as much of the tile as possible
]

SHOWCASE_TILES = {
    "norway_sognefjord": "Copernicus_DSM_COG_10_N61_00_E006_00_DEM.tif",
    "swiss_alps":        "Copernicus_DSM_COG_10_N47_00_E008_00_DEM.tif",
    "norway_coast":      "Copernicus_DSM_COG_10_N60_00_E005_00_DEM.tif",
}

LABEL_COLORS = {
    "ocean":    np.array([0,   0,   200]),
    "mountain": np.array([200, 0,   0  ]),
    "river":    np.array([0,   200, 200]),
    "forest":   np.array([0,   150, 0  ]),
}

ELEV_CMAP = mcolors.LinearSegmentedColormap.from_list("terrain", [
    (0.00, "#1a3a5c"),
    (0.07, "#2e6b9e"),
    (0.09, "#c8b97a"),
    (0.14, "#8fad6e"),
    (0.36, "#5a8a4a"),
    (0.49, "#a08060"),
    (0.72, "#888888"),
    (1.00, "#ffffff"),
])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_tile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    norm = np.clip((dem - ELEV_MIN) / (ELEV_MAX - ELEV_MIN), 0.0, 1.0)
    norm[np.isnan(dem)] = np.nan
    return dem, norm


def centre_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Extract a square crop from the centre of arr."""
    h, w = arr.shape
    r0 = max(0, (h - size) // 2)
    c0 = max(0, (w - size) // 2)
    r1 = min(h, r0 + size)
    c1 = min(w, c0 + size)
    return arr[r0:r1, c0:c1]


def downsample(arr: np.ndarray, out_size: int) -> np.ndarray:
    """Average-pool arr down to (out_size, out_size)."""
    h, w = arr.shape
    factor = h / out_size
    if abs(factor - 1.0) < 0.01:
        return arr
    # Use zoom with order=1 (bilinear) for smooth results
    return zoom(arr, out_size / h, order=1)


def derive_labels(elev_m: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*elev_m.shape, 3), dtype=np.uint8)
    ocean    = elev_m < OCEAN_MAX_M
    mountain = elev_m > MOUNTAIN_MIN_M
    forest   = (~ocean) & (~mountain) & (elev_m > FOREST_LOW_M) & (elev_m < FOREST_HIGH_M)
    smoothed = uniform_filter(np.where(np.isnan(elev_m), 0, elev_m), size=7)
    river    = (~ocean) & (~mountain) & ((elev_m - smoothed) < -5.0)
    rgb[forest]   = LABEL_COLORS["forest"]
    rgb[ocean]    = LABEL_COLORS["ocean"]
    rgb[river]    = LABEL_COLORS["river"]
    rgb[mountain] = LABEL_COLORS["mountain"]
    return rgb


def sparsify(labels: np.ndarray, keep_ratio: float = 0.04, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sparse = np.zeros_like(labels)
    mask = rng.random(labels.shape[:2]) < keep_ratio
    sparse[mask] = labels[mask]
    return sparse


# ---------------------------------------------------------------------------
# figure: 4-scale comparison strip (heightmap row + label row)
# ---------------------------------------------------------------------------

def fig_scale_comparison(elev_m: np.ndarray, norm: np.ndarray, name: str):
    """
    Two-row figure:
      Row 1: heightmap at each scale
      Row 2: derived labels (full) at each scale
    """
    n = len(SCALES)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))
    fig.suptitle(
        f"Scale comparison — {name.replace('_', ' ').title()}\n"
        "Same tile, different effective resolution (all rendered at 256×256 px)",
        fontsize=13, y=1.01,
    )

    for col, (label, res_m, crop_px) in enumerate(SCALES):
        # determine crop size
        h, w = elev_m.shape
        if crop_px is None:
            # use the largest square that fits, rounded to nearest PATCH_PX multiple
            max_side = min(h, w)
            crop_px  = (max_side // PATCH_PX) * PATCH_PX

        actual_crop = min(crop_px, min(h, w))
        km = actual_crop * 30 / 1000  # source data is 30m/px

        e_crop = centre_crop(elev_m, actual_crop)
        n_crop = centre_crop(norm,   actual_crop)

        e_ds = downsample(np.where(np.isnan(e_crop), np.nanmean(e_crop), e_crop), PATCH_PX)
        n_ds = downsample(np.where(np.isnan(n_crop), np.nanmean(n_crop), n_crop), PATCH_PX)
        n_ds = np.clip(n_ds, 0.0, 1.0)

        labels_ds = derive_labels(e_ds)
        actual_res = actual_crop * 30 / PATCH_PX

        # Row 0: heightmap
        axes[0, col].imshow(n_ds, cmap=ELEV_CMAP, vmin=0, vmax=1, interpolation="nearest")
        axes[0, col].set_title(
            f"{label}\n({actual_res:.0f} m/px · {km:.0f} km)",
            fontsize=10,
        )
        axes[0, col].axis("off")

        # Row 1: full labels
        axes[1, col].imshow(labels_ds, interpolation="nearest")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Heightmap", fontsize=10, labelpad=6)
    axes[1, 0].set_ylabel("Derived labels", fontsize=10, labelpad=6)

    legend = [Patch(color=np.array(v) / 255, label=k) for k, v in LABEL_COLORS.items()]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    out = OUT_DIR / f"scale_comparison_{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ---------------------------------------------------------------------------
# figure: label pipeline at each scale (4-panel per scale)
# ---------------------------------------------------------------------------

def fig_scale_label_pipeline(elev_m: np.ndarray, norm: np.ndarray, name: str):
    """
    For each scale: heightmap / full labels / sparse labels / overlay — one row per scale.
    """
    n = len(SCALES)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    fig.suptitle(
        f"Label pipeline at each scale — {name.replace('_', ' ').title()}",
        fontsize=13, y=1.01,
    )

    col_titles = ["Heightmap", "Full labels", "Sparse labels (4%)", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)

    h, w = elev_m.shape
    for row, (label, res_m, crop_px) in enumerate(SCALES):
        if crop_px is None:
            max_side = min(h, w)
            crop_px  = (max_side // PATCH_PX) * PATCH_PX
        actual_crop = min(crop_px, min(h, w))
        km = actual_crop * 30 / 1000
        actual_res = actual_crop * 30 / PATCH_PX

        e_crop = centre_crop(elev_m, actual_crop)
        n_crop = centre_crop(norm,   actual_crop)

        e_ds = downsample(np.where(np.isnan(e_crop), np.nanmean(e_crop), e_crop), PATCH_PX)
        n_ds = downsample(np.where(np.isnan(n_crop), np.nanmean(n_crop), n_crop), PATCH_PX)
        n_ds = np.clip(n_ds, 0.0, 1.0)

        labels_full   = derive_labels(e_ds)
        labels_sparse = sparsify(labels_full)

        overlay = (np.stack([n_ds] * 3, axis=-1) * 255).astype(np.uint8)
        has_label = labels_sparse.sum(axis=-1) > 0
        overlay[has_label] = labels_sparse[has_label]

        row_label = f"{label}\n({actual_res:.0f} m/px · {km:.0f} km)"
        axes[row, 0].set_ylabel(row_label, fontsize=9)

        for col, (img, use_cmap) in enumerate([
            (n_ds,          True),
            (labels_full,   False),
            (labels_sparse, False),
            (overlay,       False),
        ]):
            ax = axes[row, col]
            if use_cmap:
                ax.imshow(img, cmap=ELEV_CMAP, vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            ax.axis("off")

    legend = [Patch(color=np.array(v) / 255, label=k) for k, v in LABEL_COLORS.items()]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    out = OUT_DIR / f"scale_label_pipeline_{name}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(tile_key: str | None = None):
    tiles = {tile_key: SHOWCASE_TILES[tile_key]} if tile_key else SHOWCASE_TILES
    for name, filename in tiles.items():
        path = RAW_DIR / filename
        if not path.exists():
            print(f"missing {filename}, skipping {name}")
            continue
        print(f"\n{name}")
        elev_m, norm = load_tile(path)
        lo, hi = int(np.nanmin(elev_m)), int(np.nanmax(elev_m))
        print(f"  tile shape={norm.shape}  elev={lo}m–{hi}m")
        fig_scale_comparison(elev_m, norm, name)
        fig_scale_label_pipeline(elev_m, norm, name)

    print("\nAll scale preview images saved to docs/images/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", choices=list(SHOWCASE_TILES), default=None,
                        help="Process only this tile (default: all)")
    args = parser.parse_args()
    main(tile_key=args.tile)
