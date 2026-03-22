"""
Generate dataset preview images for docs/dataset.md.

Uses the same normalization and label thresholds as preprocess.py so the
visuals match what the model actually trains on.

Produces:
  docs/images/raw_tile_{name}.png        — full tile with global elevation colormap
  docs/images/patches_{name}.png         — 3×3 grid of 256×256 training patches
  docs/images/label_pipeline_{name}.png  — 4-panel: heightmap / full labels / sparse / overlay
  docs/images/normalization_compare.png  — side-by-side per-tile vs global normalization

Usage:
    uv run python data_pipeline/make_dataset_preview.py
"""

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from scipy.ndimage import uniform_filter

# Keep in sync with preprocess.py
ELEV_MIN       = -500.0
ELEV_MAX       =  5000.0
OCEAN_MAX_M    =    10.0
MOUNTAIN_MIN_M =  1800.0
FOREST_LOW_M   =   100.0
FOREST_HIGH_M  =  1600.0
MIN_RELIEF_M   =   200.0

RAW_DIR = Path("data/raw")
OUT_DIR = Path("docs/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHOWCASE_TILES = {
    "norway_sognefjord": "Copernicus_DSM_COG_10_N61_00_E006_00_DEM.tif",
    "swiss_alps":        "Copernicus_DSM_COG_10_N47_00_E008_00_DEM.tif",
    "norway_coast":      "Copernicus_DSM_COG_10_N60_00_E005_00_DEM.tif",
}

# Colormap stops are calibrated to global normalization:
#   0.0  = -500m  (below sea floor)
#   ~0.09 = 0m   (sea level)
#   ~0.11 = 100m (lowland / beach)
#   ~0.49 = 1800m (mountain threshold)
#   1.0  = 5000m  (high alpine)
ELEV_CMAP = mcolors.LinearSegmentedColormap.from_list("terrain", [
    (0.00, "#1a3a5c"),  # deep water
    (0.07, "#2e6b9e"),  # shallow water
    (0.09, "#c8b97a"),  # beach / sea level
    (0.14, "#8fad6e"),  # lowland
    (0.36, "#5a8a4a"),  # forest
    (0.49, "#a08060"),  # highland / mountain threshold
    (0.72, "#888888"),  # rock
    (1.00, "#ffffff"),  # snow / high alpine
])

LABEL_COLORS = {
    "ocean":    np.array([0,   0,   200]),
    "mountain": np.array([200, 0,   0  ]),
    "river":    np.array([0,   200, 200]),
    "forest":   np.array([0,   150, 0  ]),
}


# ---------------------------------------------------------------------------
# helpers — mirror preprocess.py logic exactly
# ---------------------------------------------------------------------------

def load_tile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (elev_m, norm) using global normalization."""
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    norm = np.clip((dem - ELEV_MIN) / (ELEV_MAX - ELEV_MIN), 0.0, 1.0)
    norm[np.isnan(dem)] = np.nan
    return dem, norm


def load_tile_pertile(path: Path) -> np.ndarray:
    """Old per-tile normalization — used only for the comparison figure."""
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    lo, hi = np.nanmin(dem), np.nanmax(dem)
    return (dem - lo) / (hi - lo + 1e-6)


def derive_labels(elev_m: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*elev_m.shape, 3), dtype=np.uint8)
    ocean    = elev_m < OCEAN_MAX_M
    mountain = elev_m > MOUNTAIN_MIN_M
    forest   = (~ocean) & (~mountain) & (elev_m > FOREST_LOW_M) & (elev_m < FOREST_HIGH_M)
    smoothed = uniform_filter(elev_m, size=7)
    river    = (~ocean) & (~mountain) & ((elev_m - smoothed) < -5.0)
    rgb[forest]   = LABEL_COLORS["forest"]
    rgb[ocean]    = LABEL_COLORS["ocean"]
    rgb[river]    = LABEL_COLORS["river"]
    rgb[mountain] = LABEL_COLORS["mountain"]
    return rgb


def sparsify(labels: np.ndarray, keep_ratio: float = 0.04,
             seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sparse = np.zeros_like(labels)
    mask = rng.random(labels.shape[:2]) < keep_ratio
    sparse[mask] = labels[mask]
    return sparse


def sample_patches(norm: np.ndarray, elev_m: np.ndarray, n: int = 9,
                   size: int = 256) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return up to n patches with >= MIN_RELIEF_M metres of relief."""
    H, W = norm.shape
    patches, attempts = [], 0
    rng = np.random.default_rng(42)
    while len(patches) < n and attempts < 1000:
        r = rng.integers(0, H - size)
        c = rng.integers(0, W - size)
        ep = elev_m[r:r+size, c:c+size]
        if np.nanmax(ep) - np.nanmin(ep) >= MIN_RELIEF_M:
            patches.append((norm[r:r+size, c:c+size], ep))
        attempts += 1
    return patches


# ---------------------------------------------------------------------------
# figure generators
# ---------------------------------------------------------------------------

def fig_raw_tile(norm: np.ndarray, elev_m: np.ndarray, name: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap=ELEV_CMAP, vmin=0, vmax=1, interpolation="bilinear")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03)
    # Label colorbar ticks in metres
    ticks = [0.0, 0.09, 0.27, 0.49, 0.72, 1.0]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{ELEV_MIN + t*(ELEV_MAX-ELEV_MIN):.0f}m" for t in ticks])
    lo, hi = int(np.nanmin(elev_m)), int(np.nanmax(elev_m))
    ax.set_title(f"{name.replace('_', ' ').title()}  ({lo}m – {hi}m)", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"raw_tile_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved raw_tile_{name}.png")


def fig_patches(norm: np.ndarray, elev_m: np.ndarray, name: str):
    patches = sample_patches(norm, elev_m, n=9)
    if not patches:
        print(f"  skipping patches_{name} — not enough relief")
        return
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, (hp, ep) in zip(axes.flat, patches):
        ax.imshow(hp, cmap=ELEV_CMAP, vmin=0, vmax=1, interpolation="bilinear")
        lo, hi = int(np.nanmin(ep)), int(np.nanmax(ep))
        ax.set_title(f"{lo}–{hi}m", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"256×256 training patches — {name.replace('_', ' ').title()}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"patches_{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved patches_{name}.png")


def fig_label_pipeline(norm: np.ndarray, elev_m: np.ndarray, name: str):
    patches = sample_patches(norm, elev_m, n=1)
    if not patches:
        return
    hp, ep = patches[0]
    labels = derive_labels(ep)
    sparse = sparsify(labels, keep_ratio=0.04)

    overlay = (np.stack([hp]*3, axis=-1) * 255).astype(np.uint8)
    overlay[sparse.sum(axis=-1) > 0] = sparse[sparse.sum(axis=-1) > 0]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, title, img, use_cmap in zip(
        axes,
        ["Heightmap (global norm)", "Full labels", "Sparse labels (4%)", "Overlay"],
        [hp, labels, sparse, overlay],
        [True, False, False, False],
    ):
        if use_cmap:
            ax.imshow(img, cmap=ELEV_CMAP, vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    legend = [Patch(color=np.array(v)/255, label=k) for k, v in LABEL_COLORS.items()]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    lo, hi = int(np.nanmin(ep)), int(np.nanmax(ep))
    fig.suptitle(f"Label pipeline — {name.replace('_', ' ').title()}  ({lo}–{hi}m patch)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"label_pipeline_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved label_pipeline_{name}.png")


def fig_normalization_compare():
    """Side-by-side showing why global normalization matters."""
    tiles = [
        ("Swiss Alps\n(0–4811m)", "Copernicus_DSM_COG_10_N47_00_E008_00_DEM.tif"),
        ("Norway Fjord\n(0–1743m)", "Copernicus_DSM_COG_10_N61_00_E006_00_DEM.tif"),
        ("Norway Coast\n(0–1021m)", "Copernicus_DSM_COG_10_N60_00_E005_00_DEM.tif"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("Per-tile normalization  vs  Global normalization", fontsize=13, y=1.01)
    axes[0, 0].set_ylabel("Per-tile\n(max always = 1)", fontsize=10)
    axes[1, 0].set_ylabel("Global\n(max = 5000m)", fontsize=10)

    for col, (title, filename) in enumerate(tiles):
        path = RAW_DIR / filename
        if not path.exists():
            continue
        elev_m, norm_global = load_tile(path)
        norm_pertile = load_tile_pertile(path)

        axes[0, col].imshow(norm_pertile, cmap=ELEV_CMAP, vmin=0, vmax=1,
                            interpolation="bilinear")
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(norm_global, cmap=ELEV_CMAP, vmin=0, vmax=1,
                            interpolation="bilinear")
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "normalization_compare.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved normalization_compare.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    for name, filename in SHOWCASE_TILES.items():
        path = RAW_DIR / filename
        if not path.exists():
            print(f"missing {filename}, skipping {name}")
            continue
        print(f"\n{name}")
        elev_m, norm = load_tile(path)
        lo, hi = int(np.nanmin(elev_m)), int(np.nanmax(elev_m))
        print(f"  shape={norm.shape}  elev={lo}m–{hi}m")
        fig_raw_tile(norm, elev_m, name)
        fig_patches(norm, elev_m, name)
        fig_label_pipeline(norm, elev_m, name)

    print("\nnormalization comparison")
    fig_normalization_compare()
    print("\nAll preview images saved to docs/images/")


if __name__ == "__main__":
    main()
