"""
Generate dataset preview images for docs/dataset.md.

Produces:
  docs/images/raw_tile_{name}.png        — raw elevation heatmap of a full tile
  docs/images/patches_{name}.png         — 3×3 grid of 256×256 patches from that tile
  docs/images/label_pipeline_{name}.png  — 4-panel: heightmap / full labels / sparse labels overlay

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
from scipy.ndimage import maximum_filter, label as ndlabel

RAW_DIR = Path("data/raw")
OUT_DIR = Path("docs/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHOWCASE_TILES = {
    "norway_sognefjord": "Copernicus_DSM_COG_10_N61_00_E006_00_DEM.tif",
    "swiss_alps":        "Copernicus_DSM_COG_10_N47_00_E008_00_DEM.tif",
    "norway_coast":      "Copernicus_DSM_COG_10_N60_00_E005_00_DEM.tif",
}

ELEV_CMAP = mcolors.LinearSegmentedColormap.from_list("terrain", [
    (0.0,  "#1a3a5c"),  # deep ocean
    (0.28, "#2e6b9e"),  # shallow ocean
    (0.30, "#c8b97a"),  # beach
    (0.35, "#8fad6e"),  # lowland
    (0.55, "#5a8a4a"),  # forest
    (0.70, "#a08060"),  # highland
    (0.85, "#888888"),  # rock
    (1.0,  "#ffffff"),  # snow
])

LABEL_COLORS = {
    "ocean":    np.array([0,   0,   200]),
    "mountain": np.array([200, 0,   0  ]),
    "river":    np.array([0,   200, 200]),
    "forest":   np.array([0,   150, 0  ]),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_normalized(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    lo, hi = np.nanmin(dem), np.nanmax(dem)
    return (dem - lo) / (hi - lo + 1e-6)


def derive_labels(h: np.ndarray) -> np.ndarray:
    """Return an RGB label image (H, W, 3) from a normalized heightmap."""
    rgb = np.zeros((*h.shape, 3), dtype=np.uint8)

    ocean    = h < 0.30
    mountain = (h > 0.75) & (h == maximum_filter(h, size=15))
    # dilate mountain peaks
    from scipy.ndimage import binary_dilation
    mountain = binary_dilation(mountain, iterations=6)
    forest   = (~ocean) & (~mountain) & (h > 0.32) & (h < 0.70)
    # river: simple proxy via local minima channels (not full D8 for preview)
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(h, size=5)
    river    = (~ocean) & (~mountain) & (h - smoothed < -0.015)

    rgb[ocean]    = LABEL_COLORS["ocean"]
    rgb[forest]   = LABEL_COLORS["forest"]
    rgb[river]    = LABEL_COLORS["river"]
    rgb[mountain] = LABEL_COLORS["mountain"]
    return rgb


def sparsify(labels: np.ndarray, keep_ratio: float = 0.04) -> np.ndarray:
    sparse = np.zeros_like(labels)
    mask = np.random.random(labels.shape[:2]) < keep_ratio
    sparse[mask] = labels[mask]
    return sparse


def random_patches(h: np.ndarray, n: int = 9, size: int = 256,
                   min_relief: float = 0.25) -> list[np.ndarray]:
    """Return up to n patches with sufficient relief."""
    H, W = h.shape
    patches, attempts = [], 0
    rng = np.random.default_rng(42)
    while len(patches) < n and attempts < 500:
        r = rng.integers(0, H - size)
        c = rng.integers(0, W - size)
        patch = h[r:r+size, c:c+size]
        if np.nanmax(patch) - np.nanmin(patch) >= min_relief:
            patches.append(patch)
        attempts += 1
    return patches


# ---------------------------------------------------------------------------
# figure generators
# ---------------------------------------------------------------------------

def fig_raw_tile(h: np.ndarray, name: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(h, cmap=ELEV_CMAP, vmin=0, vmax=1, interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Normalised elevation (0–1)", fraction=0.03)
    ax.set_title(f"Raw tile — {name.replace('_', ' ').title()}", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"raw_tile_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved raw_tile_{name}.png")


def fig_patches(h: np.ndarray, name: str):
    patches = random_patches(h, n=9)
    if not patches:
        print(f"  skipping patches_{name} — not enough relief")
        return
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, p in zip(axes.flat, patches):
        ax.imshow(p, cmap=ELEV_CMAP, vmin=0, vmax=1, interpolation="bilinear")
        ax.axis("off")
    fig.suptitle(f"256×256 patches — {name.replace('_', ' ').title()}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"patches_{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved patches_{name}.png")


def fig_label_pipeline(h: np.ndarray, name: str):
    patch = random_patches(h, n=1)[0]
    labels = derive_labels(patch)
    sparse = sparsify(labels, keep_ratio=0.04)

    # overlay: heightmap + sparse dots
    overlay = np.stack([patch]*3, axis=-1)
    overlay = (overlay * 255).astype(np.uint8)
    mask = sparse.sum(axis=-1) > 0
    overlay[mask] = sparse[mask]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    titles = ["Heightmap", "Full labels", "Sparse labels (4%)", "Overlay on heightmap"]
    imgs   = [patch, labels, sparse, overlay]
    cmaps  = [ELEV_CMAP, None, None, None]

    for ax, title, img, cmap in zip(axes, titles, imgs, cmaps):
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    legend = [
        Patch(color=np.array(LABEL_COLORS["ocean"])/255,    label="Ocean"),
        Patch(color=np.array(LABEL_COLORS["mountain"])/255, label="Mountain"),
        Patch(color=np.array(LABEL_COLORS["river"])/255,    label="River"),
        Patch(color=np.array(LABEL_COLORS["forest"])/255,   label="Forest"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Label pipeline — {name.replace('_', ' ').title()}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"label_pipeline_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved label_pipeline_{name}.png")


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
        h = load_normalized(path)
        print(f"  shape={h.shape}  relief={np.nanmax(h)-np.nanmin(h):.3f}")
        fig_raw_tile(h, name)
        fig_patches(h, name)
        fig_label_pipeline(h, name)

    print("\nAll preview images saved to docs/images/")


if __name__ == "__main__":
    main()
