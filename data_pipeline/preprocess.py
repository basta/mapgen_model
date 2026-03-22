"""
Preprocess raw GeoTIFF tiles into (sparse_label, heightmap) training pairs.

For each tile:
  1. Normalize elevation to [0, 1] (per-tile min/max)
  2. Slice into 256×256 patches with stride 128
  3. Filter patches with relief < MIN_RELIEF_M (too flat to be useful)
  4. Derive full label map: ocean / mountain / river / forest
  5. Sparsify labels to a random keep_ratio in [MIN_KEEP, MAX_KEEP]
  6. Save pair as data/processed/{split}/{idx:06d}_{label|height}.npy

Output layout:
  data/processed/
    train/
      000000_height.npy   float32 (256, 256)  normalized [0,1]
      000000_label.npy    uint8   (256, 256, 3) sparse RGB
      ...
    val/
      ...

Usage:
    uv run python data_pipeline/preprocess.py
    uv run python data_pipeline/preprocess.py --limit 500   # quick test run
"""

import argparse
import numpy as np
import rasterio
from pathlib import Path
from scipy.ndimage import maximum_filter, binary_dilation, uniform_filter
from pysheds.grid import Grid

RAW_DIR   = Path("data/raw")
OUT_DIR   = Path("data/processed")
PATCH_SIZE  = 256
STRIDE      = 128
MIN_RELIEF  = 0.20   # fraction of normalized range; ~200m on high-relief tiles
VAL_RATIO   = 0.10   # 10% of tiles held out for validation
MIN_KEEP    = 0.01   # minimum label sparsity
MAX_KEEP    = 0.20   # maximum label sparsity

# Label RGB values — must match frontend brush colors
LABEL_COLORS = {
    "ocean":    np.array([0,   0,   200], dtype=np.uint8),
    "mountain": np.array([200, 0,   0  ], dtype=np.uint8),
    "river":    np.array([0,   200, 200], dtype=np.uint8),
    "forest":   np.array([0,   150, 0  ], dtype=np.uint8),
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_tile(path: Path) -> tuple[np.ndarray, float]:
    """
    Load a GeoTIFF and return (normalized_heightmap, nodata_fraction).
    Normalization is per-tile min/max so the model sees full dynamic range.
    """
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        invalid = dem == nodata
        dem[invalid] = np.nan
        nodata_frac = invalid.mean()
    else:
        nodata_frac = 0.0

    lo, hi = np.nanmin(dem), np.nanmax(dem)
    norm = (dem - lo) / (hi - lo + 1e-6)
    return norm, nodata_frac


# ---------------------------------------------------------------------------
# label derivation
# ---------------------------------------------------------------------------

def derive_rivers(path: Path, shape: tuple) -> np.ndarray:
    """
    Derive river mask using D8 flow accumulation (pysheds).
    Returns boolean array, True where flow accumulation exceeds threshold.
    Falls back to smoothing-proxy if pysheds fails.
    """
    try:
        grid = Grid.from_raster(str(path))
        dem_grid = grid.read_raster(str(path))
        pit_filled = grid.fill_pits(dem_grid)
        flooded = grid.fill_depressions(pit_filled)
        inflated = grid.resolve_flats(flooded)
        fdir = grid.flowdir(inflated)
        acc = grid.accumulation(fdir)
        # threshold: top ~0.5% of accumulation values
        threshold = np.percentile(acc[acc > 0], 99.5)
        river_mask = np.array(acc >= threshold, dtype=bool)
        # resize to match normalized dem shape if needed
        if river_mask.shape != shape:
            from scipy.ndimage import zoom
            zy, zx = shape[0] / river_mask.shape[0], shape[1] / river_mask.shape[1]
            river_mask = zoom(river_mask.astype(np.float32), (zy, zx)) > 0.5
        return river_mask
    except Exception:
        return None  # caller will use fallback


def derive_labels(h: np.ndarray, river_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Derive a full RGB label map from a normalized heightmap patch.
    Priority (highest wins): mountain > river > ocean > forest
    """
    rgb = np.zeros((*h.shape, 3), dtype=np.uint8)

    ocean    = h < 0.30
    peaks    = (h > 0.75) & (h == maximum_filter(h, size=15))
    mountain = binary_dilation(peaks, iterations=6)
    forest   = (~ocean) & (~mountain) & (h > 0.32) & (h < 0.70)

    if river_mask is not None:
        river = river_mask & (~ocean) & (~mountain)
    else:
        # fallback: pixels that sit below their local neighborhood mean
        smoothed = uniform_filter(h, size=7)
        river = (~ocean) & (~mountain) & ((h - smoothed) < -0.012)

    # paint in priority order (last wins for overlaps)
    rgb[forest]   = LABEL_COLORS["forest"]
    rgb[ocean]    = LABEL_COLORS["ocean"]
    rgb[river]    = LABEL_COLORS["river"]
    rgb[mountain] = LABEL_COLORS["mountain"]
    return rgb


# ---------------------------------------------------------------------------
# patching + sparsification
# ---------------------------------------------------------------------------

def extract_patches(h: np.ndarray, labels: np.ndarray,
                    rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slice tile into overlapping 256×256 patches, filter flat ones."""
    H, W = h.shape
    pairs = []
    for r in range(0, H - PATCH_SIZE + 1, STRIDE):
        for c in range(0, W - PATCH_SIZE + 1, STRIDE):
            hp = h[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            if np.isnan(hp).mean() > 0.05:
                continue  # too much nodata
            relief = float(np.nanmax(hp) - np.nanmin(hp))
            if relief < MIN_RELIEF:
                continue  # too flat
            lp = labels[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            pairs.append((hp, lp))
    return pairs


def sparsify(labels: np.ndarray, keep_ratio: float,
             rng: np.random.Generator) -> np.ndarray:
    sparse = np.zeros_like(labels)
    mask = rng.random(labels.shape[:2]) < keep_ratio
    sparse[mask] = labels[mask]
    return sparse


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(limit: int | None = None):
    rng = np.random.default_rng(42)

    tiles = sorted(RAW_DIR.glob("*.tif"))
    if not tiles:
        print(f"No tiles found in {RAW_DIR}")
        return

    # deterministic train/val split by tile (not by patch)
    rng_split = np.random.default_rng(0)
    shuffled = tiles.copy()
    rng_split.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    val_set = set(t.name for t in shuffled[:n_val])

    (OUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "val").mkdir(parents=True, exist_ok=True)

    counters = {"train": 0, "val": 0}
    total_patches = 0

    for tile_idx, tile_path in enumerate(tiles):
        split = "val" if tile_path.name in val_set else "train"
        print(f"[{tile_idx+1}/{len(tiles)}] {tile_path.name} → {split}", end="  ", flush=True)

        h, nodata_frac = load_tile(tile_path)
        if nodata_frac > 0.5:
            print(f"skip (nodata={nodata_frac:.0%})")
            continue

        river_mask = derive_rivers(tile_path, h.shape)
        labels = derive_labels(h, river_mask)
        pairs = extract_patches(h, labels, rng)

        out_subdir = OUT_DIR / split
        for hp, lp in pairs:
            keep_ratio = float(rng.uniform(MIN_KEEP, MAX_KEEP))
            sparse = sparsify(lp, keep_ratio, rng)
            idx = counters[split]
            np.save(out_subdir / f"{idx:06d}_height.npy", hp.astype(np.float32))
            np.save(out_subdir / f"{idx:06d}_label.npy",  sparse)
            counters[split] += 1
            total_patches += 1

            if limit and total_patches >= limit:
                print(f"\nReached limit of {limit} patches.")
                _print_summary(counters, OUT_DIR)
                return

        print(f"{len(pairs)} patches")

    _print_summary(counters, OUT_DIR)


def _print_summary(counters: dict, out_dir: Path):
    total = sum(counters.values())
    size_gb = sum(f.stat().st_size for f in out_dir.rglob("*.npy")) / 1e9
    print(f"\nDone.  train={counters['train']}  val={counters['val']}  total={total}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many patches (for quick testing)")
    args = parser.parse_args()
    main(limit=args.limit)
