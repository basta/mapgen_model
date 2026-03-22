"""
Preprocess raw GeoTIFF tiles into (sparse_label, heightmap) training pairs.

For each tile:
  1. Normalize elevation to [0, 1] using a fixed global range [ELEV_MIN, ELEV_MAX]
     so that sea level, mountain height, etc. are consistent across all tiles.
  2. Slice into 256×256 patches with stride 128
  3. Filter patches with relief < MIN_RELIEF_M metres (too flat to be useful)
  4. Derive full label map using absolute elevation thresholds (metres)
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
from tqdm import tqdm

RAW_DIR   = Path("data/raw")
OUT_DIR   = Path("data/processed")
PATCH_SIZE  = 256
STRIDE      = 128
VAL_RATIO   = 0.10   # 10% of tiles held out for validation
MIN_KEEP    = 0.01   # minimum label sparsity
MAX_KEEP    = 0.20   # maximum label sparsity

# Global elevation range (metres) — fixed across all tiles so the model sees
# consistent absolute heights. Covers our target regions with headroom.
ELEV_MIN = -500.0
ELEV_MAX = 5000.0

# Label thresholds in metres
OCEAN_MAX_M    =    10.0   # below sea level / tidal zone
MOUNTAIN_MIN_M =  1800.0   # true alpine terrain
FOREST_LOW_M   =   100.0   # above floodplain
FOREST_HIGH_M  =  1600.0   # below treeline

# Minimum patch relief in metres to filter out flat/boring tiles
MIN_RELIEF_M = 200.0

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

def load_tile(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load a GeoTIFF and return (elev_metres, normalized_heightmap, nodata_fraction).

    elev_metres   — raw elevation values in metres (for label derivation)
    normalized    — elevation mapped to [0,1] using the global ELEV_MIN/ELEV_MAX
                    so that height is comparable across all tiles
    """
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        invalid = dem == nodata
        dem[invalid] = np.nan
        nodata_frac = float(invalid.mean())
    else:
        nodata_frac = 0.0

    norm = np.clip((dem - ELEV_MIN) / (ELEV_MAX - ELEV_MIN), 0.0, 1.0)
    norm[np.isnan(dem)] = np.nan
    return dem, norm, nodata_frac


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


def derive_labels(elev_m: np.ndarray, river_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Derive a full RGB label map from raw elevation in metres.
    Thresholds are absolute so labels are consistent across all tiles.
    Priority (highest wins): mountain > river > ocean > forest
    """
    rgb = np.zeros((*elev_m.shape, 3), dtype=np.uint8)

    ocean    = elev_m < OCEAN_MAX_M
    mountain = elev_m > MOUNTAIN_MIN_M
    forest   = (~ocean) & (~mountain) & (elev_m > FOREST_LOW_M) & (elev_m < FOREST_HIGH_M)

    if river_mask is not None:
        river = river_mask & (~ocean) & (~mountain)
    else:
        # fallback: channels sit below their local neighbourhood mean
        smoothed = uniform_filter(elev_m, size=7)
        river = (~ocean) & (~mountain) & ((elev_m - smoothed) < -5.0)

    # paint in priority order (last wins for overlaps)
    rgb[forest]   = LABEL_COLORS["forest"]
    rgb[ocean]    = LABEL_COLORS["ocean"]
    rgb[river]    = LABEL_COLORS["river"]
    rgb[mountain] = LABEL_COLORS["mountain"]
    return rgb


# ---------------------------------------------------------------------------
# patching + sparsification
# ---------------------------------------------------------------------------

def extract_patches(norm: np.ndarray, elev_m: np.ndarray, labels: np.ndarray,
                    rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slice tile into overlapping 256×256 patches, filter flat ones."""
    H, W = norm.shape
    pairs = []
    for r in range(0, H - PATCH_SIZE + 1, STRIDE):
        for c in range(0, W - PATCH_SIZE + 1, STRIDE):
            ep = elev_m[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            if np.isnan(ep).mean() > 0.05:
                continue  # too much nodata
            relief_m = float(np.nanmax(ep) - np.nanmin(ep))
            if relief_m < MIN_RELIEF_M:
                continue  # too flat
            hp = norm[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
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

    tile_bar = tqdm(tiles, unit="tile", dynamic_ncols=True)
    for tile_path in tile_bar:
        split = "val" if tile_path.name in val_set else "train"
        tile_bar.set_description(tile_path.stem[-30:])

        elev_m, norm, nodata_frac = load_tile(tile_path)
        if nodata_frac > 0.5:
            tile_bar.write(f"skip {tile_path.name} (nodata={nodata_frac:.0%})")
            continue

        river_mask = derive_rivers(tile_path, elev_m.shape)
        labels = derive_labels(elev_m, river_mask)
        pairs = extract_patches(norm, elev_m, labels, rng)

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
                tile_bar.write(f"Reached limit of {limit} patches.")
                _print_summary(counters, OUT_DIR)
                return

        tile_bar.set_postfix(patches=total_patches, split=split)

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
