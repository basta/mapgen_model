"""
Inspect the processed dataset — patch counts, class balance, relief distribution,
and a visual spot-check of random pairs.

Usage:
    uv run python data_pipeline/stats.py              # text report only
    uv run python data_pipeline/stats.py --plot       # also save docs/images/stats_*.png
    uv run python data_pipeline/stats.py --plot --n 12  # spot-check 12 random pairs
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
DOCS_IMG_DIR  = Path("docs/images")

ELEV_CMAP = mcolors.LinearSegmentedColormap.from_list("terrain", [
    (0.0,  "#1a3a5c"), (0.28, "#2e6b9e"), (0.30, "#c8b97a"),
    (0.35, "#8fad6e"), (0.55, "#5a8a4a"), (0.70, "#a08060"),
    (0.85, "#888888"), (1.0,  "#ffffff"),
])

# Label pixel color → name (must match preprocess.py)
LABEL_MAP = {
    (0,   0,   200): "ocean",
    (200, 0,   0  ): "mountain",
    (0,   200, 200): "river",
    (0,   150, 0  ): "forest",
}


# ---------------------------------------------------------------------------
# data loading helpers
# ---------------------------------------------------------------------------

def iter_pairs(split: str):
    d = PROCESSED_DIR / split
    if not d.exists():
        return
    for hf in sorted(d.glob("*_height.npy")):
        lf = hf.with_name(hf.name.replace("_height", "_label"))
        if lf.exists():
            yield np.load(hf), np.load(lf)


def count_pairs(split: str) -> int:
    d = PROCESSED_DIR / split
    return len(list(d.glob("*_height.npy"))) if d.exists() else 0


# ---------------------------------------------------------------------------
# stats collection
# ---------------------------------------------------------------------------

def collect_stats(split: str, max_samples: int = 2000) -> dict:
    """Collect relief and label-density stats from up to max_samples pairs."""
    reliefs, densities, class_counts = [], [], {k: 0 for k in LABEL_MAP.values()}
    class_counts["unlabelled"] = 0
    total_pixels = 0

    for i, (h, label) in enumerate(iter_pairs(split)):
        if i >= max_samples:
            break
        reliefs.append(float(np.nanmax(h) - np.nanmin(h)))

        labeled = label.sum(axis=-1) > 0
        densities.append(float(labeled.mean()))

        for color, name in LABEL_MAP.items():
            mask = np.all(label == np.array(color, dtype=np.uint8), axis=-1)
            class_counts[name] += int(mask.sum())
        class_counts["unlabelled"] += int((~labeled).sum())
        total_pixels += label.shape[0] * label.shape[1]

    return {
        "reliefs": reliefs,
        "densities": densities,
        "class_counts": class_counts,
        "total_pixels": total_pixels,
    }


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------

def print_report():
    for split in ("train", "val"):
        n = count_pairs(split)
        if n == 0:
            print(f"{split}: no data found in {PROCESSED_DIR / split}")
            continue

        stats = collect_stats(split)
        r = stats["reliefs"]
        d = stats["densities"]
        cc = stats["class_counts"]
        tp = stats["total_pixels"]

        print(f"\n{'='*50}")
        print(f"  {split.upper()}  —  {n} pairs")
        print(f"{'='*50}")
        print(f"  Relief (norm)  min={min(r):.3f}  median={np.median(r):.3f}  max={max(r):.3f}")
        print(f"  Label density  min={min(d):.1%}   median={np.median(d):.1%}   max={max(d):.1%}")
        print(f"\n  Class pixel share (of sampled pixels):")
        for name, count in sorted(cc.items(), key=lambda x: -x[1]):
            print(f"    {name:<12} {count/tp:6.1%}")

    size_gb = sum(f.stat().st_size for f in PROCESSED_DIR.rglob("*.npy")) / 1e9
    print(f"\n  Total disk usage: {size_gb:.2f} GB")


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_distributions(split: str):
    stats = collect_stats(split)
    r, d, cc = stats["reliefs"], stats["densities"], stats["class_counts"]
    tp = stats["total_pixels"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(r, bins=40, color="#5a8a4a", edgecolor="white", linewidth=0.4)
    axes[0].set_title("Relief distribution")
    axes[0].set_xlabel("Normalised relief (max−min)")
    axes[0].set_ylabel("Patches")

    axes[1].hist(d, bins=40, color="#2e6b9e", edgecolor="white", linewidth=0.4)
    axes[1].set_title("Label density per patch")
    axes[1].set_xlabel("Fraction of labelled pixels")

    names  = [k for k in cc if k != "unlabelled"]
    shares = [cc[k] / tp for k in names]
    colors = ["#2e6b9e", "#c83200", "#00c8c8", "#009600"]
    axes[2].bar(names, shares, color=colors[:len(names)])
    axes[2].set_title("Class pixel share (labelled only)")
    axes[2].set_ylabel("Fraction")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle(f"Dataset stats — {split}", fontsize=13)
    fig.tight_layout()
    DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_IMG_DIR / f"stats_{split}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_spot_check(split: str, n: int = 8):
    """Show n random (heightmap, sparse label) pairs side by side."""
    pairs = list(iter_pairs(split))
    if not pairs:
        return
    indices = np.random.default_rng(7).choice(len(pairs), size=min(n, len(pairs)), replace=False)

    fig, axes = plt.subplots(2, len(indices), figsize=(len(indices) * 2.5, 5.5))
    for col, idx in enumerate(indices):
        h, label = pairs[idx]
        axes[0, col].imshow(h, cmap=ELEV_CMAP, vmin=0, vmax=1, interpolation="nearest")
        axes[0, col].axis("off")
        axes[1, col].imshow(label, interpolation="nearest")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Heightmap", fontsize=10)
    axes[1, 0].set_ylabel("Sparse labels", fontsize=10)
    fig.suptitle(f"Spot check — {split} ({len(pairs)} pairs total)", fontsize=12)
    fig.tight_layout()
    out = DOCS_IMG_DIR / f"spot_check_{split}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(plot: bool, n_spot: int):
    if not PROCESSED_DIR.exists() or not any(PROCESSED_DIR.rglob("*.npy")):
        print(f"No processed data found in {PROCESSED_DIR}.")
        print("Run: uv run python data_pipeline/preprocess.py")
        return

    print_report()

    if plot:
        print()
        for split in ("train", "val"):
            if count_pairs(split) == 0:
                continue
            plot_distributions(split)
            plot_spot_check(split, n=n_spot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Save distribution + spot-check plots")
    parser.add_argument("--n", type=int, default=8, help="Number of pairs in spot-check grid")
    args = parser.parse_args()
    main(plot=args.plot, n_spot=args.n)
