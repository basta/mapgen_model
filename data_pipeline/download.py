"""
Download Copernicus GLO-30 DEM tiles from AWS S3 (public, no credentials needed).

Tile naming: Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM
Inside each "folder" the actual GeoTIFF is:
  Copernicus_DSM_COG_10_N{lat}_00_{E|W}{lon}_00_DEM/
    DEM/
      Copernicus_DSM_COG_10_N{lat}_00_{E|W}{lon}_00_DEM_COG.tif

Usage:
    uv run python data_pipeline/download.py
"""

import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

BUCKET = "copernicus-dem-30m"
OUT_DIR = Path("data/raw")

# Target regions: (name, lat_range, lon_range)
# lat/lon are integer degree tile origins (floor of coordinate)
REGIONS = [
    ("norway",      range(58, 72), range(5,  31)),   # Norwegian fjords
    ("swiss_alps",  range(45, 48), range(6,  12)),   # Swiss Alps
    ("iceland",     range(63, 67), range(-25, -12)), # Iceland (W longitudes)
    ("nz_south",    range(-46, -43), range(167, 174)), # NZ South Island (S lat)
]


def tile_key(lat: int, lon: int) -> tuple[str, str]:
    """Return (s3_key, filename) for a GLO-30 tile."""
    lat_tag = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_tag = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
    folder = f"Copernicus_DSM_COG_10_{lat_tag}_00_{lon_tag}_00_DEM"
    filename = f"{folder}.tif"
    key = f"{folder}/{filename}"
    return key, filename


def download_tile(s3, lat: int, lon: int, out_dir: Path) -> bool:
    """Download a single tile. Returns True if downloaded, False if skipped/missing."""
    key, filename = tile_key(lat, lon)
    dest = out_dir / filename

    if dest.exists():
        return False  # already have it

    try:
        s3.head_object(Bucket=BUCKET, Key=key)
    except s3.exceptions.ClientError:
        return False  # tile doesn't exist (ocean / no data)

    print(f"  downloading {filename} ...", end=" ", flush=True)
    s3.download_file(BUCKET, key, str(dest))
    size_mb = dest.stat().st_size / 1_000_000
    print(f"{size_mb:.1f} MB")
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="eu-central-1")

    total_downloaded = 0
    total_skipped = 0

    for region_name, lat_range, lon_range in REGIONS:
        print(f"\n=== {region_name} ===")
        for lat in lat_range:
            for lon in lon_range:
                downloaded = download_tile(s3, lat, lon, OUT_DIR)
                if downloaded:
                    total_downloaded += 1
                else:
                    total_skipped += 1

    print(f"\nDone. Downloaded: {total_downloaded}, Skipped (exists/ocean): {total_skipped}")
    files = list(OUT_DIR.glob("*.tif"))
    total_gb = sum(f.stat().st_size for f in files) / 1e9
    print(f"Total tiles on disk: {len(files)} ({total_gb:.2f} GB)")


if __name__ == "__main__":
    main()
