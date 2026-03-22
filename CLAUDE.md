# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Terrain Generator for Fantasy Worldbuilding. Users paint sparse geographic labels onto a canvas; a GAN fills in a physically plausible heightmap, rendered as a topographic/fantasy map.

## Architecture

### System Components

```
data_pipeline/   — DEM download, preprocessing, label auto-derivation, sparsification
training/        — pix2pix GAN (U-Net generator + PatchGAN discriminator)
rendering/       — algorithmic hillshade + elevation colormap (no model)
backend/         — Flask inference server
frontend/        — single HTML/JS canvas painting UI
```

### Data Flow

1. **Data pipeline** downloads Copernicus GLO-30 GeoTIFFs → slices into 256×256 patches → auto-derives full label maps → sparsifies to ~4% of pixels → saves `(sparse_label.npy, heightmap.npy)` pairs (~150k pairs, ~10-15GB)
2. **Training** consumes those pairs to train a pix2pix GAN (input: sparse RGB label map → output: normalized heightmap float)
3. **Backend** loads trained model, exposes a `/generate` endpoint (receives base64 PNG label map, returns base64 PNG heightmap)
4. **Rendering** runs algorithmically on the returned heightmap (hillshading via surface normals + elevation colormap) — not part of the model
5. **Frontend** single HTML file: canvas with 5 brushes → POST to backend → render result

### Label Color Encoding

| Terrain    | Color  | Derivation rule                              |
|------------|--------|----------------------------------------------|
| Ocean      | Blue   | elevation < 0.3                              |
| Mountain   | Red    | local maxima above 0.75, dilated             |
| River      | Cyan   | flow accumulation > threshold (D8 method)    |
| Forest     | Green  | mid-elevation non-peak land                  |
| Desert     | Yellow | (user-painted; auto-derivation TBD)          |

### Model

- **MVP**: pix2pix — U-Net generator + PatchGAN discriminator
- **Input**: 256×256 sparse RGB label map
- **Output**: 256×256 normalized heightmap (0–1 float)
- Training uses variable sparsity (keep_ratio 1–20% randomly per sample) so the model is robust to different painting styles
- **V2**: ControlNet-style diffusion for inpainting support

### Training Config

- Resolution: 256×256, Batch: 32, Epochs: 50
- Adam lr=2e-4, betas=(0.5, 0.999)
- Loss: BCE (adversarial) + L1×100 (reconstruction)
- Hardware: NVIDIA A40 (48GB VRAM), ~6–8 hours

## Key Libraries

| Purpose              | Library              |
|----------------------|----------------------|
| GeoTIFF loading      | rasterio             |
| Flow accumulation    | richdem              |
| Morphological ops    | scipy                |
| AWS S3 download      | boto3                |
| Synthetic augment    | noise (Perlin)       |
| Backend              | Flask                |
| Model training       | PyTorch              |

## Expected Commands

These will be established as the project is built — placeholder patterns:

```bash
# Data pipeline
python data_pipeline/download.py       # download GeoTIFF tiles from Copernicus S3
python data_pipeline/preprocess.py     # slice, normalize, derive labels, sparsify
python data_pipeline/stats.py          # inspect dataset stats / sample pairs

# Training
python training/train.py               # launch training run
python training/eval.py                # evaluate on held-out samples

# Backend
flask --app backend/app.py run         # start inference server (dev)
python backend/app.py                  # alternative

# Frontend
# Open frontend/index.html directly in browser — no build step
```

## MVP Scope

**In**: canvas UI with 5 brushes, Flask backend, topographic rendering (hillshade + colormap), 256×256, full-map generation on button press.

**Out**: regional inpainting, parchment style, multi-scale, diffusion model, export options.

## Data Source

Primary: Copernicus GLO-30 (30m resolution, free via AWS S3). Backup: SRTM (90m). Target regions: Norwegian fjords, Swiss Alps, NZ South Island, Iceland — areas with high relief variation.

## V2 Roadmap

ControlNet diffusion model, regional inpainting, parchment rendering style, 512×512+, export formats (heightmap PNG, SVG, Unity/Unreal).
