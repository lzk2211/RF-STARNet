<h1 align="center">Harmony in Time and Frequency: Lightweight Spectro-Temporal Disentanglement for UAV RF Fingerprinting</h1>

<div align="center">

[Zhenkun Liu](https://github.com/lzk2211),
[Yunling Liao](https://github.com/FishSkin2002),
[Zhiyv Dai](https://github.com/krismiledy) and 
[JunSheng Mu](https://scholar.google.com/citations?hl=zh-CN&user=ikS5u98AAAAJ)

Beijing University of Posts & Telecommunications.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-929292)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](#installation)
[![Shell](https://img.shields.io/badge/Shell-Scripts-green)](#installation)
[![GitHub - SRSpecNet](https://img.shields.io/badge/GitHub-SRSpecNet-000000?logo=github)](https://github.com/lzk2211/SRSpecNet)

</div>

<div align="center">
  <a href="graph/nn.pdf" target="_blank">
    <img src="graph/nn.jpg" width="800" alt="点击查看PDF">
  </a>
  <br>
  <!-- <p>点击上方图片或 <a href="assets/your_document.pdf">此处</a> 查看完整 PDF</p> -->
</div>

## 📣 News

- [04/Dec/2025] Initial public release of SRSpecNet repository.
- [xx/xxx/2025] Paper preprint and model weights to be released soon.
- [xx/xxx/2025] Benchmark scripts and dataset preparation utilities added.

## Abstract

Spectral super-resolution aims to reconstruct high-dimensional hyperspectral or multispectral information from conventional RGB or limited-band inputs. SRSpecNet addresses this problem with a unified, data-efficient network that leverages cross-modal priors and learned spectral bases to recover fine-grained spectral signatures while preserving spatial detail. Our approach emphasizes robust generalization across sensors and scenes through careful data preparation, multi-loss training, and evaluation on standard benchmarks. Extensive experiments demonstrate that SRSpecNet achieves state-of-the-art performance in spectral reconstruction and downstream tasks including material identification and remote sensing analysis.

<p align="center">
  <img src="assets/framework.png" alt="Framework" height="260"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/results.png" alt="Results" height="260">
</p>

## Our Framework

<div align="center">

<img src="assets/pipeline.png" width="800" alt="SRSpecNet pipeline">

</div>

SRSpecNet consists of:
- A spectral embedding module that maps inputs (e.g., RGB) to a latent spectral space.
- A reconstruction head that predicts dense spectral bands (e.g., 31-channel hyperspectral).
- Multi-scale spatial refinement to improve edge fidelity and texture details.
- Training losses combining spectral angle mapping (SAM), L1/L2, and perceptual consistency.

## Online Demo

- Coming soon: a hosted demo for quick evaluation and visualization.
- If you want to contribute a demo (Hugging Face Space or Gradio), please open an issue or a PR.

## Datasets

SRSpecNet supports common spectral SR datasets. Please ensure you comply with their licenses:
- ICVL Hyperspectral Dataset
- CAVE Hyperspectral Dataset
- NTIRE Spectral SR challenge datasets
- Custom datasets with RGB-to-HSI alignment

See [Data Preparation](#data-preparation-for-evaluation) for scripts and folder structures.

## Weights Download

- We will provide pre-trained weights for SRSpecNet soon.
- After download, place checkpoint files under `weights/`:
  - `srspecnet_rgb2hsi.pth`
  - `srspecnet_ms2hsi.pth`

Or run:
```bash
bash weights/download.sh
```
(Note: the script will be updated with public URLs once the weights are released.)

## Data Preparation for Evaluation

We provide helper scripts to organize datasets into a consistent structure.

<p></p> <details> <summary><b> ICVL </b></summary>

Download ICVL according to the official instructions, then organize as:

```bash
bash data/prepare_icvl.sh  # will normalize and place files under data/ICVL/
```

Expected layout:
```
data/
└── ICVL/
   ├── train/
   │   ├── rgb/
   │   └── hsi/  # 31 channels
   └── val/
       ├── rgb/
       └── hsi/
```

</details>
<p></p>

<p></p> <details> <summary><b> CAVE </b></summary>

Download CAVE dataset and arrange:

```bash
bash data/prepare_cave.sh
```

</details>
<p></p>

<p></p> <details> <summary><b> NTIRE </b></summary>

Follow NTIRE guidelines, then:

```bash
bash data/prepare_ntire.sh
```

</details>
<p></p>

### Data Structure

<p></p> <details> <summary><b>Organizing the Dataset</b></summary>

We recommend the following layout:

```
data/
├── ICVL/
│   ├── train/
│   │   ├── rgb/
│   │   └── hsi/
│   └── val/
│       ├── rgb/
│       └── hsi/
├── CAVE/
│   ├── train/
│   │   ├── rgb/
│   │   └── hsi/
│   └── val/
│       ├── rgb/
│       └── hsi/
└── NTIRE/
    ├── train/
    │   ├── rgb/
    │   └── hsi/
    └── val/
        ├── rgb/
        └── hsi/
```

</details>
<p></p>

## Installation and Environment Setup

The repository is primarily Python (97.5%) with some Shell utilities.

- Clone the repository:
```bash
git clone https://github.com/lzk2211/SRSpecNet.git
cd SRSpecNet
```

- Create environment:
```bash
conda env create -f environment.yaml
conda activate srspecnet
```

- Initialize submodules if present:
```bash
git submodule update --init --recursive
git submodule update --recursive --remote
```

## Quick Start

- Run a demo after placing weights:
```bash
python demo.py --input demo/rgb.png --save_dir ./demo --ckpt weights/srspecnet_rgb2hsi.pth
```

- Visualize spectra at selected pixels:
```bash
python tools/visualize_spectra.py --hsi demo/output_hsi.npy --coords 120,200 64,64
```

## Evaluation

We provide evaluation scripts for spectral reconstruction and downstream metrics.

### Test on Validation Sets

```bash
python test_spectral_sr.py --dataset icvl --rgb_dir data/ICVL/val/rgb --hsi_dir data/ICVL/val/hsi --ckpt weights/srspecnet_rgb2hsi.pth --save_dir ./eval/icvl

python test_spectral_sr.py --dataset cave --rgb_dir data/CAVE/val/rgb --hsi_dir data/CAVE/val/hsi --ckpt weights/srspecnet_rgb2hsi.pth --save_dir ./eval/cave
```

Supported metrics:
- RMSE, PSNR (per band and averaged)
- SAM (Spectral Angle Mapper)
- MAE
- Optional perceptual metrics for RGB reconstruction

## Figures and Graphs

The repository includes publication-ready figures under `graph/` (EPS/PDF). EPS/PDF are optimized for LaTeX and journal submissions; preview locally or convert to PNG if needed.

- Acc–FLOPs/Params: `graph/markers_chart_flops.eps`, `graph/markers_chart_params.eps`
- SNR curves (macro metrics): `graph/acc.eps`, `graph/precision.eps`, `graph/recall.eps`, `graph/f1.eps`
- Mosaic diffs (Fusion vs branches): `graph/mosaic_diff_fusion_minus_HARNet.pdf`, `graph/mosaic_diff_fusion_minus_SSTA.pdf`
- Other diagrams: `graph/HARNet.pdf`, `graph/SSTA.pdf`, `graph/nn.pdf`, `graph/methodologies.pdf`, `graph/result.pdf`, `graph/Loss.pdf`
- Legacy chart (kept for reference): `graph/bubble_chart_high_contrast.eps`

Notes:
- SNR labels display `10dB` (formerly shown as `pure`) for consistency across all plots.
- Figures are sized for single-column placement in two-column papers (Times New Roman fonts, compact legends).

### Reproduce the Figures

Generate the model scatter charts (accuracy vs FLOPs/Params):

```bash
python plot_graph/markers.py
```

Generate the per-SNR metric curves (accuracy/precision/recall/F1):

```bash
python plot_graph/plot_graph4_5test.py
```

Produce mosaic and diff-mosaic plots via the test pipeline (artifacts are saved under your experiment folder; you may copy selected PDFs to `graph/`):

```bash
python main.py --mode Test --model <YourModelName> \
  --exp_name <your/exp/folder> --seed 42
```

Optional: convert EPS/PDF to PNG for quick previews (requires Ghostscript):

```bash
gs -sDEVICE=pngalpha -o graph/markers_chart_flops.png -r300 graph/markers_chart_flops.eps
gs -sDEVICE=pngalpha -o graph/markers_chart_params.png -r300 graph/markers_chart_params.eps
```

### Command Line Options

Common flags:
- `--dataset [icvl|cave|ntire|custom]`
- `--rgb_dir PATH`
- `--hsi_dir PATH`
- `--ckpt PATH`
- `--save_dir PATH`
- `--save_figs` to save visualization images
- `--bands 31` for target spectral channels

## Training

See [Training Guide](./train_orders/README.md) for details.

Typical command:
```bash
python train.py \
  --dataset icvl \
  --rgb_dir data/ICVL/train/rgb \
  --hsi_dir data/ICVL/train/hsi \
  --val_rgb_dir data/ICVL/val/rgb \
  --val_hsi_dir data/ICVL/val/hsi \
  --epochs 200 \
  --batch_size 8 \
  --lr 2e-4 \
  --save_dir runs/icvl_srspecnet
```

Notes:
- Adjust learning rate and batch size based on GPU memory.
- Use mixed precision (`--amp`) for faster training.
- Enable spectral losses in `configs/loss.yaml`.

## TODO List

- [x] Initial Release
- [x] Evaluation Scripts
- [x] Training Code
- [ ] Pre-trained Weights
- [ ] Online Demo
- [ ] Extended Benchmarks and More Datasets
- [ ] Model Zoo and Ablations

## Acknowledgement

We thank the maintainers and authors of spectral SR datasets and baselines. This repository includes inspiration and utilities from:
- Spectral SR community datasets (ICVL, CAVE, NTIRE)
- PyTorch ecosystem and scientific computing libraries

If any third-party components are used under specific licenses, please refer to their respective repositories and comply accordingly.

## Citation

If you find SRSpecNet useful, please consider citing:

```bibtex
@inproceedings{your2025srspecnet,
  title={SRSpecNet: Spectral Super-Resolution and Reconstruction Network},
  author={Your Name and Collaborators},
  booktitle={Proceedings of the Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

Update this entry with your actual paper details when available.

## License

This repository is under the [Apache-2.0 license](./LICENSE).

Please review third-party dataset and baseline licenses before use.
