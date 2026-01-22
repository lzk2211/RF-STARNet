<!-- <h1 align="center">
  <img src="graph/title.png">
</h1> -->

# RF-STARNet: A Multi-modal Lightweight Fusion Framework of UAV RF Fingerprinting Identification for Public Low-Altitude Safety

<div align="center">

<!-- [Zhenkun Liu](https://github.com/lzk2211),
[Yunling Liao](https://github.com/FishSkin2002),
[Zhiyv Dai](https://github.com/krismiledy) and 
[JunSheng Mu](https://scholar.google.com/citations?hl=zh-CN&user=ikS5u98AAAAJ)
(Provisional) -->

Beijing University of Posts & Telecommunications.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-929292)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](#installation)
[![Pytorch](https://img.shields.io/badge/Pytorch-2.4.0-red)](#installation)
[![Shell](https://img.shields.io/badge/Shell-Scripts-green)](#installation)
[![GitHub - RF-STARNet](https://img.shields.io/badge/Github-RFSTARNet-pink?logo=github)](https://github.com/lzk2211/RF-STARNet)

</div>

<div align="center">
  <a href="graph/methodologies.pdf" target="_blank">
    <img src="graph/methodologies.jpg" width="800" alt="点击查看PDF">
  </a>
  <br>
  <p>click <a href="graph/methodologies.pdf">here</a> to see the PDF</p>
</div>

## 📣 News

- [06/Nov/2025] Initial public release of RF-STARNet repository.
- [20/Nov/2025] Complete the experiments for the proposed innovations, design rigorous comparative experiments, and begin preparing figures for this paper.
- [04/Dec/2025] The main body of the paper is essentially complete. Detailed updates have been made to the GitHub repository.
- [22/Jan/2026] Finish the paper writing, and update Github repository.
<!-- - [xx/xxx/2025] Paper preprint and model weights to be released soon.
- [xx/xxx/2025] Benchmark scripts and dataset preparation utilities added. -->

## 📝 Abstract(Updating)

With the rapid increase of Low-Altitude Wireless Networks (LAWN), the effective management and regulation of Unmanned Aerial Vehicles (UAVs) have become critically important. Radio Frequency (RF) fingerprinting has emerged as a key technology for UAV detection, witnessing a paradigm shift from traditional model-driven approaches to data-driven deep learning methods. 
However, existing methods fall short of the efficiency required for agile deployment. They treat RF signal or spectrograms as generic sequences or images, thereby overlooking the unique stationary behaviors inherent in the signals. Instead, they tend to over-learn massive dynamic redundant features, which consequently limits identification performance and increase model complexity.
To address these, we propose a lightweight **R**adio-**F**requency **S**pectro-**T**emporal **A**ttention **R**esidual **Net**work (**RF-STARNet**), which leverages dual-stream spectro-temporal learning to extract structured features for efficient detection. Specifically, we design a Spectral-Aware Residual Network (**SARNet**) to extract sparse, stationary spectral patterns. Simultaneously, a Sparse Sub-band Temporal Attention (**SSTA**) mechanism is used to capture global temporal autocorrelations and cyclostationary features.
To deeply integrate these heterogeneous features, we propose an Adaptive Gating Mechanism to dynamically calibrate branch contributions, augmented by a novel deep supervision strategy that enforces feature diversity to ensure the learning of complementary discriminative knowledge.
Experimental results demonstrate that RF-STARNet achieves a superior accuracy of 81.44\% with an ultra-low computational cost of 3.8 MFLOPs, exhibiting remarkable robustness even in low Signal-to-Noise Ratio (SNR) environments.

## 🏗️ Framework

The overall architecture of RF-STARNet consists of three main stages:

<div align="center">
  <a href="graph/nn.pdf" target="_blank">
    <img src="graph/nn.jpg" width="800" alt="点击查看PDF">
  </a>
  <br>
  <p>click <a href="graph/nn.pdf">here</a> to see the PDF</p>
</div>

- **SARNet (Spectral Branch)**: Extracts time-averaged PSD features using multi-dilation convolutions to capture micro-scale and macro-scale spectral details.
- **SSTA (Temporal Branch)**: Utilizes local average pooling and grouped self-attention with relative position bias to model long-range temporal dependencies (e.g., OFDM symbol periodicity, FHSS hopping).
- **Gated Fusion & Deep Supervision**: Uses robust spectral features to guide temporal activation and employs multi-loss optimization (Main + Aux + Orthogonal) to ensure feature complementarity.

## 📂 Dataset

We use the **[DroneRFa](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT230570)** dataset for evaluation.

- **Signals**: Raw I/Q signals collected via USRP.
- **Preprocessing**: Signals are sliced into 50ms segments and transformed into **512x512 Spectrograms** via STFT.
- **Categories**: 10 classes (9 Commercial UAVs + 1 Background Noise).

**Data Structure**

Please organize the dataset as follows:

```bash
data_set_5e6/
├── T0000/
├── T0001/
├── T0010/
├── ...
└── T1001/
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/lzk2211/RF-STARNet.git
cd RF-STARNet
```
2. Install dependencies:
```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 🚀 Usage

The main entry point is main.py. You can switch between training and testing modes using command line arguments.

**Training**

To train the **RF-STARNet (Small version)** with Multi-Loss Deep Supervision:

```bash
python main.py \
  --mode Train \
  --model F_T_small_mulitloss\
  --exp_name F_T_small_mulitloss \
  --root_path /path/to/your/data_set_5e6 \
  --batch_size 64 \
  --epochs 250 \
  --lr 0.0001 \
  --seed 42
```

Key Arguments:

- `--epochs`:         Number of training epochs
- `--lr`:             Learning rate
- `--weight_decay`:   Weight decay for optimizer
- `--batch_size`:     Batch size
- `--resume`:         train from checkpoint
- `--ratio`:          ratio
- `--mode`:           switch mode
- `--exp_name`:       the name of the experiment
- `--class_num`:      class number
- `--model`:          switch model
- `--root_path`:      dataset path
- `--seed`:           Random seed for reproducibility
- `--device`:         Device to use for training/testing



## 📊 Results

Performance comparison on DroneRFa dataset (Avg Accuracy vs. Efficiency)

### 1. **Performance Superiority**
The proposed dual-stream fusion network, RF-STARNet, demonstrates superior performance across all evaluation metrics compared to its individual sub-branches (SARNet and SSTA). specifically, RF-STARNet achieves an average accuracy of **81.44%**, outperforming the standalone SARNet and SSTA branches by *3.63%* and *5.65%*, respectively. This validates the effectiveness of fusing temporal and spectral features, leveraging the complementary strengths of CNN-based and Transformer-based architectures.

<div align="center">
<table>
  <tr>
    <td style="text-align:center">
      <img src="graph/acc.png" width="400" alt="Fusion vs SARNet">
      <br>
      <b>(a) Accuracy</b>
    </td>
    <td style="text-align:center">
      <img src="graph/precision.png" width="400" alt="Fusion vs SSTA">
      <br>
      <b>(b) Precision</b>
    </td>
  </tr>

  <tr>
    <td style="text-align:center">
      <img src="graph/recall.png" width="400" alt="Fusion vs SSTA">
      <br>
      <b>(c) Recall</b> </td>
    <td style="text-align:center">
      <img src="graph/f1.png" width="400" alt="Fusion vs SSTA">
      <br>
      <b>(d) F1-score</b> </td>
  </tr>
</table>
</div>

### 2. Efficiency vs. Accuracy Trade-off
The trade-off between model accuracy and computational complexity. 

Lightweight Design: RF-STARNet maintains an extremely low parameter count (**0.188 M**) and computational cost (**3.80 M MACs**). Compared to the heavyweight ResNet18 (11.18 M Params), our model reduces the model size by approximately 98% while achieving a higher accuracy (**81.44% vs. 76.11%**).

Inference Speed: The optimized architecture allows for rapid inference (**3.65s on GPU**), making it highly suitable for real-time drone detection on edge devices. The high Model Efficiency Index (**MEI = 6.87**) further confirms that RF-STARNet achieves the best balance between performance and resource consumption among all compared methods.

<div align="center">
<table>
<tr>
<td style="text-align:center">
<img src="graph/markers_chart_flops.png" width="400" alt="Fusion vs SARNet">

<b>(a) The Average Accuracy vs. Computational Cost</b>
</td>
<td style="text-align:center">
<img src="graph/markers_chart_params.png" width="400" alt="Fusion vs SSTA">

<b>(b) The Average Accuracy vs. Model Size</b>
</td>
</tr>
</table>
</div>

### 3. Comprehensive Analysis: Ablation Study & Performance Comparison (Small Scale Focus)

### Ablation study and performance comparison

**Part I**: Impact of specific modules on the baseline architecture.  
**Part II**: Benchmarking against comparison methods.

| Model / Configuration | Avg Acc (%) | Params (M) | MACs (M) | MEI (Score) | GPU Infer. Time (ms) | CPU Infer. Time (ms) | Memory (MB) | Train Cost (GPU/h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Part I: Module Ablation Study** | | | | | | | | |
| *1. Frequency Domain Architecture: SARNet* | | | | | | | | |
| Baseline | 69.43 | 0.019 | 0.59 | 208.32 | 0.70 | 0.97 | 0.18 | 0.09 |
| w/ SE block | 70.29 <span style="color:red">(+0.86%)</span> | 0.031 | 0.68 | 207.84 <span style="color:green">(-0.23%)</span> | 0.68 | 0.96 | 0.18 | 0.09 |
| w/ MRFSM | 72.53 <span style="color:red">(+3.10%)</span> | 0.019 | 0.59 | 227.34 <span style="color:red">(+8.37%)</span> | 0.69 | 1.03 | 0.18 | 0.11 |
| **SARNet (Proposed)** | **77.81** <span style="color:red">(+8.38%)</span> | 0.111 | 2.01 | **231.69** <span style="color:red">(+10.09%)</span> | 0.68 | 1.07 | 0.19 | 0.09 |
| *2. Time Domain Architecture: SSTA* | | | | | | | | |
| Baseline (RPB) | 74.18 | 0.550 | 273.70 | 168.56 | 0.55 | 2.93 | 1.11 | 0.14 |
| w/ Stem | 74.33 <span style="color:red">(+0.15%)</span> | 0.550 | 34.26 | 180.75 <span style="color:red">(+6.74%)</span> | 0.68 | 1.15 | 0.29 | 0.12 |
| w/ GSA | 75.21 <span style="color:red">(+1.03%)</span> | 0.042 | 13.65 | 208.92 <span style="color:red">(+19.32%)</span> | 1.05 | 42.98 | 1.03 | 0.48 |
| **SSTA (Proposed)** | **75.79** <span style="color:red">(+1.61%)</span> | 0.042 | 1.75 | **229.58** <span style="color:red">(+26.58%)</span> | 0.68 | 2.23 | 0.27 | 0.06 |
| *3. Fusion Architecture: RF-STARNet* | | | | | | | | |
| Baseline | 78.89 | 0.144 | 3.75 | 230.38 | 0.72 | 2.84 | 0.27 | 0.11 |
| w/ AGM | 80.41 <span style="color:red">(+1.52%)</span> | 0.177 | 3.79 | 237.43 <span style="color:red">(+2.97%)</span> | 0.70 | 4.06 | 0.27 | 0.11 |
| w/ Multi-Loss | 79.32 <span style="color:red">(+0.43%)</span> | 0.154 | 3.76 | 232.30 <span style="color:red">(+0.83%)</span> | 0.71 | 2.57 | 0.29 | 0.21 |
| **RF-STARNet (Proposed)** | **81.44** <span style="color:red">(+2.55%)</span> | 0.188 | 3.80 | **242.99** <span style="color:red">(+5.19%)</span> | 0.55 | 2.86 | 0.27 | 0.16 |
| **Part II: Benchmark Comparison** | | | | | | | | |
| MobileNetV3 | 77.36 | 0.930 | 316.03 | 179.64 | 0.85 | 19.08 | 2.12 | 0.37 |
| ResNet18 | 76.11 | 11.180 | 9115.93 | 147.91 | 2.27 | - | 5.17 | 0.33 |
| ResNet-1D | 73.92 | 0.135 | 3.54 | 203.18 | 0.52 | 1.08 | 0.19 | 0.10 |
| S3R | 72.04 | 34.530 | 547.85 | 138.47 | 1.63 | - | 1.68 | 0.60 |

## 🔬 Analysis & Visualization

To investigate the internal working mechanism of RF-STARNet, we provide detailed visualizations of feature activation and error distribution.

### 1. Class Activation Mapping (CAM)

We utilize CAM to visualize the regions of interest for different network branches.

<div align="center">
  <a href="graph/result.pdf" target="_blank">
    <img src="graph/result.jpg" width="800" alt="点击查看PDF">
  </a>
  <br>
  <p>click <a href="graph/result.pdf">here</a> to see the PDF</p>
</div>

- **SARNet**: Displays distinct horizontal stripe patterns, indicating its focus on positional information of critical frequency blocks.

- **SSTA**: Leverages global temporal correlations to produce precise masks covering all relevant frequency blocks (e.g., VTS and FCS signals), demonstrating superior noise resistance.

- **2D-CNN**: Tends to perceive frequency blocks as a single holistic object, lacking signal-specific refinement compared to our specialized branches.

### 2. Mosaic Error Analysis

To validate the synergistic effect of the dual-branch fusion, we conducted a detailed error analysis using mosaic plots. Green pixels indicate samples where the fused model corrected the error of a single branch.

<div align="center">
<table>
<tr>
<td style="text-align:center">
<img src="graph/mosaic_diff_fusion_minus_SARNet.jpg" width="400" alt="Fusion vs SARNet">

<b>(a) Fusion vs. SARNet Branch</b>
</td>
<td style="text-align:center">
<img src="graph/mosaic_diff_fusion_minus_SSTA.jpg" width="400" alt="Fusion vs SSTA">

<b>(b) Fusion vs. SSTA Branch</b>
</td>
</tr>
</table>
</div>

- **Fig (a)**: RF-STARNet compensates for SARNet's deficiencies in **lower SNR regions** (dense green points), proving that temporal features are crucial when spectral features are corrupted by noise.

- **Fig (b)**: While SSTA may perform suboptimally at high SNRs, the fused model maintains high accuracy by leveraging SARNet. In **low-SNR conditions**, RF-STARNet effectively exploits SSTA's robust global correlation features to ensure stability.

**Conclusion**: In this paper, we propose RF-STARNet, a multi-model lightweight architecture designed to address the trade-off between accuracy and computation overhead in UAV RF fingerprinting. By effectively decoupling the signal processing into stationary spectral and cyclostationary temporal domains, the proposed SARNet and SSTA modules extract sparse, complementary fingerprints with a lower overhead. Furthermore, the integration of an adaptive gating mechanism and a deep supervision strategy ensures robust feature fusion.
Extensive experiments validate that SARNet, SSTA, and RF-STARNet achieve a superior balance between high recognition accuracy and low computational cost, exhibiting remarkable efficiency and robustness compared to comparison methods. These attributes make the proposed methods highly suitable for edge deployment. Future work will focus on the embedded implementation of this framework in complex, real-world electromagnetic environments.


## 🔗 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{liu2025harmony,
  title={RF-STARNet: A Dual-Stream Lightweight Framework for UAV RF Fingerprinting Identification},
  author={Zhenkun Liu, Yunling Liao, Zhiyv Dai and JunSheng Mu},
  journal={arXiv preprint},
  year={2025}
}
```
## License

This repository is under the [Apache-2.0 license](./LICENSE).
