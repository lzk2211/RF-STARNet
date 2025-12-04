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
  <a href="graph/mothed.pdf" target="_blank">
    <img src="graph/mothed.jpg" width="800" alt="点击查看PDF">
  </a>
  <br>
  <p>click <a href="graph/mothed.pdf">here</a> to see the PDF</p>
</div>

## 📣 News

- [06/Nov/2025] Initial public release of SRSpecNet repository.
- [20/Nov/2025] Complete the experiments for the proposed innovations, design rigorous comparative experiments, and begin preparing figures for this paper.
- [04/Dec/2025] The main body of the paper is essentially complete. Detailed updates have been made to the GitHub repository.
<!-- - [xx/xxx/2025] Paper preprint and model weights to be released soon.
- [xx/xxx/2025] Benchmark scripts and dataset preparation utilities added. -->

## 📝 Abstract(Updating)

Radio Frequency (RF) fingerprinting is a promising technique for Unmanned Aerial Vehicle (UAV) identification. However, existing methods often struggle to balance recognition accuracy with computational efficiency, especially under low Signal-to-Noise Ratio (SNR) conditions.

This paper proposes **H-starNet**, a lightweight **H**armonic-aware **S**pectro-**T**emporal **A**ttention **R**esidual **Net**work. Our approach disentangles the RF signals into two complementary domains:

- **Spectral Branch (HARNet)**: Captures stationary frequency fingerprints (e.g., filter shapes) using a Multi-Resolution Frequency Sparse Module (MRFSM).
- **Temporal Branch (SSTA)**: Captures dynamic temporal patterns (e.g., hopping sequences) using Sparse Sub-band Temporal Attention.

Coupled with a **Gated Cross-Domain** Fusion mechanism and a **Deep Supervision** training strategy, H-starNet achieves state-of-the-art accuracy with significantly lower computational costs compared to traditional CNNs and Transformers.

## 🏗️ Framework

The overall architecture of H-starNet consists of three main stages:

<div align="center">
  <a href="graph/nn.pdf" target="_blank">
    <img src="graph/nn.jpg" width="800" alt="点击查看PDF">
  </a>
  <br>
  <p>click <a href="graph/nn.pdf">here</a> to see the PDF</p>
</div>

- **HARNet (Spectral Branch)**: Extracts time-averaged PSD features using multi-dilation convolutions to capture micro-scale and macro-scale spectral details.
- **SSTA (Temporal Branch)**: Utilizes local average pooling and grouped self-attention with relative position bias to model long-range temporal dependencies (e.g., OFDM symbol periodicity, FHSS hopping).
- **Gated Fusion & Deep Supervision**: Uses robust spectral features to guide temporal activation and employs multi-loss optimization (Main + Aux + Orthogonal) to ensure feature complementarity.

📂 Dataset

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
git clone [https://github.com/lzk2211/SRSpecNet.git](https://github.com/lzk2211/SRSpecNet.git)
cd SRSpecNet
```
2. Install dependencies:
```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 🚀 Usage

The main entry point is main.py. You can switch between training and testing modes using command line arguments.

**Training**

To train the **H-starNet (Small version)** with Multi-Loss Deep Supervision:

```bash
python main.py \
  --mode Train \
  --model F_T_small_mulitloss_1122 \
  --exp_name F_T_small_mulitloss_1122 \
  --root_path /path/to/your/data_set_5e6 \
  --batch_size 64 \
  --epochs 250 \
  --lr 0.0001 \
  --seed 42
```

Key Arguments:

- `--model`: The model factory function name (e.g., F_T for fusion model).
- `--exp_name`: The runing results save in this project.
- `--root_path`: Path to the preprocessed spectrogram dataset.

## 📊 Results

Performance comparison on DroneRFa dataset (Avg Accuracy vs. Efficiency):

### Comprehensive Analysis: Ablation Study & Performance Comparison (Small Scale Focus)

**Table Caption:** Part I details the impact of specific modules on the Small architecture relative to the baseline. Part II compares our final models against state-of-the-art methods. MEI is calculated as $Acc / \log(Params \times MACs)$.

<table>
    <caption>
        <strong>Comprehensive Analysis: Ablation Study & Performance Comparison (Small Scale Focus).</strong><br>
        Part I details the impact of specific modules on the Small architecture relative to the baseline.
        Part II compares our final models against state-of-the-art methods.<br>
        <em>MEI is calculated as Acc / log(Params * MACs).</em>
    </caption>
    <thead>
        <tr>
            <th style="text-align:left;">Model / Configuration</th>
            <th style="text-align:center;">Avg Acc<br>(%)</th>
            <th style="text-align:center;">Params<br>(M)</th>
            <th style="text-align:center;">MACs<br>(M)</th>
            <th style="text-align:center;">MEI<br>(Score)</th>
            <th style="text-align:center;">GPU Infer.<br>Time (s)</th>
            <th style="text-align:center;">CPU Infer.<br>Time (s)</th>
            <th style="text-align:center;">Memory<br>(MB)</th>
            <th style="text-align:center;">Train Cost<br>(GPU/h)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan="9" style="text-align:center; background-color:#f6f8fa; font-weight:bold;">
                Part I: Module Ablation Study
            </td>
        </tr>
        
        <tr><td colspan="9" style="text-align:left; font-style:italic;"><strong>1. HARNet Architecture</strong></td></tr>
        <tr>
            <td>Baseline</td>
            <td align="center">69.43</td>
            <td align="center">0.019</td>
            <td align="center">0.59</td>
            <td align="center">6.90</td>
            <td align="center">4.68</td>
            <td align="center">6.49</td>
            <td align="center">1.21</td>
            <td align="center">0.62</td>
        </tr>
        <tr>
            <td>w/ SE Block</td>
            <td align="center">70.29 (+0.86%)</td>
            <td align="center">0.031</td>
            <td align="center">0.68</td>
            <td align="center">6.81 (-1.30%)</td>
            <td align="center">4.56</td>
            <td align="center">6.44</td>
            <td align="center">1.21</td>
            <td align="center">0.60</td>
        </tr>
        <tr>
            <td>w/ MRFSM</td>
            <td align="center">72.53 (+3.10%)</td>
            <td align="center">0.019</td>
            <td align="center">0.59</td>
            <td align="center"><strong>7.21</strong> (+4.49%)</td>
            <td align="center">4.62</td>
            <td align="center">6.87</td>
            <td align="center">1.20</td>
            <td align="center">0.76</td>
        </tr>
        <tr>
            <td><strong>HARNet (Ours)</strong></td>
            <td align="center"><strong>77.81</strong> (+8.38%)</td>
            <td align="center">0.111</td>
            <td align="center">2.01</td>
            <td align="center">6.86 (-0.58%)</td>
            <td align="center">4.54</td>
            <td align="center">7.13</td>
            <td align="center">1.29</td>
            <td align="center">0.60</td>
        </tr>

        <tr><td colspan="9" style="text-align:left; font-style:italic;"><strong>2. SSTA Architecture</strong></td></tr>
        <tr>
            <td>Baseline</td>
            <td align="center">74.18</td>
            <td align="center">0.550</td>
            <td align="center">273.70</td>
            <td align="center">5.23</td>
            <td align="center">3.66</td>
            <td align="center">19.63</td>
            <td align="center">7.44</td>
            <td align="center">0.95</td>
        </tr>
        <tr>
            <td>Abs. Pos. Enc.</td>
            <td align="center">30.81 (-43.37%)</td>
            <td align="center">0.042</td>
            <td align="center">1.75</td>
            <td align="center">2.83 (-45.89%)</td>
            <td align="center">4.76</td>
            <td align="center">21.74</td>
            <td align="center">1.78</td>
            <td align="center">0.45</td>
        </tr>
        <tr>
            <td>w/ Stem</td>
            <td align="center">74.33 (+0.15%)</td>
            <td align="center">0.550</td>
            <td align="center">34.26</td>
            <td align="center">5.60 (+7.07%)</td>
            <td align="center">4.55</td>
            <td align="center">7.69</td>
            <td align="center">1.93</td>
            <td align="center">0.78</td>
        </tr>
        <tr>
            <td>w/ Grouped</td>
            <td align="center">75.21 (+1.03%)</td>
            <td align="center">0.042</td>
            <td align="center">13.65</td>
            <td align="center">6.40 (+22.37%)</td>
            <td align="center">7.06</td>
            <td align="center">287.63</td>
            <td align="center">6.86</td>
            <td align="center">3.19</td>
        </tr>
        <tr>
            <td><strong>SSTA (Ours)</strong></td>
            <td align="center"><strong>75.79</strong> (+1.61%)</td>
            <td align="center">0.042</td>
            <td align="center">1.75</td>
            <td align="center"><strong>6.97</strong> (+33.27%)</td>
            <td align="center">4.52</td>
            <td align="center">14.91</td>
            <td align="center">1.79</td>
            <td align="center">0.37</td>
        </tr>

        <tr><td colspan="9" style="text-align:left; font-style:italic;"><strong>3. H-starNet (Fusion) Architecture</strong></td></tr>
        <tr>
            <td>Baseline</td>
            <td align="center">78.89</td>
            <td align="center">0.144</td>
            <td align="center">3.75</td>
            <td align="center">6.72</td>
            <td align="center">4.81</td>
            <td align="center">19.02</td>
            <td align="center">1.80</td>
            <td align="center">0.73</td>
        </tr>
        <tr>
            <td>w/ Gating Mechanism</td>
            <td align="center">80.41 (+1.52%)</td>
            <td align="center">0.177</td>
            <td align="center">3.79</td>
            <td align="center">6.80 (+1.19%)</td>
            <td align="center">4.71</td>
            <td align="center">27.16</td>
            <td align="center">1.80</td>
            <td align="center">0.72</td>
        </tr>
        <tr>
            <td>w/ Multi-Loss</td>
            <td align="center">79.32 (+0.43%)</td>
            <td align="center">0.154</td>
            <td align="center">3.76</td>
            <td align="center">6.74 (+0.30%)</td>
            <td align="center">4.72</td>
            <td align="center">17.18</td>
            <td align="center">1.93</td>
            <td align="center">1.38</td>
        </tr>
        <tr>
            <td><strong>H-starNet (Ours)</strong></td>
            <td align="center"><strong>81.44</strong> (+2.55%)</td>
            <td align="center">0.188</td>
            <td align="center">3.80</td>
            <td align="center"><strong>6.87</strong> (+2.23%)</td>
            <td align="center">3.65</td>
            <td align="center">19.12</td>
            <td align="center">1.80</td>
            <td align="center">1.07</td>
        </tr>

        <tr>
            <td colspan="9" style="text-align:center; background-color:#f6f8fa; font-weight:bold;">
                Part II: Benchmark Comparison
            </td>
        </tr>
        <tr>
            <td>MobileNetV3-Small [b7]</td>
            <td align="center">77.36</td>
            <td align="center">0.930</td>
            <td align="center">316.03</td>
            <td align="center">5.35</td>
            <td align="center">5.67</td>
            <td align="center">-</td>
            <td align="center">14.19</td>
            <td align="center">-</td>
        </tr>
        <tr>
            <td>ResNet18</td>
            <td align="center">76.11</td>
            <td align="center">11.180</td>
            <td align="center">9115.93</td>
            <td align="center">4.48</td>
            <td align="center">15.21</td>
            <td align="center">-</td>
            <td align="center">34.59</td>
            <td align="center">-</td>
        </tr>
        <tr>
            <td>1D-LW-ResNet18 [b8]</td>
            <td align="center">73.92</td>
            <td align="center">0.135</td>
            <td align="center">3.54</td>
            <td align="center">6.33</td>
            <td align="center">3.51</td>
            <td align="center">7.22</td>
            <td align="center">1.29</td>
            <td align="center">-</td>
        </tr>
        <tr>
            <td>S3R [b6]</td>
            <td align="center">72.04</td>
            <td align="center">34.530</td>
            <td align="center">547.85</td>
            <td align="center">4.43</td>
            <td align="center">10.93</td>
            <td align="center">-</td>
            <td align="center">11.26</td>
            <td align="center">-</td>
        </tr>
    </tbody>
</table>
## 🔬 Analysis & Visualization

To investigate the internal working mechanism of H-starNet, we provide detailed visualizations of feature activation and error distribution.

### 1. Class Activation Mapping (CAM)

We utilize CAM to visualize the regions of interest for different network branches.

<div align="center">
  <a href="graph/result.pdf" target="_blank">
    <img src="graph/result.jpg" width="800" alt="点击查看PDF">
  </a>
  <br>
  <p>click <a href="graph/result.pdf">here</a> to see the PDF</p>
</div>

- **HARNet**: Displays distinct horizontal stripe patterns, indicating its focus on positional information of critical frequency blocks.

- **SSTA**: Leverages global temporal correlations to produce precise masks covering all relevant frequency blocks (e.g., VTS and FCS signals), demonstrating superior noise resistance.

- **2D-CNN**: Tends to perceive frequency blocks as a single holistic object, lacking signal-specific refinement compared to our specialized branches.

### 2. Mosaic Error Analysis

To validate the synergistic effect of the dual-branch fusion, we conducted a detailed error analysis using mosaic plots. Green pixels indicate samples where the fused model corrected the error of a single branch.

<div align="center">
<table>
<tr>
<td style="text-align:center">
<img src="graph/mosaic_diff_fusion_minus_HARNet.jpg" width="400" alt="Fusion vs HARNet">

<b>(a) Fusion vs. HARNet Branch</b>
</td>
<td style="text-align:center">
<img src="graph/mosaic_diff_fusion_minus_SSTA.jpg" width="400" alt="Fusion vs SSTA">

<b>(b) Fusion vs. SSTA Branch</b>
</td>
</tr>
</table>
</div>

- **Fig (a)**: H-starNet compensates for HARNet's deficiencies in **lower SNR regions** (dense green points), proving that temporal features are crucial when spectral features are corrupted by noise.

- **Fig (b)**: While SSTA may perform suboptimally at high SNRs, the fused model maintains high accuracy by leveraging HARNet. In **low-SNR conditions**, H-starNet effectively exploits SSTA's robust global correlation features to ensure stability.

**Conclusion**: The fusion mechanism successfully synthesizes the distinct advantages of both architectures, achieving consistent and robust performance across the entire SNR spectrum.


## 🔗 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{liu2025harmony,
  title={Harmony in Time and Frequency: Lightweight Spectro-Temporal Disentanglement for UAV RF Fingerprinting},
  author={Zhenkun Liu, Yunling Liao, Zhiyv Dai and JunSheng Mu},
  journal={arXiv preprint},
  year={2025}
}
```
## License

This repository is under the [Apache-2.0 license](./LICENSE).
