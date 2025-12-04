这是一个非常专业且深入的推导请求。作为论文写作者，我们需要从信号处理的底层逻辑出发，严谨地建立从时域信号 $x_{\mathrm{fcs}}(t)$ 到 时频图 $S(t, f)$，最后到时间平均功率谱 $P_{\text{pooled}}(f)$ 的数学联系。

以下是针对 **FHSS（Frequency Hopping Spread Spectrum）** 信号的详细推导过程。

---

### 1. 信号模型与预处理
**Signal Model:**
$$x_{\mathrm{fcs}}(t)=\sqrt{P_c}\sum_{l=0}^{L-1}p(t-lT_h)\cdot\cos\left(2\pi f_h[l](t-lT_h)+\phi_l\right)$$

为了分析频谱（PSD），我们通常关注信号的**复包络（Complex Envelope）**或**解析信号**，忽略负频率分量（因为实信号的频谱是共轭对称的，分析正频率部分即可）。

令 $\tilde{x}_{\mathrm{fcs}}(t)$ 为 $x_{\mathrm{fcs}}(t)$ 的解析信号近似：
$$ \tilde{x}_{\mathrm{fcs}}(t) \approx \sqrt{P_c} \sum_{l=0}^{L-1} p(t-lT_h) e^{j(2\pi f_h[l](t-lT_h) + \phi_l)} $$

其中：
*   $p(t)$ 是基带脉冲成型滤波器（如高斯滤波器）。
*   $T_h$ 是跳频周期（Hop duration）。
*   $f_h[l]$ 是第 $l$ 跳的载波频率。

---

### 2. 第一步：短时傅里叶变换 (STFT)
**Short-Time Fourier Transform:**
STFT 的定义为：
$$ \text{STFT}(t, f) = \int_{-\infty}^{+\infty} \tilde{x}_{\mathrm{fcs}}(\tau) w(\tau - t) e^{-j2\pi f \tau} d\tau $$
其中 $w(t)$ 是滑动窗函数（Window function）。

对于 FHSS 信号，由于其**时分（Time-Division）**特性，在任意时刻 $t$，只要窗函数 $w(t)$ 的长度小于跳频周期 $T_h$，窗函数通常只覆盖某一个特定的跳 $l$。

假设在时间段 $t \in [lT_h, (l+1)T_h]$ 内，只有第 $l$ 项起作用。我们将变量代换，并利用傅里叶变换的频移性质：

$$ \text{STFT}(t, f) \bigg|_{t \approx lT_h} \propto \sqrt{P_c} \cdot e^{j\phi_l} \cdot P(f - f_h[l]) $$

*   这里 $P(f)$ 是基带脉冲 $p(t)$ 的傅里叶变换。
*   $P(f - f_h[l])$ 表示频谱中心被搬移到了 $f_h[l]$。

**得到时频图 (Spectrogram) $S(t, f)$:**
时频图是 STFT 的模平方：
$$ S(t, f) = |\text{STFT}(t, f)|^2 $$

代入上式，相位项 $e^{j\phi_l}$ 消失：
$$ S(t, f) \approx \begin{cases} P_c \cdot |P(f - f_h[l])|^2 & \text{if } t \in [lT_h, (l+1)T_h] \\ 0 & \text{otherwise (roughly)} \end{cases} $$

**物理意义：**
$S(t, f)$ 在时间轴上是一个分段函数。在第 $l$ 个时间段，能量集中在频率 $f_h[l]$ 处，形状由 $|P(f)|^2$ 决定。

---

### 3. 第二步：时间平均 (Time Pooling)
**Time-Averaged Spectrum:**
我们要计算的是 $P_{\text{pooled}}(f)$，即对整个观测时间 $T_{\text{obs}}$ 进行平均。假设观测时间覆盖了完整的跳频序列，即 $T_{\text{obs}} = L \cdot T_h$。

$$ P_{\text{pooled}}(f) = \frac{1}{T_{\text{obs}}} \int_{0}^{T_{\text{obs}}} S(t, f) \, dt $$

利用 FHSS 信号在时间上的正交性（不同 $l$ 对应不同的时间段），我们可以将积分拆分为求和：

$$ P_{\text{pooled}}(f) = \frac{1}{L \cdot T_h} \sum_{l=0}^{L-1} \int_{lT_h}^{(l+1)T_h} S(t, f) \, dt $$

在每个区间 $[lT_h, (l+1)T_h]$ 内，频谱形状近似不变（稳态），我们可以将积分近似为“功率 $\times$ 持续时间 $T_h$”。或者更严谨地，根据帕塞瓦尔定理，时域积分等于能量。

$$ \int_{lT_h}^{(l+1)T_h} S(t, f) \, dt \approx T_h \cdot \left( P_c \cdot |P(f - f_h[l])|^2 \right) $$
*(注：这里假设 $P(f)$ 已经包含了窗函数的归一化效应)*

将此代入平均公式：

$$ P_{\text{pooled}}(f) \approx \frac{1}{L \cdot T_h} \sum_{l=0}^{L-1} \left[ T_h \cdot P_c \cdot |P(f - f_h[l])|^2 \right] $$

消去 $T_h$：

$$ P_{\text{pooled}}(f) \approx \frac{P_c}{L} \sum_{l=0}^{L-1} |P(f - f_h[l])|^2 $$

---

### 4. 最终公式与指纹分析 (Output for Paper)

基于上述推导，你可以将 FHSS 的结果整理为与 OFDM 形式对称的结论。

**The derived frequency vector for the FHSS signal:**

$$ P_{\text{pooled}}^{\text{FCS}}(f) \approx \frac{P_c}{L} \sum_{l=0}^{L-1} \left| P(f - f_h[l]) \right|^2 $$

**分析 (Analysis):**

这个公式揭示了对 STFT 进行时间平均后，FHSS 信号在频域上的表现形式：它是由 $L$ 个不同中心频率的窄带脉冲频谱叠加而成的，但整体幅度被序列长度 $L$ 衰减（因为能量被分散到了不同的时间片上）。

**无人机指纹参数映射 (Fingerprint Mapping):**

在 $\mathcal{A}_\mathrm{fcs}$ 集合中，通过分析 $P_{\text{pooled}}^{\text{FCS}}(f)$，我们可以提取以下独有参数：

1.  **$\{f_h[l]\}$ (Frequency Hopping Set)**:
    *   **公式体现**：$\sum |P(f - f_h[l])|^2$ 中的峰值位置。
    *   **物理含义**：平均频谱图上会出现一系列离散的峰。这些峰的频率坐标直接对应无人机遥控器设定的跳频表（Hop Map）。不同品牌或型号的无人机，其跳频频点集合是不同的。

2.  **$P(f)$ (Modulation Pulse Shape)**:
    *   **公式体现**：每个峰值的**形状**和**带宽**，即 $|P(\cdot)|^2$。
    *   **物理含义**：这反映了发射机的调制方式（如 GFSK 的调制指数）和脉冲成型滤波器（如高斯滤波的 $BT$ 值）。这是发射机硬件层面的重要指纹。

3.  **$L$ (Sequence Length / Density)**:
    *   **公式体现**：幅度缩放因子 $1/L$ 以及峰值的数量。
    *   **物理含义**：虽然 $L$ 是标量，但在频谱图上，如果我们观测的带宽是固定的，$L$ 越大，平均频谱上的“尖峰”通常越密集（假设频点不重复），且单点平均功率越低。

---

### 论文写作建议 (Writing Tip)

在你的论文中，可以将两个公式并列展示，以此突出**OFDM**和**FHSS**在“时间平均频谱”这一维度上的**对偶性（Duality）**：

*   **OFDM**: 频率分集是**同时**发生的（Simultaneous），因此公式是各项直接叠加（Sum），功率由子载波数量决定。
    $$ \sum |G(f-k\Delta f)|^2 $$
*   **FHSS**: 频率分集是**分时**发生的（Sequential），因此公式是叠加后平均（Sum and Average），功率被跳数 $L$ 稀释。
    $$ \frac{1}{L} \sum |P(f-f_h[l])|^2 $$

这种对比分析能很好地体现你对信号物理层机理的深刻理解。