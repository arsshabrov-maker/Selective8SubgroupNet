# Selective8SubgroupNet: Activation-Free Structural Nonlinearity via Amplitude-Conditioned Stopping Subgroups

**Author:** Arsenii Shabrov  
**Date:** November 2026  

## Abstract
Standard deep neural networks collapse into trivial linear transformations if pointwise nonlinear activation functions (such as ReLU or GELU) are removed. We introduce **Selective8SubgroupNet**, a PyTorch architecture demonstrating that **structural routing alone generates full universal nonlinearity without any pointwise activation function**. Each hidden layer partitions units into **8 specialized subgroups** combining two complementary width-depth energy-conservation polarities (Group A: large activations expand in width, small travel deep; Group B: small activations expand in width, large shoot deep) with four value-comparison stopping rules as signals traverse downstream layers: **approximate equality (`|h_src - h_dst| ~ 0`)**, **less-than (`h_src < h_dst`)**, **greater-than (`h_src > h_dst`)**, and **unconditional pass-through (`∞`)**. Across 4 benchmarks (averaged over 3 random seeds), **Activation-Free `Selective8SubgroupNet`** achieves **99.56% accuracy** on nonlinear *Nested Circles* (where an activation-free MLP fails at `62.52%`) and **95.93%** on *8×8 Handwritten Digits*—outperforming a standard `MLP + GELU` (`99.41%` and `95.85%`) with fewer parameters (`4,596` vs. `4,839`). When paired with GELU, `Selective8SubgroupNet` ranks **#1 overall (`98.19%`)**.

## 1. Eight-Subgroup Selective Stopping Mechanism
Under edge-budget conservation ($\sqrt{\alpha}$ to width, $\sqrt{1 - \alpha}$ to depth), each neuron $i$ belongs to one of 8 subgroups determined by polarity $p_i \in \{+1, -1\}$ and stopping rule $r_i \in \{\approx, <, >, \infty\}$. As the depth signal $s_i$ reaches layer $k > l$ with target activation $h_k$, a differentiable stopping gate $g_{i,k} \in [0, 1]$ deposits $g_{i,k} w_{l,k,i} s_i$ into neuron $k$ while attenuating the continuing beam by $(1 - 0.5 g_{i,k})$. Because $g_{i,k}$ depends dynamically on the comparison between $h_l$ and $h_k$, deep compositions remain strictly nonlinear even when all pointwise activation functions are identity maps ($\sigma(z) = z$).

## 2. Experimental Verification (4 Benchmarks × 3 Seeds)
* **Nested Circles (Nonlinear):** Activation-Free MLP (`62.52%`) vs. **Activation-Free 8-Subgroup Net (`99.56%`)** vs. Standard MLP + GELU (`99.41%`) vs. **8-Subgroup Net + GELU (`99.70%`)**.
* **Digits 8×8 (Real Vision):** Activation-Free MLP (`94.59%`) vs. **Activation-Free 8-Subgroup Net (`95.93%`)** vs. Standard MLP + GELU (`95.85%`) vs. **8-Subgroup Net + GELU (`96.37%`)**.
* **Overall 4-Task Mean (3 Seeds):** **8-Subgroup Net + GELU (`98.19%`, `4,596` params)** > Standard MLP + GELU (`98.17%`, `4,839` params) > **Activation-Free 8-Subgroup Net (`98.04%`, `4,596` params)** > Activation-Free MLP (`88.63%`, `4,839` params).
