# CMP719 Computer Vision Project Proposal

**Student:** Erkani Mert Tosun  
**Student ID:** N25123091

## Problem Definition

Designing neural networks that are both fast and accurate is a core challenge in
computer vision. The dominant approach has been to reduce the number of
floating-point operations (FLOPs) by substituting standard convolutions with
depthwise convolutions (DWConv), as seen in MobileNets and ShuffleNets.
However, reducing FLOPs does not automatically reduce latency. The key insight
missing from prior work is that latency is governed not only by FLOPs but by
floating-point operations per second (FLOPS), the effective computational
throughput. DWConv, despite its low FLOPs, causes excessive memory access due to
the need to widen the network to recover accuracy, which severely suppresses
FLOPS and results in unexpectedly high latency.

## Selected Paper

Jierun Chen, Shiu-hong Kao, Hao He, Weipeng Zhuo, Song Wen, Chul-Ho Lee, and
S.-H. Gary Chan. **Run, Don't Walk: Chasing Higher FLOPS for Faster Neural
Networks.** In *Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR)*, 2023. arXiv:2303.03667.

## Proposed Extension

While PConv efficiently extracts spatial features from a subset of channels, it
treats all `c_p` output channels uniformly. Each channel contributes equally to
the subsequent pointwise convolution (PWConv) regardless of how informative its
spatial features are. This uniform treatment is a missed opportunity: after
spatial convolution, some channels inevitably encode more discriminative
patterns than others, yet the network has no explicit mechanism to amplify or
suppress them at this stage.

We propose adding a lightweight Channel Gate Module (CGM) immediately after
PConv within each FasterNet block, modifying the block structure.

The CGM operates only on the `c_p` channels produced by PConv and consists of
three steps:

1. Global average pooling to produce a channel-wise descriptor vector of
   dimension `c_p`.
2. A two-layer fully connected bottleneck (`c_p -> c_p/r -> c_p`, with reduction
   ratio `r`, e.g. `r=4`) with ReLU activation in between.
3. A sigmoid function to produce per-channel scale factors in `[0, 1]`.

The PConv output is then rescaled channel-wise by these factors before being
passed to `PWConv_1`. Importantly, the CGM does not interfere with PConv's
memory access pattern, so the contiguous-access advantage that gives PConv its
high FLOPS is fully preserved.

## Planned Analysis

### Stage-Wise Analysis

We will evaluate variants where CGM is applied selectively, only to early stages
(1-2) or only to late stages (3-4), to determine at which depth channel
recalibration provides the most benefit.

### Overhead Analysis

We will precisely measure the FLOPs and inference latency added by CGM across
all FasterNet variants (T0, T1), verifying quantitatively that the overhead is
negligible and that the FLOPS advantage of PConv is preserved.

### Gate Visualization

We will visualize the learned gate weights produced by CGM across different
layers and input images, providing interpretable evidence of which channels are
suppressed or amplified and whether this correlates with the spatial content of
the feature maps.

## Experimental Plan

We will train FasterNet-T0 and FasterNet-T1 with and without CGM on CIFAR-100
and Tiny-ImageNet, reporting:

- top-1 accuracy,
- parameter count,
- FLOPs,
- inference latency.
