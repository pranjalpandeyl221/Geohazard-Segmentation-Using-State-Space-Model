# Geohazard-Segmentation-Using-State-Space-Model
This repository contains the implementation of Geohazard Segmentation using a State-Space Model-Driven Multiscale Attention Method. The project focuses on accurately segmenting geological hazards from remote sensing imagery by leveraging advanced deep learning techniques inspired by state-space models.
# 🌍 DEM-Aware Landslide Segmentation using SAM + Mamba2D + Topographic Modulation

This repository implements a **DEM-aware landslide segmentation model** that fuses:

- 🧠 **SAM Image Encoder (ViT-L)**
- 🧩 **Multi-scale Local Patch Encoder**
- 🗻 **Topographic-Aware Feature Modulation (TAFM)**
- 🔄 **Hierarchical Gated Decoder with Boundary Refinement**

The model outputs crisp **256×256 binary masks**, supports **mixed-precision training**, **comprehensive metrics**, **automated visualization**, **CSV logging**, and an **optional Mamba 2D refinement** block for enhanced feature representation.

---

## 🚀 Highlights

| Component | Description |
|------------|-------------|
| **DEMFeatureExtractor** | Derives elevation, slope, aspect, and curvature from a single-channel DEM to form a normalized **4-channel topographic tensor**. |
| **SAMFeatureExtractor** | Fuses SAM ViT-L features with Local Patch Encoder outputs, applies **CBAM**, and falls back to a `1×1 Conv2d` if SAM weights are unavailable. |
| **Hierarchical Gated Decoder** | Upsamples via **gated skip connections**, includes **auxiliary heads**, and uses **Boundary Refinement** for sharper edges. |
| **Evaluator** | Reports **Overall Accuracy**, per-class **Precision/Recall/F1/IoU**, **mIoU**, and **FWIoU** via a confusion-matrix pipeline. |
| **Mamba2D (Optional)** | Refines bottleneck features `[B,256,64,64]` using **State Space Models (SSMs)** for improved contextual learning. |

---

## 🧱 Architecture Overview

### Inputs

- **SAM Image:** 1024×1024 RGB (for ViT-L encoder)  
- **Local Image:** 256×256 RGB (for patch encoding)  
- **DEM:** 256×256 single-channel → 4 derived topographic channels  
- **Output:** Binary mask (1×256×256)

---

### 🔁 End-to-End Flow

```text
        +-------------------+           +---------------------+
Image --+--> SAM branch ----+----+      |  DEMFeatureExtractor|
(RGB)   |   (1024x1024)     |    |      |     DEM (256x256)   |
        |                   |    |      +---------------------+
        |                   v    |                   v
        |               SAM Encoder        Topographic-Aware
        |               [256,64,64]  --->  Feature Modulation (TAFM)
        |                        |
        |                        v
        |   +----------------+   |
        |   | LocalPatchEnc  |   |
        |   | (256x256 RGB)  |---+----> Fuse + CBAM
        |   +----------------+        |
        |                             v
        |                        +---------+
        |                        | Mamba2D |  (optional refinement)
        |                        +---------+
        |                             |
        |                      Refined Bottleneck
        |                         [256,64,64]
        |                             |
        |         +-----------------------------------+
        |         |                                   |
        |         v                                   v
        |   Skip1 (pre-M2D)                  Skip2/Skip3 (post-M2D)
        |         \                                   /
        |          \                                 /
        |           \                               /
        |            v                             v
        +--------------------> Hierarchical Gated Decoder
                                  ↓
                             Boundary Refinement
                                  ↓
                             Final Mask (1×256×256)
