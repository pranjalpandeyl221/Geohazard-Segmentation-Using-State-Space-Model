# Geohazard-Segmentation-Using-State-Space-Model
This repository contains the implementation of Geohazard Segmentation using a State-Space Model-Driven Multiscale Attention Method. The project focuses on accurately segmenting geological hazards from remote sensing imagery by leveraging advanced deep learning techniques inspired by state-space models.
Overview
This repository implements a DEM‑aware landslide segmentation model that fuses a SAM image encoder, a multi‑scale Local Patch Encoder, topographic modulation (TAFM), and a hierarchical gated decoder with boundary refinement to output 256×256 binary masks. The pipeline supports mixed‑precision training, comprehensive metrics, automated visualizations, CSV logging, and a pluggable Mamba 2D refinement block at the encoder bottleneck.​

Highlights
DEMFeatureExtractor derives elevation, slope, aspect, and curvature from a single‑channel DEM to form a normalized 4‑channel topographic tensor.​

SAMFeatureExtractor fuses SAM ViT‑L features with Local Patch Encoder outputs and applies CBAM, with a 1×1 Conv2d fallback if SAM weights are unavailable.​

A Hierarchical Gated Feature Pyramid Decoder upsamples with gated skip connections, auxiliary heads, and a Boundary Refinement Block for crisp edges.​

Evaluator reports Overall Accuracy, per‑class Precision/Recall/F1/IoU, mIoU, and FWIoU from a confusion‑matrix pipeline for robust monitoring.​

Architecture
Inputs include a 1024×1024 SAM image, a 256×256 local image, and a 256×256 DEM that is converted to 4 topographic channels and used to modulate fused visual features via TAFM. The fused encoder representation is [B, 256, 64, 64], processed by CBAM and optionally refined by a Mamba 2D block before hierarchical decoding and boundary refinement produce a 1‑channel 256×256 mask.​

Line diagram
The diagram below summarizes the end‑to‑end flow through the encoder, the optional Mamba 2D bottleneck, and the decoder.​

text
        +-------------------+           +---------------------+
Image --+--> SAM branch ----+----+      |  DEMFeatureExtractor|--> DEM 4ch
(RGB)   |   (1024x1024)     |    |      +---------------------+
        |                   |    |                   |
        |                   v    |                   v
        |               SAM Enc. |           Topographic-Aware
        |             [256,64,64]|           Feature Modulation (TAFM)
        |                        |                   |
        |                        |                   v
        |   +----------------+   |          Modulated fused features
        |   | Local PatchEnc |   |                 [256,64,64]
        |   | (256x256) ---> |---+--------------------+
        |   +----------------+        Fuse + CBAM     |
        |                                           + v +
        |                                           |M2D|  (Mamba 2D; optional)
        |                                           + v +
        |                                             |
        |                                      Refined bottleneck
        |                                         [256,64,64]
        |                                             |
        |         +-----------------------------------+
        |         |
        v         v
   Skip1 (pre‑M2D)     Skip2/Skip3 (post‑M2D)
        \               /
         \             /
          \           /
           v         v
          Hierarchical Gated Decoder --> Boundary Refinement --> Mask (1,256,256)
The encoder records Skip1 before Mamba 2D and Skip2/Skip3 after it, ensuring both pre‑ and post‑refinement features are available to the decoder’s gated fusions.​

Mamba 2D refinement
The encoder defines self.mamba_block and applies it immediately after fusion, optional TAFM, and CBAM, providing the correct bottleneck location for a channel‑preserving 2D SSM block on [B, 256, 64, 64]. Replace the Identity with a real module such as Mamba2D(d_model=256, d_state=16, kernel_size=3) and enable use_mamba=True to refine the bottleneck features and the later skip tensors without changing interfaces
