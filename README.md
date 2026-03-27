# DeepLense - GSoC 2026 Evaluation Tests

**Applicant:** Ashish Jaimon George  
**Organization:** ML4SCI (Machine Learning for Science)  
**Project:** Foundation Model for Gravitational Lensing  
**Sub-project:** DEEPLENSE  

## Project Overview

This repository contains my solutions for the DeepLense GSoC 2026 evaluation tests. The work demonstrates a progression from standard deep learning classification to physics-informed models and foundation model approaches for gravitational lensing analysis.

## Tests Completed

### Common Test I — Multi-Class Classification
- **Architecture:** ResNet-18 (pretrained, fine-tuned)
- **Task:** Classify lensing images into 3 classes (no substructure, subhalo, vortex)
- **Results:** Macro AUC: **0.9775**
- **Notebook:** `Common_Test_I.ipynb`

### Test VII — Physics-Guided Machine Learning
- **Architecture:** Dual-branch model (ResNet-18 + convergence estimation network)
- **Task:** Same classification task, enhanced with gravitational lensing equation
- **Physics:** Convergence map estimation, Fourier-space deflection angle computation, Poisson equation regularization
- **Results:** Macro AUC: **0.9858** (improvement over Test I baseline on all classes)
- **Notebook:** `Test_VII_Physics_Guided_ML.ipynb`

### Test IX.A — Foundation Model (MAE Pre-training + Classification)
- **Architecture:** Masked Autoencoder (ViT-based) with classification head
- **Task:** Self-supervised pre-training on no_sub class, fine-tuned for 3-class classification (no_sub, cdm, axion)
- **Results:** Macro AUC: **0.9922**
- **Notebook:** `Test_IX_A_Foundation_Model.ipynb`

### Test IX.B — Foundation Model (Super-Resolution Fine-tuning)
- **Architecture:** Pre-trained MAE encoder from IX.A + progressive upsampling decoder
- **Task:** Upscale low-resolution (75×75) lensing images to high-resolution (150×150)
- **Results:** MSE: **0.000093** | SSIM: **0.9658** | PSNR: **40.30 dB**
- **Notebook:** `Test_IX_B_Super_Resolution.ipynb`

## Summary of Results

| Test | Task | Key Metric | Result |
|------|------|-----------|--------|
| Common Test I | Classification | Macro AUC | 0.9775 |
| Test VII | Physics-Guided Classification | Macro AUC | 0.9858 |
| Test IX.A | Foundation Model → Classification | Macro AUC | 0.9922 |
| Test IX.B | Foundation Model → Super-Resolution | PSNR | 40.30 dB |

## Foundation Model Demonstration

Tests IX.A and IX.B together demonstrate the foundation model paradigm:
- A single MAE encoder was pre-trained via self-supervised learning on unlabeled lensing images
- The same encoder was successfully fine-tuned for two fundamentally different downstream tasks: classification and super-resolution
- This validates that self-supervised pre-training learns transferable representations for gravitational lensing analysis

## Repository Structure

```
├── README.md
├── Common_Test_I.ipynb
├── Test_VII_Physics_Guided_ML.ipynb
├── Test_IX_A_Foundation_Model.ipynb
├── Test_IX_B_Super_Resolution.ipynb
├── roc_curves_test1.png
├── roc_curves_test7.png
├── convergence_maps_test7.png
├── mae_reconstruction.png
├── roc_curves_test9a.png
└── sr_comparison_test9b.png
```

Model weights are available via Google Drive (linked in submission).

## Tech Stack

- Python, PyTorch
- Vision Transformers, Masked Autoencoders
- ResNet-18 (transfer learning)
- Physics-Informed Neural Networks

## Contact

- **Email:** [ashishjaimon98@gmail.com]
