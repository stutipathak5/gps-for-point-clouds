# GP-PCS: One-Shot Feature-Preserving Point Cloud Simplification with Gaussian Processes on Riemannian Manifolds [ICPR 2024 (Oral)]

[![Published Paper](https://img.shields.io/badge/Published-Paper-blue)](https://doi.org/10.1007/978-3-031-78456-9_28)
[![arXiv](https://img.shields.io/badge/arXiv-2303.15225-b31b1b.svg)](https://arxiv.org/abs/2303.15225)

## 🔍 Overview

 We propose a novel, one-shot point cloud simplification method which preserves both the salient structural features and the overall shape of a point cloud without any prior surface reconstruction step. Our method employs Gaussian processes suitable for functions defined on Riemannian manifolds, allowing us to model the surface variation function across any given point cloud. A simplified version of the original cloud is obtained by sequentially selecting points using a greedy sparsification scheme. The selection criterion used for this scheme ensures that the simplified cloud best represents the surface variation of the original point cloud.

![Teaser](./teaser.png)

## 📝 Citation

Please consider citing the following if you find this work useful:

```bibtex
@inproceedings{pathak2025gp,
  title={GP-PCS: One-shot Feature-Preserving Point Cloud Simplification with Gaussian Processes on Riemannian Manifolds},
  author={Pathak, Stuti and Baldwin-McDonald, Thomas and Sels, Seppe and Penne, Rudi},
  booktitle={International Conference on Pattern Recognition},
  pages={436--452},
  year={2025},
  organization={Springer}
}
```

