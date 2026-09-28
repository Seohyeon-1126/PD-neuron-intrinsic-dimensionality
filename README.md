# PD-neuron-intrinsic-dimensionality

Code for **“Intrinsic dimensionality of deep learning representations reveals cell death-associated heterogeneity in Parkinson’s disease iPSC-derived neurons.”**

## Overview

### Fig_1_Train_Models
Image preprocessing, CNN training, model evaluation, and latent representation extraction.

### Fig_2_SYTOX_Related
SYTOX burden quantification and post-hoc analysis of learned representations.

### Fig_3_Intrinsic_Dimensionality
Representation reproducibility and intrinsic-dimensionality analyses across SYTOX burden.

### Supplementary
Additional analyses of local latent-space geometry across SYTOX burden.

## Analysis

    Fig_1_Train_Models/
    ├── 00_Image_Preprocessing/
    │   ├── 01_maximum_intensity_projection.py
    │   ├── 02_image_cropping.py
    │   └── 03_image_quality_control.py
    ├── 01_ResNet18.py
    ├── 02_SupMoCo.py
    ├── 03_linear_probe_accuracy.py
    └── 04_extract_latent_representation.py

    Fig_2_SYTOX_Related/
    ├── 01_SYTOX_Burden_Calculating/
    │   ├── 01_stardist_nuclei_segmentation.py
    │   ├── 02_crop_sytox_and_nuclei_masks.py
    │   └── 03_calculate_sytox_burden.py
    ├── 02_sytox_prediction_ridge_regression.py
    ├── 03_umap_post_hoc_sytox_embedding.py
    └── 04_phate_post_hoc_sytox_embedding.py

    Fig_3_Intrinsic_Dimensionality/
    ├── 01_linear_cka.py
    ├── 02_pairwise_distance_preservation.py
    ├── 03_sytox_window_mle.py
    ├── 04_sytox_window_twonn.py
    ├── 05_sytox_window_gride.py
    └── 06_sytox_window_shannon_effective_rank.py

    Supplementary/
    ├── 01_local_10nn_radius.py
    ├── 02_median_laplace_beltrami_mean_curvature.py
    └── 03_local_tangent_chordal_distance.py

## Requirements

- Python 3.x
- PyTorch
- NumPy
- pandas
- scikit-learn
- SciPy
- tifffile
- OpenCV
- StarDist

## Data availability

Data availability is described in the associated manuscript.

## Citation

If you use this code, please cite:


## Code archive

The archived version of this repository is available on Zenodo:
10.5281/zenodo.23004622
