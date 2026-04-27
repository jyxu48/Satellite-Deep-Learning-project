# Deep Learning and Remote Sensing

This repository collects course projects and experiments at the intersection of **deep learning**, **remote sensing**, and **geospatial analysis**. It currently contains a full final project on land cover segmentation and elephant habitat suitability modeling in Thailand, as well as a smaller EuroSAT classification project.

## Repository Structure

```text
.
├── final_project.ipynb
├── final_project_files/
├── FINAL_PROJECT.md
├── FINAL_PROJECT_PROPOSAL.md
├── MIDTERM.md
└── eurosat project/
    ├── eurosat.ipynb
    └── EUROSAT.md
```

## Final Project: Thailand EFCOM Region

The main project in this repository is a two-part workflow built for **MUSA 650**.

### Part I. Land Cover Segmentation

This section uses **multi-temporal Sentinel-2 imagery** and **JAXA 10 m Southeast Asia land cover labels** to build a 5-class land cover segmentation pipeline for Thailand's EFCOM region.

The final target classes are:
- Forest
- Plantation
- Cropland
- Human-use
- Water/Wetland

Two modeling approaches were compared:
- **Random Forest baseline** using pixel-level spectral features
- **U-Net semantic segmentation models** using 24-channel bi-monthly Sentinel-2 composites

Held-out validation results show that the optimized **U-Net v2** outperformed both the Random Forest baseline and the initial U-Net run:

| Model | Accuracy | Macro F1 | mIoU |
|---|---:|---:|---:|
| Random Forest baseline | 88.9% | 76.2% | 63.2% |
| U-Net v1 | 81.1% | 69.4% | 55.3% |
| U-Net v2 | 90.2% | 82.8% | 72.7% |

### Part II. Elephant Habitat Suitability Modeling

The segmentation outputs are then transformed into landscape composition predictors for habitat modeling.

Using:
- land-cover proportions within a neighborhood around each sample point,
- NDVI,
- elephant occurrence points,
- and randomly generated background points,

a **Random Forest habitat suitability model** was trained to estimate elephant presence probability across the study area.

The final habitat suitability model achieved:
- **AUC = 0.947**

This project demonstrates how deep learning-based land cover mapping can be used not only for classification, but also as a structured ecological input for downstream species distribution and conservation analysis.

## Key Files

### Main Notebook
- [`final_project.ipynb`](./final_project.ipynb): full project notebook containing methodology, code, figures, evaluation, and discussion.

### Companion Files
- [`final_project_files/`](./final_project_files/): supporting figures, tables, scripts, configs, and references used by the notebook.

This folder includes:
- `figures/`: validation figures, patch comparisons, study-region maps, training curves
- `tables/`: saved metrics and model summaries
- `scripts/`: preprocessing, training, evaluation, and prediction scripts
- `configs/`: class remapping configuration
- `refs/`: bibliography template

### Project Documents
- [`FINAL_PROJECT.md`](./FINAL_PROJECT.md): final project write-up
- [`FINAL_PROJECT_PROPOSAL.md`](./FINAL_PROJECT_PROPOSAL.md): proposal document
- [`MIDTERM.md`](./MIDTERM.md): midterm milestone document

## EuroSAT Project

This repository also includes a smaller project based on the **EuroSAT** dataset:

- [`eurosat project/eurosat.ipynb`](./eurosat%20project/eurosat.ipynb)
- [`eurosat project/EUROSAT.md`](./eurosat%20project/EUROSAT.md)

This section contains an additional remote sensing deep learning exercise and is kept separately from the main final project workflow.

## Methods Summary

Across the repository, the main remote sensing and machine learning techniques include:
- multi-temporal Sentinel-2 compositing
- raster label remapping and alignment
- patch-based semantic segmentation
- Random Forest classification
- U-Net-based deep learning segmentation
- spatial feature engineering for habitat modeling
- geospatial visualization and ecological interpretation

## Dependencies

The final project notebook uses Python tools commonly used in geospatial machine learning workflows, including:
- `numpy`
- `pandas`
- `matplotlib`
- `rasterio`
- `geopandas`
- `shapely`
- `scikit-learn`
- `torch`
- `segmentation-models-pytorch`
- `scipy`

If this repository is reused in a clean environment, the dependency list may need to be expanded beyond the minimal `requirements.txt` currently included in the original working directory.

## Authors

- Jinyang Xu
- Qianmu Zheng
- Xiao Yu

## Acknowledgments

This repository was developed for coursework in **deep learning and remote sensing**, with a focus on practical geospatial machine learning, semantic segmentation, and ecological modeling.
