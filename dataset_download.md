# 📊 Dataset Download & Information

This repository provides the Nuclei Segmentation dataset in various stages of processing. All files are hosted via GitHub Releases to ensure high-speed downloads and repository stability.

## 📥 Download Links

| Dataset Version | Content Description | Download Link |
| :--- | :--- | :--- |
| **Original Dataset** | The raw stage1_train data from the 2018 Data Science Bowl. | [Download ZIP](https://github.com/ibrar-syed/u_segmentation_nuclei/releases/download/v1.0.0/stage1_train.zip) |
| **Semantic TIF Data** | Unified semantic masks converted from PNG to **.tif** format. | [Download ZIP](https://github.com/ibrar-syed/u_segmentation_nuclei/releases/download/v1.0.0/Semantic_nuclei_data.zip) |
| **Train Split** | Processed training set ready for U-Net input. | [Download ZIP](https://github.com/ibrar-syed/u_segmentation_nuclei/releases/download/v1.0.0/nuclei_train_data.zip) |
| **Validation Split** | Processed validation set for model evaluation. | [Download ZIP](https://github.com/ibrar-syed/u_segmentation_nuclei/releases/download/v1.0.0/nuclei_val_data.zip) |

---

## ⚖️ Licensing & Attribution

### Original Data License
The original dataset is provided under **CC0: Public Domain**. You are free to use, modify, and distribute the data without restriction.

### Derivative Work
This processed version (Semantic TIFs and Train/Val splits) is shared by **Ibrar Syed** under the same **CC0** terms. Proper attribution to the original researchers is required for academic use.

---

## ℹ️ Dataset Details & Processing Info

The following technical details apply to the files provided in this repository:

*   **Image Modalities:** The dataset includes diverse microscopy images (brightfield, fluorescence, etc.) featuring various cell types and nuclei shapes.
*   **Format Conversion:** Original masks consisted of multiple individual instance files. These have been merged into single **Semantic Masks**. To preserve data integrity and prevent compression artifacts common in JPEGs, all masks in the processed set have been converted to **.tif** format.
*   **Split Strategy:** The data has been partitioned into **Train** and **Validation** folders. This fixed split ensures that anyone reproducing this work uses the exact same images for training and testing, allowing for fair performance comparison.
*   **Compatibility:** The folder structure is designed for direct compatibility with standard Deep Learning frameworks (TensorFlow/Keras and PyTorch).

### 📂 Expected Directory Structure
After unzipping the Train and Val files, your local `data/` directory should look like this:
```text
data/
├── train/
│   ├── images/  <-- (.tif source images)
│   └── masks/   <-- (.tif semantic masks)
└── val/
    ├── images/
    └── masks/
