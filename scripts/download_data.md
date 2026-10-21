# How to Download the Datasets

This document explains how to obtain the datasets needed by CropCare AI.

> **Important:** Raw datasets are NOT committed to the repository.
> You must download them yourself and place them in the correct folders.

---

## 1. PlantVillage Leaf Image Dataset

**Source:** Kaggle — [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

### Option A: Kaggle CLI (recommended)

```bash
# Install the Kaggle CLI (if not already)
pip install kaggle

# Set up your Kaggle API token:
#   1. Go to https://www.kaggle.com/settings → API → Create New Token
#   2. Save kaggle.json to:
#        Linux/Mac: ~/.kaggle/kaggle.json
#        Windows:   C:\Users\<USER>\.kaggle\kaggle.json

# Download and extract
kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p data/raw/ --unzip
```

### Option B: Manual download

1. Visit https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
2. Click **Download** (≈ 3 GB zip).
3. Extract into `data/raw/` so the structure looks like:

```
data/raw/
└── New Plant Diseases Dataset(Augmented)/
    ├── train/
    │   ├── Apple___Apple_scab/
    │   ├── Apple___Black_rot/
    │   ├── Apple___Cedar_apple_rust/
    │   ├── Apple___healthy/
    │   ├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
    │   ├── ...
    │   └── Tomato___healthy/
    └── valid/
        ├── Apple___Apple_scab/
        ├── ...
        └── Tomato___healthy/
```

### After downloading

Run the subset creator to extract only the four target crops:

```bash
python scripts/make_subset.py
```

This creates `data/subset/` (train, 300 images/class) and `data/subset_test/` (test, 100 images/class).

---

## 2. Fertilizer Prediction Dataset

**Source:** Kaggle — [Fertilizer Prediction](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction)

### Option A: Kaggle CLI

```bash
kaggle datasets download -d gdabhishek/fertilizer-prediction -p data/raw/ --unzip
```

### Option B: Manual download

1. Visit https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction
2. Download the CSV file.
3. Place it at:

```
data/raw/fertilizer.csv
```

Expected columns: `Temperature`, `Humidity`, `Moisture`, `Soil Type`,
`Crop Type`, `Nitrogen`, `Potassium`, `Phosphorous`, `Fertilizer Name`.

---

## Dataset Licences

| Dataset | Licence | Notes |
|---------|---------|-------|
| PlantVillage (New Plant Diseases Dataset) | CC0 Public Domain | Original: Hughes & Salathé, 2015 |
| Fertilizer Prediction | CC0 Public Domain | Kaggle community dataset |

Both datasets are used under their respective licences for educational/research purposes only.
