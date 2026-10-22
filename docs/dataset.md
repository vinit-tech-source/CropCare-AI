# Dataset Documentation — CropCare AI

## 1. Overview of Datasets

CropCare AI uses two complementary datasets:
1. **Plant Pathology Imagery**: Subset of the *PlantVillage* (New Plant Diseases Dataset) for vision-based disease classification.
2. **Soil and Climate Tabular Dataset**: Kaggle *Fertilizer Prediction Dataset* for baseline chemical fertilizer recommendation.

---

## 2. PlantVillage Leaf Image Dataset

### 2.1 Scope and Target Crops
The dataset is curated for four economically critical crops across 21 pathological classes:

| Crop | Class Identifier | Scientific / Common Name | Pathogen / Causative Agent |
|---|---|---|---|
| **Apple** | `Apple___Apple_scab` | Apple Scab | *Venturia inaequalis* (Fungus) |
| **Apple** | `Apple___Black_rot` | Black Rot (Frogeye Leaf Spot) | *Botryosphaeria obtusa* (Fungus) |
| **Apple** | `Apple___Cedar_apple_rust` | Cedar Apple Rust | *Gymnosporangium juniperi-virginianae* (Fungus) |
| **Apple** | `Apple___healthy` | Healthy Apple Foliage | None (Healthy) |
| **Corn** | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Gray Leaf Spot | *Cercospora zeae-maydis* (Fungus) |
| **Corn** | `Corn_(maize)___Common_rust_` | Common Rust | *Puccinia sorghi* (Fungus) |
| **Corn** | `Corn_(maize)___Northern_Leaf_Blight` | Northern Leaf Blight | *Exserohilum turcicum* (Fungus) |
| **Corn** | `Corn_(maize)___healthy` | Healthy Corn Foliage | None (Healthy) |
| **Potato** | `Potato___Early_blight` | Early Blight | *Alternaria solani* (Fungus) |
| **Potato** | `Potato___Late_blight` | Late Blight | *Phytophthora infestans* (Oomycete) |
| **Potato** | `Potato___healthy` | Healthy Potato Foliage | None (Healthy) |
| **Tomato** | `Tomato___Bacterial_spot` | Bacterial Spot | *Xanthomonas vesicatoria* (Bacterium) |
| **Tomato** | `Tomato___Early_blight` | Early Blight | *Alternaria solani* (Fungus) |
| **Tomato** | `Tomato___Late_blight` | Late Blight | *Phytophthora infestans* (Oomycete) |
| **Tomato** | `Tomato___Leaf_Mold` | Leaf Mold | *Passalora fulva* (Fungus) |
| **Tomato** | `Tomato___Septoria_leaf_spot` | Septoria Leaf Spot | *Septoria lycopersici* (Fungus) |
| **Tomato** | `Tomato___Spider_mites Two-spotted_spider_mite` | Two-Spotted Spider Mite | *Tetranychus urticae* (Arachnid Pest) |
| **Tomato** | `Tomato___Target_Spot` | Target Spot | *Corynespora cassiicola* (Fungus) |
| **Tomato** | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato Yellow Leaf Curl Virus (TYLCV) | Begomovirus / Geminiviridae (Virus) |
| **Tomato** | `Tomato___Tomato_mosaic_virus` | Tomato Mosaic Virus (ToMV) | Tobamovirus (Virus) |
| **Tomato** | `Tomato___healthy` | Healthy Tomato Foliage | None (Healthy) |

### 2.2 Dataset Partitioning
The dataset is structured with strict class balance to prevent majority-class bias:
- **Training Subset (`data/subset/`)**: 300 images per class $\times 21$ classes = **6,300 images**.
- **Test Subset (`data/subset_test/`)**: 100 images per class $\times 21$ classes = **2,100 images**.
- **Total Subset Size**: **8,400 images**.
- All partitions and metadata are tracked deterministically in `data/processed/split_manifest.json` with SHA-256 hashes.

---

## 3. Soil and Fertilizer Tabular Dataset

Located at `data/raw/fertilizer.csv`, this dataset maps physical soil chemistry and environmental variables to commercial fertilizer formulations.

### 3.1 Feature Schema
- **Numerical Features (6)**:
  - `Temparature`: Ambient temperature in degrees Celsius ($^{\circ}\text{C}$). Range: $25 - 38^{\circ}\text{C}$.
  - `Humidity`: Relative atmospheric humidity percentage ($\% \text{RH}$). Range: $50 - 72\%$.
  - `Moisture`: Volumetric soil moisture percentage ($\% \text{VMC}$). Range: $25 - 65\%$.
  - `Nitrogen`: Available soil Nitrogen ($N$) in $\text{kg/ha}$. Range: $4 - 42 \text{ kg/ha}$.
  - `Phosphorous`: Available soil Phosphorus ($P$) in $\text{kg/ha}$. Range: $0 - 42 \text{ kg/ha}$.
  - `Potassium`: Available soil Potassium ($K$) in $\text{kg/ha}$. Range: $0 - 19 \text{ kg/ha}$.
- **Categorical Features (2)**:
  - `Soil Type`: 5 classes (*Sandy*, *Loamy*, *Black*, *Red*, *Clayey*).
  - `Crop Type`: 11 agricultural crops (*Maize*, *Sugarcane*, *Cotton*, *Tobacco*, *Paddy*, *Barley*, *Wheat*, *Millets*, *Oil seeds*, *Pulses*, *Ground Nuts*).
- **Target Variable**:
  - `Fertilizer Name`: 7 discrete commercial fertilizer formulations (*Urea*, *DAP*, *14-35-14*, *28-28*, *17-17-17*, *20-20*, *10-26-26*).
- **Dataset Size**: 99 validated agricultural extension records.
