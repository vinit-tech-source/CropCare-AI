# System Architecture — CropCare AI

## 1. Executive Summary

**CropCare AI** is an explainable, classical machine learning system for crop disease detection and agronomic advisory. The system combines computer vision feature extraction on leaf imagery with multi-class tabular classifiers and an agricultural extension knowledge base to produce disease-aware fertilizer recommendations.

Unlike deep learning black-box models, CropCare AI relies exclusively on interpretable mathematical features (color distributions, texture co-occurrence matrices, invariant moments, and lesion severity ratios) coupled with scikit-learn statistical pipelines.

---

## 2. End-to-End Pipeline Overview

```mermaid
graph TD
    A["Farmer Leaf Image Input<br/>(RGB, 256×256)"] --> B["Image Preprocessing<br/>(Resize + Gaussian Blur)"]
    B --> C["Color Space Transformations<br/>(RGB → CIELAB & HSV)"]
    C --> D["KMeans Color Segmentation<br/>(Lab a*b* Channels, k=3)"]
    D --> E["Cluster Isolation<br/>(Leaf Mask & Disease Lesions)"]
    E --> F["106D Hand-Crafted Feature Vector"]
    
    subgraph Feature_Extraction ["Feature Extraction (106 Dimensions)"]
        F1["Color: HSV Histograms & Moments (60D)"]
        F2["Texture: GLCM Statistics (20D)"]
        F3["Texture: LBP Uniform Histogram (18D)"]
        F4["Shape: Hu Invariant Moments (7D)"]
        F5["Lesion Area Severity % (1D)"]
    end
    
    F --> F1 & F2 & F3 & F4 & F5
    F1 & F2 & F3 & F4 & F5 --> G["StandardScaler Normalization"]
    G --> H["PCA Dimensionality Reduction<br/>(95% Explained Variance)"]
    H --> I["Disease Classifier<br/>(SVM / Random Forest / KNN / LR)"]
    
    I --> J["Disease Diagnosis & Confidence"]
    
    subgraph Fertilizer_Engine ["Agronomic Recommendation Engine"]
        K["Soil & Climate Parameters<br/>(N, P, K, Soil, Temp, Humidity)"] --> L["ML Fertilizer Pipeline<br/>(ColumnTransformer + RF)"]
        L --> M["Base Chemical Fertilizer"]
        J --> N["Rule-Based Knowledge Layer<br/>(ICAR / TNAU / FAO Rules)"]
        M --> N
        N --> O["Disease-Specific Adjustments<br/>(N Reduction, K Fortification, Micronutrients)"]
    end
    
    O --> P["Rich Terminal Interface & Visual Overlays"]
```

---

## 3. High-Level Component Architecture

```mermaid
graph LR
    subgraph CLI_Interface ["User & CLI Interface"]
        CLI["src/cli.py<br/>(argparse)"]
        RICH["Rich Terminal UI<br/>(Tables, Panels)"]
    end

    subgraph Core_Pipelines ["Core Machine Learning Modules"]
        PRE["src/preprocessing.py"]
        SEG["src/segmentation.py"]
        FEAT["src/features.py"]
        TRAIN_D["src/train_disease.py"]
        TRAIN_F["src/train_fertilizer.py"]
        EVAL["src/evaluate.py"]
        REC["src/recommend.py"]
        PRED["src/predict.py"]
    end

    subgraph Knowledge_Data ["Knowledge Base & Storage"]
        CFG["configs/config.yaml"]
        KB1["fertilizer_rules.json"]
        KB2["diseases.json"]
        KB3["disease_fertilizer_prescriptions.json"]
        MODELS["models/*.joblib"]
        REPORTS["reports/figures & CSVs"]
    end

    CLI --> PRED & TRAIN_D & TRAIN_F & EVAL
    PRED --> PRE --> SEG --> FEAT --> TRAIN_D
    PRED --> REC --> TRAIN_F
    REC --> KB1 & KB2 & KB3
    PRED --> RICH
    CFG --> PRE & SEG & FEAT & TRAIN_D & TRAIN_F
```

---

## 4. Key Architectural Decisions

### 4.1 Strict Classical Machine Learning (No Neural Networks)
- **Rationale**: Real-world agricultural deployment in rural edge settings requires ultra-low compute overhead, rapid inference (<0.5 seconds on a single CPU core), and complete transparency for agricultural extension officers.
- **Explainability**: Every feature in the 106-dimensional vector corresponds to a physical botanical characteristic (e.g., chlorosis indicated by yellow/brown HSV bins, tissue necrosis captured by GLCM dissimilarity, and systemic spread captured by lesion severity ratio).

### 4.2 CIELAB Color Space for KMeans Segmentation
- **Decoupling Luminance from Chromance**: The $L^*$ channel represents perceptual lightness, while $a^*$ (green–red) and $b^*$ (blue–yellow) capture chromaticity. By clustering strictly in the $a^*b^*$ subspace, the segmentation algorithm remains robust against uneven sunlight, shadows, and camera exposure variances.

### 4.3 Leakage-Proof Scikit-Learn Pipeline
- To avoid data leakage, preprocessing (`StandardScaler`), dimensionality reduction (`PCA(n_components=0.95)`), and classifiers (`SVC`, `RandomForestClassifier`, `LogisticRegression`, `KNeighborsClassifier`) are encapsulated inside a single `sklearn.pipeline.Pipeline`.
- Normalization statistics and PCA projection matrices are fitted strictly on the training folds during cross-validation.

### 4.4 Two-Tier Fertilizer Advisory System
1. **Tier 1 (ML Classification)**: A Random Forest classifier with `ColumnTransformer` (StandardScaler for continuous soil parameters + OneHotEncoder for soil and crop types) predicts the baseline synthetic or complex fertilizer (e.g., Urea, DAP, 10-26-26, 20-20).
2. **Tier 2 (Agronomic Rule Engine)**: The baseline recommendation is dynamically adjusted using rule tables curated from ICAR (Indian Council of Agricultural Research) and agricultural universities. For example:
   - Fungal blights require a 25–30% reduction in nitrogen to halt succulent tissue proliferation, accompanied by a 20% increase in potassium to strengthen epidermal cell walls.
   - Calcium chloride and zinc sulphate foliar sprays are added based on the specific pathogen pathology.

---

## 5. Storage and Serialization

| Artifact Type | Path | Purpose | Format |
|---|---|---|---|
| **Disease Model** | `models/random_forest_model.joblib` | Serialized PCA + Classifier + Encoder | Joblib binary |
| **Fertilizer Model** | `models/fertilizer_model.joblib` | Serialized ColumnTransformer + Pipeline | Joblib binary |
| **Knowledge Base** | `data/knowledge_base/*.json` | Disease prescriptions, symptoms, dosages | UTF-8 JSON |
| **Configuration** | `configs/config.yaml` | Hyperparameters, paths, random seed (42) | YAML |
| **Evaluation Metrics**| `reports/model_comparison.csv` | Cross-validation scores, accuracy, F1 | CSV |
| **Visualizations** | `reports/figures/*.png` | Confusion matrices, ROC curves, elbow plot | PNG |
| **Overlay Output** | `outputs/overlays/*_segmented.png` | Segmented disease lesion mask overlays | PNG |
