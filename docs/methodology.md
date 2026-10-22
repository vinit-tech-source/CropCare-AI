# Methodology — CropCare AI

## 1. Overview and Problem Formulation

Automated plant pathology detection requires extracting discriminative patterns that distinguish diseased tissue from healthy foliage under varying field conditions. **CropCare AI** adopts a classical machine learning paradigm, avoiding deep neural networks to ensure computational efficiency, interpretability, and deterministic behavior.

The task is formulated as a two-stage problem:
1. **Multi-Class Disease Classification**: Given a leaf photograph $I \in \mathbb{R}^{H \times W \times 3}$, predict one of 21 crop–disease categories $y \in \mathcal{C}$ and calculate lesion severity percentage $\mathcal{S}$.
2. **Agronomic Fertilizer Recommendation**: Given soil chemistry metrics $(N, P, K)$, soil category, crop type, atmospheric variables (temperature, humidity, moisture), and the detected plant disease $y$, prescribe a targeted fertilizer regimen with specific nutrient adjustments.

---

## 2. Image Preprocessing

Leaf images obtained from varied sources display irregular resolutions and sensor noise. The preprocessing pipeline standardizes all inputs:

```
Raw Leaf Image ──► Resizing (256×256) ──► Gaussian Blur (5×5) ──► Color Space Conversion (Lab & HSV)
```

1. **Spatial Rescaling**: Images are resized to $256 \times 256$ pixels using bilinear interpolation, preserving structural leaf margins while maintaining consistent spatial dimensions for texture operator computations.
2. **Noise Reduction**: A 2D Gaussian filter with kernel size $5 \times 5$ and standard deviation $\sigma = 0$ (derived automatically from kernel size) is applied to suppress high-frequency sensor noise without blurring necrotic lesion boundaries.
3. **Color Space Transformations**:
   - **RGB $\rightarrow$ CIELAB**: Luminance $L^* \in [0, 100]$, chromaticity $a^* \in [-128, 127]$ (green to red), and $b^* \in [-128, 127]$ (blue to yellow).
   - **RGB $\rightarrow$ HSV**: Hue $H \in [0, 180)$, Saturation $S \in [0, 255]$, and Value $V \in [0, 255]$.

---

## 3. KMeans Color-Based Segmentation

Disease segmentation isolates healthy leaf lamina from pathological lesions.

### 3.1 Subspace Clustering
Clustering is executed strictly in the 2D $a^*b^*$ chromaticity subspace of the CIELAB color space. Eliminating the $L^*$ (lightness) channel makes the segmentation invariant to illumination intensity, shadow gradients, and flash reflections.

$$X_{\text{cluster}} = \{(a_i^*, b_i^*) \mid i = 1, \dots, N_{\text{pixels}}\}$$

Using $k=3$ clusters via the Lloyd algorithm:
- **Cluster 0**: Background (white/grey backdrop or non-foliar elements)
- **Cluster 1**: Healthy foliar tissue (characterized by negative $a^*$ and positive $b^*$, indicating green chlorophyll)
- **Cluster 2**: Diseased lesion tissue (characterized by yellow, brown, or red necrosis and chlorosis)

### 3.2 Cluster Identification Heuristic
A botanical color centroid metric identifies the clusters:
- The **healthy cluster** has the maximum Euclidean greenness score in $a^*b^*$ space:
  $$\text{Greenness} = \arg\max_{c} (-a_c^*)$$
- The **diseased cluster** is identified among non-background pixels by computing the Euclidean distance to standard necrotic/chlorotic reference chromaticities:
  $$D_{\text{necrotic}} = \sqrt{(a_c^* - a_{\text{ref}}^*)^2 + (b_c^* - b_{\text{ref}}^*)^2}$$
- **Severity Percentage**:
  $$\text{Severity} = \frac{\sum \text{Mask}_{\text{lesion}}}{\sum \text{Mask}_{\text{leaf}}} \times 100\%$$

---

## 4. Hand-Crafted Feature Vector (106 Dimensions)

The feature extraction module extracts 106 numerical descriptors across four domains:

| Category | Dimensions | Mathematical Methods | Botanical Significance |
|---|---|---|---|
| **Color Features** | 60 | HSV Histograms (8 bins/channel $\times$ 3 channels) + Mean & Std Dev moments for both whole leaf and isolated lesions | Captures chlorophyll breakdown, yellow chlorosis halos, and brown necrotic lesions |
| **Texture — GLCM** | 20 | Gray-Level Co-occurrence Matrix properties across distances $d \in \{1, 3\}$ and angles $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$: Contrast, Homogeneity, Energy, Correlation, Dissimilarity | Quantifies surface roughness, necrotic pustules, and fungal spore texture |
| **Texture — LBP** | 18 | Local Binary Patterns ($P=16$ points, radius $R=2$, uniform pattern histogram with $P+2 = 18$ bins) | Invariant local micro-texture capturing leaf venation and fungal velvety mycelium |
| **Shape Features** | 7 | Hu Invariant Moments with logarithmic sign-preserving transformation: $h_i = -\text{sign}(\mu_i) \cdot \log_{10}(|\mu_i| + 10^{-12})$ | Scale, rotation, and translation invariant descriptors of lesion shape |
| **Severity Feature** | 1 | Percentage of leaf area covered by diseased lesions ($[0.0, 100.0]\%$) | Directly quantifies pathological infection spread |
| **Total** | **106** | Concatenated 1D floating-point vector | Complete numerical representation |

---

## 5. Model Training and Evaluation Protocol

### 5.1 Leakage-Proof Scikit-Learn Pipeline
The complete learning pipeline consists of:
1. **StandardScaler**: Centers each feature to mean 0 and scales to unit variance.
2. **Principal Component Analysis (PCA)**: Retains components explaining 95% of total variance ($n_{\text{components}}=0.95$), reducing feature collinearity and eliminating noise.
3. **Supervised Classifier**: One of four candidate algorithms:
   - **Support Vector Machine (SVM)**: RBF and Linear kernels with class-weighted regularization $C \in [0.1, 100]$ and $\gamma \in [\text{'scale'}, 0.01]$.
   - **Random Forest (RF)**: 100–300 ensemble trees with balanced class weights and tree depth tuning.
   - **Logistic Regression (LR)**: Multinomial formulation with L-BFGS solver and $L_2$ regularization.
   - **K-Nearest Neighbors (KNN)**: $k \in \{3, 5, 7, 9, 11, 13, 15\}$ with uniform and distance weighting.

### 5.2 Cross-Validation Strategy
- **Hyperparameter Optimization**: 5-fold Stratified K-Fold cross-validation using `GridSearchCV`.
- **Final Performance Reporting**: 10-fold Stratified K-Fold cross-validation reporting mean and standard deviation for accuracy, macro-averaged precision, recall, and F1-score.

---

## 6. Disease-Aware Fertilizer Recommendation Logic

The agronomic advisory combines a data-driven model with a rule-based knowledge layer:

### 6.1 Baseline ML Prediction
- A Random Forest classifier trained on 99 agricultural soil and climate records predicts the optimal base chemical fertilizer (`Urea`, `DAP`, `10-26-26`, `20-20`, `28-28`, `17-17-17`, `14-35-14`).
- Preprocessing utilizes a `ColumnTransformer` with `StandardScaler` for continuous metrics ($N, P, K$, temperature, humidity, moisture) and `OneHotEncoder(handle_unknown='ignore')` for categorical attributes (soil type, crop type).

### 6.2 Agronomic Extension Rule Integration
Field research from ICAR, TNAU, and FAO confirms that pathogens interact directly with plant nutrition:
1. **Excess Nitrogen Suppression**: High nitrogen induces thin epidermal cell walls and succulent tissue, accelerating fungal hyphal penetration in blights (*Phytophthora*, *Alternaria*). The rule engine automatically prescribes a **20–30% nitrogen reduction** upon blight detection.
2. **Potassium Fortification**: Potassium enhances lignin synthesis and cell wall thickness, conferring physical resistance to fungal penetration. The rule engine prescribes a **15–25% potassium increase** using Potassium Sulphate ($K_2SO_4$).
3. **Foliar Micronutrients**: Calcium chloride ($CaCl_2$ @ 0.5%) sprays are prescribed for fruit rots to strengthen pectin middle lamella; Zinc Sulphate ($ZnSO_4$ @ 0.5%) is recommended for viral stunting.
4. **Soil Physics Integration**: Leaching-prone sandy soils trigger split-application advice (3–4 splits), whereas heavy black vertisols prompt gypsum drainage recommendations.
