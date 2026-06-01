# Technical Architecture & Code Specification Document

This document provides a detailed overview of the **Enterprise Diabetes Prediction System** codebase.

---

## 🏗️ System Architecture Overview

```text
diabetes-prediction/
├── requirements.txt         # Project dependency definitions
├── data/
│   └── Pakistani_Diabetes_Dataset.csv # Synthetic clinical dataset
├── src/
│   ├── generate_data.py     # Script to generate synthetic medical features
│   ├── train.py             # Model Training & Preprocessing Pipeline
│   └── dashboard.py         # Streamlit Enterprise Dashboard & Inference Engine
├── models/
│   ├── diabetes_rf_model.pkl # Serialized Random Forest model weights
│   ├── scaler.pkl           # Fitted StandardScaler for input normalization
│   └── model_columns.pkl    # Serialized feature column order to ensure exact matching
```

---

## 📂 File-by-File Technical Specifications

### 1. Data Pipeline (`src/`)

#### 📄 `src/generate_data.py`
*   **Purpose**: Creates a synthetic but statistically realistic dataset of 5,000 patients mirroring real-world clinical distributions.
*   **Mechanics**:
    *   Uses `numpy.random` to sample clinical markers (SysBP, DiaBP, HbA1c, Fasting Blood Sugar) from Gaussian distributions, injecting overlapping noise between Diabetic and Non-Diabetic cohorts to prevent unrealistic model perfection.
    *   Saves the generated DataFrame to `data/Pakistani_Diabetes_Dataset.csv`.

#### 📄 `src/train.py`
*   **Purpose**: Central machine learning pipeline. It handles feature scaling, categorical encoding, model training, and artifact serialization.
*   **Code Mechanics**:
    *   **Categorical Encoding**: Uses `pd.get_dummies` to one-hot encode the `Region` column, preparing categorical strings for mathematical operations.
    *   **StandardScaler**: Normalizes feature magnitudes: $z = \frac{x - \mu}{\sigma}$. This prevents large-scale inputs (like Blood Sugar ~180) from dominating smaller-scale inputs (like HbA1c ~8.0).
    *   **Random Forest**: Fits a robust ensemble of decision trees (`n_estimators=150`, `max_depth=10`), balancing classes automatically.
    *   **Serialization**: Uses `joblib.dump` to export three critical artifacts to the `models/` directory: the trained model, the fitted scaler, and the exact column ordering list.

### 2. Product Deployment Layer (`src/`)

#### 📄 `src/dashboard.py`
*   **Purpose**: The interactive frontend application where clinicians can input live patient vitals.
*   **Code Mechanics**:
    *   **Caching Controls**: Utilizes `@st.cache_resource` to load `.pkl` artifacts into memory exactly once, preventing disk reads on every slider adjustment.
    *   **Live Inference Workflow**:
        1. Captures all user inputs from Streamlit widgets into a dictionary.
        2. Converts the dictionary into a Pandas DataFrame.
        3. One-hot encodes the `Region` column and strictly enforces the exact column order loaded from `models_columns.pkl`. Any missing dummy regions are initialized to 0.
        4. Passes the aligned DataFrame through the pre-loaded `scaler`.
        5. Computes `predict_proba` and visually maps the probability threshold to a clinical warning level (🟢 Low Risk, 🟡 Pre-Diabetic, 🔴 High Clinical Risk).
