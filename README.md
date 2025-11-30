# AI House Price Estimator (Streamlit + CatBoost)

This project is a **house price prediction web app** built with **Streamlit** and a
**CatBoost regression model** trained in Google Colab.

The app lets users configure key house characteristics (location, size, quality, rooms, etc.)
and then predicts an estimated sale price in real-time.

---

## 1. Project Overview

- **Goal**: Predict house sale prices as accurately as possible (high R², low RMSE)  
  while keeping the model stable and avoiding overfitting.
- **Model**: `CatBoostRegressor` trained on **log1p(SalePrice)**  
  with **feature engineering** and **5‑fold cross‑validation**.
- **Front‑end**: Streamlit app with a structured UI:
  - 📍 *Lokasi & Tahun* (Neighborhood, year built, garage)
  - 🧱 *Fisik* (overall quality, living area, lot area)
  - 🛏️ *Ruangan* (bathrooms, bedrooms, building/roof type)
- **Back‑end**: Saved CatBoost model + metadata bundle loaded at runtime.

This repo is intended as a **mini end‑to‑end ML project**: from training in Colab
to a small, production‑style web interface.

---

## 2. Files & Structure

Typical project layout:

```text
.
├── app.py                         # Streamlit app
├── train_100k.csv                 # Training dataset (tabular house data)
├── house_price_catboost_log.cbm   # Trained CatBoost model (log-target)
├── house_price_catboost_meta.pkl  # Metadata: feature names, cat feature indices
└── README.md                      # This file
```

> **Note:** `train_100k.csv` is required by the app to:
> - Infer realistic slider ranges from the training distribution
> - Populate dropdowns for `Neighborhood`, `BuildingType`, and `RoofStyle`

---

## 3. Model Training (Colab)

Model training was done in **Google Colab** with the following setup:

1. **Load data & feature engineering**
   - Base features:
     - Numeric: `LotArea`, `OverallQual`, `YearBuilt`, `GrLivArea`,
       `GarageCars`, `FullBath`, `Bedrooms`
     - Categorical: `Neighborhood`, `BuildingType`, `RoofStyle`
   - Engineered features:
     ```python
     log_LotArea     = np.log1p(LotArea)
     log_GrLivArea   = np.log1p(GrLivArea)
     HouseAge        = max(YearBuilt) - YearBuilt
     Qual_x_GrLiv    = OverallQual * GrLivArea
     Baths_per_Bed   = FullBath / (Bedrooms + 0.5)
     RoomsPlusBaths  = Bedrooms + FullBath
     ```

2. **Target transformation**
   - Train on `log1p(SalePrice)` instead of raw `SalePrice`
   - At prediction time, convert back with `np.expm1(pred_log)`

3. **Cross‑validation (5‑fold KFold)**
   - CatBoost uses the categorical columns directly (via `cat_features` indices)
   - Metrics per fold and averaged on **price scale**:
     - Mean CV **R² ≈ 0.924**
     - Mean CV **RMSE ≈ 20,000**

4. **Final model**
   - Train `CatBoostRegressor` on the full dataset with the same settings:
     - `loss_function="RMSE"`
     - `learning_rate=0.05`
     - `depth=7`
     - `l2_leaf_reg=3.0`
     - `bagging_temperature=1.0`
     - `iterations=2000` with early stopping (`od_type="Iter"`, `od_wait=100`)
   - Save artifacts:
     ```python
     final_model.save_model("house_price_catboost_log.cbm")
     joblib.dump({
         "feature_names": list(X.columns),
         "cat_cols": cat_cols,
         "num_cols": num_cols,
         "cat_feature_indices": cat_feature_indices,
     }, "house_price_catboost_meta.pkl")
     ```

If you want to retrain the model, you can reuse the above logic in a Colab notebook.

---

## 4. Streamlit App

The Streamlit app lives in `app.py`. It does the following:

1. **Load model & metadata**
   ```python
   model = CatBoostRegressor()
   model.load_model("house_price_catboost_log.cbm")
   meta  = joblib.load("house_price_catboost_meta.pkl")
   feature_names        = meta["feature_names"]
   cat_feature_indices  = meta["cat_feature_indices"]
   ```

2. **Load training stats for UI**
   - Reads `train_100k.csv`
   - Computes min/median/quantiles for numeric features
   - Extracts unique values for categorical features to build dropdowns

3. **Feature engineering (must match training)**
   - Recomputes the same engineered features on the user input
   - Reorders columns to `feature_names` before calling `model.predict()`

4. **Real‑time prediction**
   - As the user adjusts sliders/dropdowns, the app:
     - Builds a one‑row `DataFrame` from inputs
     - Applies feature engineering
     - Calls `model.predict()` → `pred_log`
     - Converts to price with `np.expm1(pred_log)`
   - Displays:
     - Estimated price in large text
     - A simple uncertainty band: `± CV_RMSE`

---

## 5. Installation & Usage

### 5.1. Requirements

You can install dependencies via pip:

```bash
pip install streamlit catboost pandas numpy joblib
```

Python 3.9+ is recommended.

### 5.2. Running the Streamlit App

From the project folder (where `app.py` and the model files live):

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

Make sure the following files are present in the same directory:

- `app.py`
- `train_100k.csv`
- `house_price_catboost_log.cbm`
- `house_price_catboost_meta.pkl`

If `train_100k.csv` is missing, the app will stop with an error message,
because it uses that file to populate dropdown options and slider ranges.

---

## 6. Inputs & Features in the UI

The app groups inputs into three sections:

- **📍 Lokasi & Tahun**
  - `Neighborhood` (dropdown from training data)
  - `YearBuilt` (slider)
  - `GarageCars` (dropdown: 0–max in training data)

- **🧱 Fisik**
  - `OverallQual` (slider 1–9)
  - `GrLivArea` (slider; 10–90th percentile of training distribution)
  - `LotArea` (slider; 10–90th percentile of training distribution)

- **🛏️ Ruangan**
  - `FullBath` (slider)
  - `Bedrooms` (slider)
  - `BuildingType` (dropdown)
  - `RoofStyle` (dropdown)

All categorical inputs are **restricted to valid options** from the training set,
so the model can fully use its learned encodings.

---

## 7. Model Performance & Limitations

- The model achieves about **R² ≈ 0.924** on 5‑fold cross‑validation, with
  **RMSE ≈ 20k** on the original price scale.
- There is a small gap between train R² and CV R² (≈ 0.93 vs 0.924),
  which indicates **mild, acceptable overfitting**.
- Performance is limited by the available features:
  - No detailed geographic coordinates
  - No macro‑economic variables
  - No renovation/condition history
- Predictions should be treated as **statistical estimates**, not official appraisals.

---

## 8. Possible Extensions

Some ideas to extend this project:

- Add more engineered features (e.g., price per square meter, interaction terms)
- Train and compare additional models (XGBoost, LightGBM, ensembles)
- Add simple SHAP or permutation feature importance explanations
- Deploy the Streamlit app to a cloud platform (Streamlit Community Cloud, etc.)
- Localize the UI fully in Bahasa Indonesia or support multiple languages

---

## 9. How to Cite / Mention

If you use this project in a report or portfolio, you can describe it as:

> "A house price prediction web app built with Streamlit and a CatBoost regression
> model trained on 100k+ house records with feature engineering and cross‑validation."

