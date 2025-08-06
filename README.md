# California_Housing_Price_Analysis_Finished

This repository contains an end-to-end machine learning pipeline to analyze and predict housing prices in California, using structured tabular data. The project includes **data preprocessing**, **feature engineering**, **model training**, **hyperparameter tuning**, **interpretation with SHAP**, then **model evaluation and selection**.

---

## A. Objective
- Understand key drivers of California housing prices.
- Build predictive models using various machine learning techniques.
- Compare model performance using reliable regression metrics.
- Interpret model decisions using SHAP explainability techniques.

---

## B. Dataset
- **Source**: Kaggle `(https://www.kaggle.com/datasets/camnugent/california-housing-prices)`.
- **Records**: 20,640 housing blocks/areas.
- **Target Variable**: `median_house_value` in USD.
- **Key features/variables/columns** include: latitude, longitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, and ocean_proximity.

---

## C. Data Preparation & Feature Engineering
Several key steps were performed to ensure data quality and readiness for modeling:

i. Missing values imputation using `SimpleImputer (median value)` for total_bedrooms variable.

ii. Created a new variable named bedroom_ratio = total_bedrooms / total_rooms as Feature Engineering.

iii. Applied `winsorization` for outlier handling in numerical variables.

iv. Applied `one-hot encoding` on ocean_proximity, which is a categorical variable.

v. Used `RobustScaler` to reduce the impact of outliers as Feature Scaling.

vi. Stratified split with `80% training` and `20% testing` and `random_state=42` for Train-Test Split.

## D. Tools & Libraries Used
- Python.
- pandas, numpy, matplotlib, seaborn.
- scikit-learn.
- Linear Regression, LightGBM, XGBoost, and RandomForest.
- Optuna for hyperparameter tuning.
- SHAP for model interpretability.

## E. Models Trained & Evaluated

- Four `supervised regression` models were built and tuned using `Optuna`:

i. Ridge Linear Regression + Optuna.

ii. Random Forest Regressor + Optuna.

iii. LightGBM Regressor + Optuna.

iv. XGBoost Regressor + Optuna.

- Evaluation metrics used `MAE (Mean Absolute Error)`, `RMSE (Root Mean Squared Error)`, and `R² Score`.

--- 

## F. 
