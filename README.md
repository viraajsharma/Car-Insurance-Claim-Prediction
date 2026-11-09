# Car Insurance Claim Prediction

An end-to-end machine learning project exploring two problems in insurance analytics: predicting whether a customer will make an insurance claim and estimating the claim amount when a claim occurs.

## Project Overview

Insurance companies need to understand both the likelihood of claims and the potential financial cost associated with them.

This project approaches the problem through two machine learning tasks:

- **Classification:** Predict whether a policyholder will make an insurance claim.
- **Regression:** Predict the claim amount for policyholders who have made a claim.

## Dataset

**Source:** [Car Insurance Claim Data – Kaggle](https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data)

The dataset contains approximately 8,000 policyholder records, with information relating to:

- Customer demographics
- Income and home ownership
- Education and occupation
- Driving history
- Vehicle characteristics
- Previous insurance claims

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- CatBoost
- SciPy
- Statsmodels

## Project Workflow

### 1. Data Cleaning and Exploration

- Data inspection and cleaning
- Handling missing values
- Duplicate checks
- Exploratory Data Analysis
- Correlation analysis
- Variance Inflation Factor (VIF)

### 2. Feature Engineering and Preprocessing

- Custom Scikit-learn transformers
- KNN imputation
- Simple imputation
- Ordinal encoding
- One-Hot Encoding
- Square-root transformation for selected numerical features
- Standardisation using StandardScaler
- ColumnTransformer and Pipeline

### 3. Classification

Compared multiple classification algorithms:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Linear SVM
- XGBoost
- AdaBoost
- Gradient Boosting
- Bagging
- CatBoost

Used cross-validation and F1-based evaluation to compare models, followed by XGBoost hyperparameter optimisation.

### 4. Regression

Filtered the dataset to policyholders with positive claim amounts and evaluated:

- Linear Regression
- SGDRegressor
- Decision Tree Regressor
- Random Forest Regressor
- KNN Regressor
- Support Vector Regression
- XGBoost Regressor

Used RMSE for cross-validation and further hyperparameter optimisation of SGDRegressor.

### 5. Model Evaluation

Evaluated regression predictions using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Actual vs. predicted visualisation
- Residual analysis

## Regression Results

The final recorded SGDRegressor test results were:

| Metric | Result |
|---|---:|
| RMSE | 8,378.85 |
| MAE | 3,556.26 |
| R² | -0.0228 |

The negative R² indicates that the regression model did not outperform a mean-prediction baseline on the test set.

The experiments also revealed that claim amounts were highly right-skewed, with a skewness of approximately 5.47.

This made predicting large claim amounts particularly challenging.

*These are recorded notebook results and should be updated if the final leakage-free evaluation produces different values.*

## Key Learnings

Through this project, I gained practical experience with:

- Building end-to-end machine learning workflows
- Designing custom Scikit-learn transformers
- Using ColumnTransformer and Pipeline
- Handling numerical and categorical data
- Comparing classification and regression algorithms
- Applying K-fold cross-validation
- Performing RandomizedSearchCV and GridSearchCV
- Understanding data leakage and model generalisation
- Evaluating regression errors and skewed target distributions

## Limitations and Future Work

Potential improvements include:

- Improving claim severity prediction
- Exploring additional feature engineering
- Investigating class imbalance
- Experimenting with advanced boosting algorithms
- Applying SHAP and permutation importance
- Improving cross-validation methodology
- Developing a combined two-stage insurance risk prediction system

## Repository Structure

```text
car-insurance-claim-prediction/
│
├── data/
│   └── car_insurance_claim.csv
│
├── car_insurance_claim.ipynb
│
├── README.md
│
└── requirements.txt
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/viraajsharma/Car-Insurance-Claim-Prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook car_insurance_claim.ipynb
```

4. Run the notebook cells in order.

## Author

**Viraaj Sharma**

Interested in Machine Learning, Data Science and AI research.

- GitHub: [@viraajsharma](https://github.com/viraajsharma)
- LinkedIn: [Viraaj Sharma](https://www.linkedin.com/in/viraaj-sharma/)
