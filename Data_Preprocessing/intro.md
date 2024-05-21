# Data Preprocessing
## Step 1. Null/Missing Value Estimation:

Handle missing values in the dataset by using mean imputation for numerical features as already done.
Ensure no categorical features have missing values. If any, use mode imputation.
Data Imbalance Processing:

Check the distribution of the target variable (Top_100).
If there's an imbalance, apply SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class.

## Step 2. Feature Scaling & Categorical-Numerical Feature Transformation
Feature Scaling:

Apply Standardization (Z-Score Standardization) to numerical features to ensure they have a mean of 0 and a standard deviation of 1.
Categorical-Numerical Feature Transformation:

Convert categorical features (if any) using One-Hot Encoding.
## Step 3. Feature Selection
Filter Methods:

Use correlation matrix to remove highly correlated features that do not add value independently.
Apply statistical tests to rank features and select top features based on their scores.
Embedded Methods:

Use Lasso (L1 Regularization) and Ridge (L2 Regularization) regression to perform feature selection by penalizing less important features.
## Step 4. Dimension Reduction
Principal Component Analysis (PCA):
Apply PCA to reduce dimensionality while retaining most of the variance in the dataset.