import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LassoCV, RidgeCV

# Load dataset
file_path = 'path_to_your_file/Top100.csv'  # Update this to the correct path
data = pd.read_csv(file_path)

# Rename columns
data.columns = ['Company_Code', 'Revenue', 'Revenue_Growth_Rate', 'Net_Income_Loss', 
                'Net_Worth', 'Return_on_Equity', 'Stock', 'Total_Assets', 'Return_on_Assets', 
                'Asset_Turnover_Ratio', 'Debt_Ratio', 'EPS', 'Top_100']


# Handle missing values using mean imputation for numerical features
numerical_features = ['Revenue', 'Revenue_Growth_Rate', 'Net_Income_Loss', 'Net_Worth', 
                      'Return_on_Equity', 'Stock', 'Total_Assets', 'Return_on_Assets', 
                      'Asset_Turnover_Ratio', 'Debt_Ratio', 'EPS']

imputer = SimpleImputer(strategy='mean')
data[numerical_features] = imputer.fit_transform(data[numerical_features])

# Convert target variable to numeric
data['Top_100'] = data['Top_100'].astype(int)

# Check for data imbalance
target_distribution = data['Top_100'].value_counts()

# Apply SMOTE to handle data imbalance
X = data.drop(['Company_Code', 'Top_100'], axis=1)
y = data['Top_100']

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert resampled data to DataFrame
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled = pd.DataFrame(y_resampled, columns=['Top_100'])

# Standardize numerical features
scaler = StandardScaler()
X_resampled[numerical_features] = scaler.fit_transform(X_resampled[numerical_features])

# Check if there are any categorical features to encode
categorical_features = X_resampled.select_dtypes(include=['object']).columns

# Apply One-Hot Encoding to categorical features if any
if len(categorical_features) > 0:
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_features = encoder.fit_transform(X_resampled[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    X_resampled = X_resampled.drop(categorical_features, axis=1)
    X_resampled = pd.concat([X_resampled, encoded_df], axis=1)

# Lasso (L1 Regularization) for feature selection
lasso = LassoCV(cv=5)
lasso.fit(X_resampled, y_resampled.values.ravel())

# Select features where coefficient is non-zero
lasso_selected_features = X_resampled.columns[lasso.coef_ != 0]

# Ridge (L2 Regularization) for feature selection
ridge = RidgeCV(cv=5)
ridge.fit(X_resampled, y_resampled.values.ravel())

# Select features where coefficient is non-zero
ridge_selected_features = X_resampled.columns[ridge.coef_ != 0]

# Combine selected features from Lasso and Ridge
selected_features = list(set(lasso_selected_features).union(set(ridge_selected_features)))

# Reduce dataset to selected features
X_final = X_resampled[selected_features]

# Display the resulting DataFrames
print(X_final.head())
print(y_resampled.head())
