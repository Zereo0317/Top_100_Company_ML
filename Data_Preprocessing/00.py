import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the CSV file and remove the first row, use the second row as header
file_path = 'Top100.csv'
data = pd.read_csv(file_path, skiprows=1, header=0)

# Remove rows with more than 30% missing values
threshold = len(data.columns) * 0.30
data_cleaned = data.dropna(thresh=threshold)

# Separate company ID, company name, and the rest of the columns
company_info = data_cleaned.iloc[:, :2]
data_cleaned = data_cleaned.iloc[:, 2:]

# Function to clean numeric values in each column
def clean_column(column):
    return pd.to_numeric(column.astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')

# Apply the cleaning function to each column
data_cleaned = data_cleaned.apply(clean_column)

# Separate numeric and non-numeric columns
numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
non_numeric_cols = data_cleaned.select_dtypes(exclude=[np.number]).columns

# Separate features and target
X_numeric = data_cleaned[numeric_cols]
X_non_numeric = data_cleaned[non_numeric_cols]
y = data_cleaned['Top_100']

# Handle missing values with SimpleImputer for numeric data
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_numeric)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Convert the scaled array back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)

# Save the preprocessed data along with company info and non-numeric data
X_scaled_df['Top_100'] = y.reset_index(drop=True)
preprocessed_data = pd.concat([company_info.reset_index(drop=True), X_scaled_df, X_non_numeric.reset_index(drop=True)], axis=1)
preprocessed_data.to_csv('Top100_preprocessed.csv', index=False)
