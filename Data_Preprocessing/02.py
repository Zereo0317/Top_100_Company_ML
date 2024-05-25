import pandas as pd
from imblearn.combine import SMOTEENN

# Load the imputed data
file_path = 'Top100_imputed.csv'
data = pd.read_csv(file_path)

# Separate company info, features, and target
company_info = data.iloc[:, :2]
X_imputed = data.drop(columns=['Top_100', *company_info.columns])
y = data['Top_100']

# Check data imbalance
print("Data imbalance before SMOTEENN:")
print(y.value_counts())

# Use SMOTE+ENN to balance the dataset
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_imputed, y)

# Check data imbalance after SMOTEENN
print("Data balance after SMOTEENN:")
print(pd.Series(y_resampled).value_counts())

# Create a new DataFrame for the balanced data
balanced_data = pd.DataFrame(X_resampled, columns=X_imputed.columns)
balanced_data['Top_100'] = y_resampled

# Since the indices are shuffled, we need to create new indices for company info
# Replicate company info to match the new data length
company_info_resampled = company_info.sample(n=len(balanced_data), replace=True, random_state=42).reset_index(drop=True)

# Combine the balanced data with company info
balanced_data = pd.concat([company_info_resampled, balanced_data], axis=1)
balanced_data.to_csv('Top100_balanced.csv', index=False)
