import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split

# Load the balanced data
file_path = 'Top100_balanced.csv'
data = pd.read_csv(file_path)

# Separate company info, features, and target
company_info = data.iloc[:, :2]
X_resampled = data.drop(columns=['Top_100', *company_info.columns])
y_resampled = data['Top_100']

# Ensure only numeric columns are used for feature selection
numeric_cols = X_resampled.select_dtypes(include=[np.number]).columns
X_resampled_numeric = X_resampled[numeric_cols]

# Perform feature selection using Lasso and Ridge
X_train, X_test, y_train, y_test = train_test_split(X_resampled_numeric, y_resampled, test_size=0.3, random_state=42)

# Lasso feature selection
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
lasso_selected_features = X_resampled_numeric.columns[lasso.coef_ != 0]
print(f"Lasso selected features: {lasso_selected_features}")

# Ridge feature selection
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_selected_features = X_resampled_numeric.columns[ridge.coef_ != 0]
print(f"Ridge selected features: {ridge_selected_features}")

# Combine selected features
selected_features = list(set(lasso_selected_features).union(set(ridge_selected_features)))
print(f"Selected features: {selected_features}")

# Save the final dataset for modeling
final_data = pd.concat([pd.DataFrame(X_resampled_numeric, columns=X_resampled_numeric.columns)[selected_features], pd.Series(y_resampled, name='Top 100')], axis=1)
final_data = pd.concat([company_info.iloc[y_resampled.index].reset_index(drop=True), final_data], axis=1)
final_data.to_csv('Top100_final.csv', index=False)

print("Final data saved for modeling.")
