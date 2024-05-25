# Data Preprocessing (done by Yu-Wen)
### General idea for the preprocessing: 
1. Load and Clean Data
2. Impute Missing Values
3. Check Data Imbalance
4. Balance Data Using SMOTE+ENN
5. Feature Selection Using Lasso and Ridge
6. Save the Final Dataset

**Python code are divided into four steps: 00.py, 01.oy, 02.py, 03.py**


### Challenge:
The biggest challange is apply neural network in missing value estimation, because tensorflow somehow dose not work on apple sillicon. That's also why the code is divided. Additionally, I am not familiar with how neural network work, so it took me some time to understand the mechanism.

## Workflow
### 00_Preprocessing
```python
#see 00.py
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
```
#### Key strategy
1. Remove rows with more than 30% missing values
2. Substitute missing values with mean first
3. Standardize the features 

### 01_TensorFlow Imputation
```python
#see 01.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the preprocessed data
file_path = 'Top100_preprocessed.csv'
data = pd.read_csv(file_path)

# Separate company info, features, and target
company_info = data.iloc[:, :2]
X_scaled = data.drop(columns=['Top_100', *company_info.columns])
y = data['Top_100']

# Build the neural network
model = Sequential()
model.add(Dense(64, input_dim=X_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(X_scaled.shape[1], activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network
model.fit(X_scaled, X_scaled, epochs=100, batch_size=10, verbose=0)

# Predict missing values
X_pred = model.predict(X_scaled)

# Replace missing values in the original dataset
X_imputed = np.where(np.isnan(X_scaled), X_pred, X_scaled)

# Save the imputed data along with company info
imputed_data = pd.DataFrame(X_imputed, columns=X_scaled.columns)
imputed_data['Top_100'] = y
imputed_data = pd.concat([company_info.reset_index(drop=True), imputed_data], axis=1)
imputed_data.to_csv('Top100_imputed.csv', index=False)
```

Estimate Missing Values Using a **_Neural Network_**
##### How we Handling Missing Values from 00.py&01.py:
1. **SimpleImputer:** Initially, we use the mean strategy to fill in missing values.
2. **StandardScaler:** The features are scaled to have zero mean and unit variance.
3. **Neural Network:** A neural network with two hidden layers (64 and 32 neurons) and an output layer with the same number of neurons as the input features is built to learn the pattern of the data and predict missing values.
4. **Training:** The network is trained using the non-missing data.
5. **Redicting Missing Values:** The trained model predicts missing values which are then used to fill in the original dataset.

### 02_Data Balance: SMOTE+ENN
```python
#see 02.py
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

```
Data imbalance before SMOTEENN:

|Top_100|No.|
|---|---|---|
|1.0|61|
|0.0|890|


Data balance after SMOTEENN:

|Top_100|No.|
|---|---|---|
1.0|868|
0.0|825|

### 03_Feature Selection
```python
#see 03.py
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

```
I used Lasso and Ridge for feature selection. Below are the result from two algorithm respectively: 

* **Lasso selected features:** 
Revenue, Income, Networth, Return_Eq, Stock, Total_assete, Reture_A, Turnover, Liab_ratio, EPS
* **Ridge selected features:** 
Revenue, SV, Income, Networth, Return_Eq, Stock, Total_assete, Reture_A, Turnover, Liab_ratio, EPS
* **Final selected features:**
['Income', 'Liab_ratio', 'Total_assete', 'EPS', 'Networth', 'Return_Eq', ' Revenue ', 'Turnover', 'Stock', 'Reture_A', 'SV']
      
### Check "Top100_final.csv" for modeling.
