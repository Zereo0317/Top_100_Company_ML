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
