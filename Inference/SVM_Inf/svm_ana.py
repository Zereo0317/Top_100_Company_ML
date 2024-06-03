import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 讀取資料
data = pd.read_csv('dataset/Top100_final.csv')
wanna_know_data = pd.read_csv('dataset/test_cody.csv')

# 移除不必要的欄位
data = data.drop(columns=['N_name', 'Name'])
wanna_know_data = wanna_know_data.drop(columns=['N_name', 'Name'])




# 分離特徵和目標
X = data.drop(columns='Top 100')
wanna_know_data_X = wanna_know_data.drop(columns='Top_100')
y = data['Top 100']

# 分割訓練和測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
wanna_know_data_X_scaled = scaler.transform(wanna_know_data_X)

# 訓練SVM模型
svm_classifier = SVC(kernel='rbf', C=7, gamma=0.3, probability=True)
svm_classifier.fit(X_train_scaled, y_train)

# 印出原始資料
print("調整前的原始資料:")
print(pd.DataFrame(wanna_know_data_X_scaled, columns=wanna_know_data_X.columns))

# 計算預測集中每個樣本的決策函數值
decision_values = svm_classifier.decision_function(wanna_know_data_X_scaled)

# 找出需要調整的樣本（假設我們想要將所有不是Top_100的樣本變成Top_100）
samples_to_adjust = wanna_know_data_X_scaled[decision_values < 0]

# 計算每個特徵的梯度
gradients = np.dot(svm_classifier.dual_coef_, svm_classifier.support_vectors_).flatten()

# 縮放調整量 大略的極限是0.004
###############
scale_shrink = 0.004
###############
# 計算調整量
adjustments = scale_shrink * gradients * -decision_values[decision_values < 0].reshape(-1, 1)

# 印出調整量
print("\n調整量:")
print(pd.DataFrame(adjustments, columns=wanna_know_data_X.columns))

# 計算調整後的特徵值
adjusted_samples = samples_to_adjust + adjustments

# 印出調整後的特徵值
print("\n調整後的特徵值:")
print(pd.DataFrame(adjusted_samples, columns=wanna_know_data_X.columns))

# 對調整後的樣本進行預測
adjusted_samples_predictions = svm_classifier.predict(adjusted_samples)

# 印出調整後的預測結果
print("\n調整後的預測結果:")
print(adjusted_samples_predictions)

# 印出原始的預測結果
print("\n原始的預測結果:")
print(svm_classifier.predict(wanna_know_data_X_scaled))

# 計算每個特徵的改變量
feature_changes = adjusted_samples - scaler.inverse_transform(samples_to_adjust)

# 取得特徵名稱
features = list(wanna_know_data_X.columns)

# 繪製長條圖
plt.figure(figsize=(10, 5))

# 繪製第一筆資料的特徵改變量
plt.subplot(1, 2, 1)
plt.barh(features, feature_changes[0], color='skyblue')
plt.xlabel('Feature Change')
plt.title('南光')
plt.gca().invert_yaxis()

# 繪製第二筆資料的特徵改變量
plt.subplot(1, 2, 2)
plt.barh(features, feature_changes[1], color='salmon')
plt.xlabel('Feature Change')
plt.title('八方雲集')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()