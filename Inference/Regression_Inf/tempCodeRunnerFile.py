import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 讀取訓練資料
train_data = pd.read_csv("dataset/Top100_final.csv", encoding='utf-8')

# 選取特徵和目標變量
X_train = train_data.iloc[:, 2:12]  # 第2到第11列為特徵
y_train = train_data['Top 100']     # 目標欄

# 標準化訓練資料
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 訓練模型
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train_scaled, y_train)

# 獲取決策邊界的係數和截距
coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]


my_data = pd.read_csv("dataset/test_cody.csv")
my_data_x = my_data.iloc[:, 2:12]
 
# 對測試資料進行與訓練資料相同的標準化處理
my_data_x_scaled = scaler.transform(my_data_x)

# 使用訓練好的模型進行預測
predictions = logreg.predict(my_data_x_scaled)



# 計算需要改變的特徵值
def calculate_feature_changes(data_x, coefficients, intercept, scaler, max_change=5, max_return_change=4):
    data_x_scaled = scaler.transform(data_x)
    decision_values = np.dot(data_x_scaled, coefficients) + intercept
    
    changes = np.zeros_like(data_x_scaled)
    for i in range(len(data_x)):
        if decision_values[i] >= 0:
            continue  # 如果達標，不需要調整
        for j in range(len(data_x.columns)):
            required_change = (0 - decision_values[i]) / np.abs(coefficients[j])
            if j in ['Reture_A', 'Return_Eq']:
                if np.abs(required_change) > max_return_change:
                    # 不能超過限制
                    changes[i][j] = np.sign(coefficients[j]) * max_return_change * scaler.scale_[j]
                else:
                    changes[i][j] = required_change * np.sign(coefficients[j]) * scaler.scale_[j]
            elif np.abs(required_change) > max_change:
                # 最多只能跟限制的一樣多
                changes[i][j] = np.sign(coefficients[j]) * max_change * scaler.scale_[j]
            else:
                changes[i][j] = required_change * np.sign(coefficients[j]) * scaler.scale_[j]
    
    changes_scaled = changes / scaler.scale_
    
    return changes_scaled

# 計算改變特徵值以達到 Top 100 = 1
changes_needed = calculate_feature_changes(my_data_x, coefficients, intercept, scaler)
changes_df = pd.DataFrame(changes_needed, columns=my_data_x.columns)
changes_df["Top_100"] = predictions

# 打印結果
print("各特徵應改變的量:")
print(changes_df)


# 將調整後的特徵應用到測試資料集中
adjusted_data_x = my_data_x.copy()
adjusted_data_x += changes_df



# 使用訓練好的模型進行預測
predictions_after_adjustment = logreg.predict(adjusted_data_x)

# 打印預測結果
print("調整後的預測結果:")
print(predictions_after_adjustment)


import matplotlib.pyplot as plt

# 設置圖形大小
plt.figure(figsize=(10, 6))

# 繪製每個特徵的變化量
for feature in my_data_x.columns:
    plt.bar(feature, changes_df[feature].mean(), label=feature)

# 繪製最大調整量的水平線
plt.axhline(y=5, color='red', linestyle='--', label='最大調整量')
plt.axhline(y=-5, color='red', linestyle='--')

# 添加標籤和標題
plt.xlabel('特徵')
plt.ylabel('變化量')
plt.title('特徵變化')

# 添加圖例
plt.legend()

# 旋轉特徵標籤
plt.xticks(rotation=45)

# 顯示圖形
plt.tight_layout()
plt.show()
