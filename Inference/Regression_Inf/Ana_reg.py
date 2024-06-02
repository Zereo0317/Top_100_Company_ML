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
def calculate_feature_changes(data_x, coefficients, intercept, scaler):
    data_x_scaled = scaler.transform(data_x)
    decision_values = np.dot(data_x_scaled, coefficients) + intercept
    
    changes = []
    for i, decision_value in enumerate(decision_values):
        if decision_value >= 0:
            changes.append([0] * data_x.shape[1])
        else:
            required_change = (0 - decision_value) / np.abs(coefficients)
            changes.append(required_change)
    
    changes = np.array(changes)
    changes_scaled = changes / scaler.scale_
    
    return changes_scaled

# 計算改變特徵值以達到 Top 100 = 1
changes_needed = calculate_feature_changes(my_data_x, coefficients, intercept, scaler)
changes_df = pd.DataFrame(changes_needed, columns=my_data_x.columns)
changes_df["Top_100"] = predictions

# 打印結果
print("各特徵應改變的量:")
print(changes_df)

changes_df.to_csv("dataset/reg_result.csv",index=False)
