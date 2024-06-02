# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # 讀取 Top100.csv 文件
# data = pd.read_csv("dataset/Top100.csv")

# # 提取特徵和目標變量
# X = data.drop(columns=["Top_100"])
# y = data['Top_100']

# # 初始化並擬合 StandardScaler
# scaler = StandardScaler()
# scaler.fit(X)

# # 保存縮放參數（均值和標準差）
# mean = scaler.mean_
# std = scaler.scale_

# # 載入目標數據
# target_data = pd.read_csv("dataset/reg_result.csv")
# target_data = target_data.drop(columns=["Top_100"])


# # 使用訓練數據的縮放參數對目標數據進行反向轉換
# target_data_inverse = (target_data * std) + mean

# target_data_inverse.to_csv("dataset/final_result.csv",index=False)