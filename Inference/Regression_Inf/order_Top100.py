import pandas as pd


data = pd.read_csv("dataset/Top100.csv")

# # 删除第一列
# data = data.iloc[:,1:]

# data.to_csv("dataset/Top100.csv",index=False)

result_data = pd.read_csv("dataset/reg_result.csv")
feature_order = result_data.columns.tolist()

top100_data_sorted = data[feature_order]

top100_data_sorted.to_csv("dataset/Top100.csv", index=False)