from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

data = pd.read_csv('Top100_final.csv')
####
wanna_know_data = pd.read_csv('test_cody.csv')


data = data.drop(columns=['N_name', 'Name'])
####
wanna_know_data = wanna_know_data.drop(columns=['N_name', 'Name'])

# 分離特徵和目標
X = data.drop(columns='Top 100')
####
wanna_know_data_X = wanna_know_data.drop(columns="Top_100")

y = data['Top 100']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)



# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
wanna_know_data_X_scaled = scaler.transform(wanna_know_data_X)

svm_classifier = SVC(kernel='rbf', C=7, gamma=0.3)

svm_classifier.fit(X_train_scaled, y_train)

# y_pred = svm_classifier.predict(X_test_scaled)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred)) 

# 對測試集進行預測
y_pred = svm_classifier.predict(X_test_scaled)

# 輸出混淆矩陣和分類報告
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 對新的資料進行預測
pred = svm_classifier.predict(wanna_know_data_X_scaled)
print(pred)