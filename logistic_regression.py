# -*- coding: utf-8 -*-
"""Logistic Regression

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1f0uh7NnOxxFym7aoff62gXpP2wQsjHgU
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path="/content/drive/MyDrive/CSV原始檔/Top100_final-1.csv"
data=pd.read_csv(path)
display (data)

pip install pandas scikit-learn

pip install matplotlib scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 讀取資料
path = "/content/drive/MyDrive/CSV原始檔/Top100_final-1.csv"
data = pd.read_csv(path, encoding='utf-8')

# 查看資料前幾行
print(data.head())

# 選取特徵和目標變量
X = data.iloc[:, 1:12]  # B欄到L欄
y = data['Top 100']     # M欄

# 分割資料為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 訓練模型
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train_scaled, y_train)

# 預測
y_pred = logreg.predict(X_test_scaled)

# 評估模型
print(classification_report(y_test, y_pred))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# 讀取資料
path = "/content/drive/MyDrive/CSV原始檔/Top100_final-1.csv"
data = pd.read_csv(path, encoding='utf-8')

# 檢查缺失值
print(data.isnull().sum())
data.fillna(data.mean(), inplace=True)

# 選取特徵和目標變量
X = data.iloc[:, 1:12]  # B欄到L欄
y = data['Top 100']     # M欄

# 分割資料為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 訓練模型
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train_scaled, y_train)

# 預測
y_pred = logreg.predict(X_test_scaled)
y_prob = logreg.predict_proba(X_test_scaled)[:, 1]

# 評估模型
print(classification_report(y_test, y_pred))

# 繪製混淆矩陣
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 繪製ROC曲線
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# 讀取資料
path = "/content/drive/MyDrive/CSV原始檔/Top100_final-1.csv"
data = pd.read_csv(path, encoding='utf-8')

# 檢查缺失值並填充
data.fillna(data.mean(), inplace=True)

# 選取特徵和目標變量
X = data.iloc[:, 1:12]  # B欄到L欄
y = data['Top 100']     # M欄

# 分割資料為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 訓練模型
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train_scaled, y_train)

# 預測
y_pred = logreg.predict(X_test_scaled)
y_prob = logreg.predict_proba(X_test_scaled)[:, 1]

# 評估模型
print(classification_report(y_test, y_pred))

# 繪製混淆矩陣
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 繪製ROC曲線
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 繪製每個特徵的回歸係數
coefficients = logreg.coef_[0]
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, coefficients, color='blue')
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients from Logistic Regression')
plt.grid(True)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# 讀取資料
path = "/content/drive/MyDrive/CSV原始檔/Top100_final-1.csv"
data = pd.read_csv(path, encoding='utf-8')

# 檢查缺失值並填充
data.fillna(data.mean(), inplace=True)

# 選取特徵和目標變量
X = data.iloc[:, 1:12]  # B欄到L欄
y = data['Top 100']     # M欄

# 分割資料為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 創建 subplot 的排列
n_features = X.shape[1]
n_rows = (n_features + 2) // 3
n_cols = min(n_features, 3)

# 創建 subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

# 迭代每個特徵，繪製 Logistic Regression 結果圖表
for idx, feature in enumerate(X.columns):
    # 創建 Logistic Regression 模型
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train_scaled[:, idx].reshape(-1, 1), y_train)

    # 預測
    y_pred = logreg.predict(X_test_scaled[:, idx].reshape(-1, 1))
    y_prob = logreg.predict_proba(X_test_scaled[:, idx].reshape(-1, 1))[:, 1]

    # 評估模型
    print(f"Classification Report for Feature {feature}:")
    print(classification_report(y_test, y_pred))

    # 繪製分類曲線
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    ax = axes[idx // n_cols, idx % n_cols]
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic for Feature {feature}')
    ax.legend(loc="lower right")

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取資料
path = "/content/drive/MyDrive/CSV原始檔/Top100_final-1.csv"
data = pd.read_csv(path, encoding='utf-8')

# 檢查缺失值並填充
data.fillna(data.mean(), inplace=True)

# 選取特徵欄位
features = data.columns[1:12]  # B欄到L欄

# 設置圖表大小
plt.figure(figsize=(15, 10))

# 迭代每個特徵，繪製資料詳細散佈圖
for i, feature in enumerate(features, 1):
    plt.subplot(3, 4, i)
    sns.histplot(data=data, x=feature, kde=True)
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# 讀取資料
path = "/content/drive/MyDrive/CSV原始檔/Top100_final-1.csv"
data = pd.read_csv(path, encoding='utf-8')

# 檢查缺失值並填充
data.fillna(data.mean(), inplace=True)

# 選取特徵和目標變量
X = data.iloc[:, 1:12]  # B欄到L欄
y = data['Top 100']     # M欄

# 分割資料為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 設置圖表排列
n_features = X.shape[1]
n_rows = (n_features + 2) // 3
n_cols = min(n_features, 3)

# 定義顏色列表
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

# 創建 subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

# 迭代每個特徵，繪製 Logistic Regression 圖表
for idx, feature in enumerate(X.columns):
    row_idx = idx // n_cols
    col_idx = idx % n_cols
    ax = axes[row_idx, col_idx]

    # 單獨創建 Logistic Regression 模型並進行訓練
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train_scaled[:, idx].reshape(-1, 1), y_train)

    # 分組繪製散佈圖和曲線
    for label in np.unique(y_test):
        mask = (y_test == label)
        ax.scatter(X_test_scaled[:, idx][mask], y_test[mask], color=colors[label], label=f'Top {label}')

    # 使用訓練的模型預測特徵的平均曲線
    x_range = np.linspace(X_test_scaled[:, idx].min(), X_test_scaled[:, idx].max(), 300)
    y_prob_range = logreg.predict_proba(x_range.reshape(-1, 1))[:, 1]
    ax.plot(x_range, y_prob_range, color='blue', linewidth=3, label='Logistic Regression Curve')

    # 單獨計算每個特徵的Accuracy、Precision、Recall和F1-score
    feature_pred = logreg.predict(X_test_scaled[:, idx].reshape(-1, 1))
    feature_accuracy = accuracy_score(y_test, feature_pred)
    feature_precision = precision_score(y_test, feature_pred, zero_division=1)
    feature_recall = recall_score(y_test, feature_pred, zero_division=1)
    feature_f1 = f1_score(y_test, feature_pred, zero_division=1)

    ax.set_title(f'{feature.capitalize()} vs Top 100\nAccuracy: {feature_accuracy:.2f}, Precision: {feature_precision:.2f}\nRecall: {feature_recall:.2f}, F1-score: {feature_f1:.2f}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Top 100')
    ax.legend()

# 調整圖表排列
for idx in range(n_features, n_rows * n_cols):
    fig.delaxes(axes.flatten()[idx])

plt.tight_layout()
plt.show()