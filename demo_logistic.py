import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def print_line(title):
    print("*" * 30 + " {} ".format(title) + "*" * 30)


# 数据集地址：https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin

print_line("数据集")
df = pd.read_csv("breast-cancer-wisconsin.data.csv")
print(df.info())

# 缺失值处理，把`?`替换成`np.nan`
df.replace(to_replace="?", value=np.nan, inplace=True)
df.dropna(inplace=True)

# 前面10列是特征值，最后一列为目标值
x = df.iloc[:, range(10)]
print_line("x")
print(x.info())

y = df.iloc[:, [10]]
print_line("y")
print(y.info())

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 逻辑回归需要对特征数据做标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

# 逻辑回归模型
# C: 正则化力度
lg = LogisticRegression(C=1.0)
lg.fit(x_train, y_train.values.flatten())
print_line("逻辑回归建模")
print("回归系数：{}".format(lg.coef_))

# 预测
print_line("真实值")
print(y_test[:20].values.flatten())
print_line("预测值")
y_predict = lg.predict(x_test)
print(y_predict[:20])

# 召回率
print_line("召回率")
report = classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"])
print(report)




