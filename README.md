# LogisticRegression
逻辑回归

练习一下逻辑回归的API，数据集用Wisconsin大学的乳腺癌数据集：[breast-cancer-wisconsin](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin)。

# 预处理
我把数据集下载到本地了，直接读取，然后预处理：
```
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
```

输出：
```
****************************** 数据集 ******************************
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 698 entries, 0 to 697
Data columns (total 11 columns):
1000025    698 non-null int64
5          698 non-null int64
1          698 non-null int64
1.1        698 non-null int64
1.2        698 non-null int64
2          698 non-null int64
1.3        698 non-null object
3          698 non-null int64
1.4        698 non-null int64
1.5        698 non-null int64
2.1        698 non-null int64
dtypes: int64(10), object(1)
memory usage: 60.1+ KB
None
****************************** x ******************************
<class 'pandas.core.frame.DataFrame'>
Int64Index: 682 entries, 0 to 697
Data columns (total 10 columns):
1000025    682 non-null int64
5          682 non-null int64
1          682 non-null int64
1.1        682 non-null int64
1.2        682 non-null int64
2          682 non-null int64
1.3        682 non-null object
3          682 non-null int64
1.4        682 non-null int64
1.5        682 non-null int64
dtypes: int64(9), object(1)
memory usage: 58.6+ KB
None
****************************** y ******************************
<class 'pandas.core.frame.DataFrame'>
Int64Index: 682 entries, 0 to 697
Data columns (total 1 columns):
2.1    682 non-null int64
dtypes: int64(1)
memory usage: 10.7 KB
None
```

# 逻辑回归

建模：
```
# 逻辑回归模型
# C: 正则化力度
lg = LogisticRegression(C=1.0)
lg.fit(x_train, y_train.values.flatten())
print_line("逻辑回归建模")
print("回归系数：{}".format(lg.coef_))
```

输出：
```
****************************** 逻辑回归建模 ******************************
回归系数：[[0.07121828 1.25589804 0.61335804 0.78423711 0.84763091 0.20571784
  1.35411147 0.95385004 0.7373611  0.60707194]]
```

# 预测
```
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
```

输出：
```
****************************** 真实值 ******************************
[2 2 4 2 2 2 2 2 4 2 2 2 4 2 4 2 2 4 2 2]
****************************** 预测值 ******************************
[2 2 2 2 2 2 2 2 2 2 2 2 4 2 4 2 2 4 2 2]
****************************** 召回率 ******************************
             precision    recall  f1-score   support

         良性       0.97      0.96      0.97       112
         恶性       0.93      0.95      0.94        59

avg / total       0.96      0.96      0.96       171
```

对于本业务场景来说，准确意义不大，所以使用召回率来评价模型。
