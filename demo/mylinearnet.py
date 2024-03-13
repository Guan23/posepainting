#!usr/bin/env python
# _*_ encoding: utf-8 _*_
# Author: Guan
# Create Time: 2021/12/15 上午10:18

import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import joblib

data_folder = "/home/gxk/HRnet/human_size/"

# X是自变量，Y是因变量
data = pd.read_csv(data_folder + "test.csv")
X = data.iloc[:-2, :8]
Y = data.iloc[:-2, 8:]
x = np.array(X)
y = np.array(Y)

# print("x:\n{}".format(x))
print("x.shape:{}\n".format(x.shape))
# print("y:\n{}".format(y))
print("y.shape:{}".format(y.shape))

pred = np.array(data.iloc[-1, :8]).reshape(-1, 8)
print("\npred:{}\n".format(pred))

clf = MultiOutputRegressor(Ridge(random_state=123)).fit(x, y)  # 多输出的回归器

# 保存权重
joblib.dump(clf, data_folder + 'clf.pkl', compress=0)

# 加载权重
clf3 = joblib.load(data_folder + 'clf.pkl')

# print(clf3.get_params())
print(clf3.predict(pred))
# print(type(clf.predict(pred)))
# print(clf.predict(x[[0]]))
