# -*- coding: utf-8 -*-
# 新型冠状病毒确诊患者预测，使用时间序列ARMA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 数据加载
df = pd.read_csv('./wuhan.csv')
# 将时间作为df的索引
df.Timestamp = pd.to_datetime(df.Timestamp)
df.index = df.Timestamp
# 数据探索
print(df.head())

ps = range(0, 3)
qs = range(0, 3)
parameters = product(ps, qs)
parameters_list = list(parameters)
# 寻找最优ARMA模型参数，即best_aic最小
results = []
best_aic = float("inf") # 正无穷
for param in parameters_list:
    try:
        model = ARMA(df.amount,order=(param[0], param[1])).fit()
    except ValueError:
        print('参数错误:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
# 输出最优模型
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print('最优模型: ', best_model.summary())
# 预测
df2 = df[['amount']]
date_list = [datetime(2020, 1, 30), datetime(2020, 1, 31), datetime(2020, 2, 1), datetime(2020, 2, 2),]
future = pd.DataFrame(index=date_list, columns=df.columns)
df2 = pd.concat([df2, future])
df2['forecast'] = best_model.predict(start=0, end=14)
print(df2)
# 预测结果显示
plt.figure(figsize=(20,7))
df2.amount.plot(label=u'实际')
df2.forecast.plot(color='r', ls='--', label='预测')
plt.legend()
plt.title('确诊人数')
plt.xlabel('日期')
plt.ylabel('人')
plt.show()
