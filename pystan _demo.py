import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 读入数据集
df = pd.read_csv('data\\1.信用卡\\1.信用卡_merged.csv')
df.rename(columns={'Time': 'ds', 'Count': 'y'}, inplace=True)

print(df.head())
# 拟合模型
m = Prophet()
m.fit(df)

# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = m.make_future_dataframe(periods=10)
future.tail()
# 预测数据集
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# 展示预测结果
m.plot(forecast)
# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
m.plot_components(forecast)
plt.show()
