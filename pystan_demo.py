import os
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import matplotlib.pyplot as plt

# === 函数定义 ===
def load_and_preprocess(path):
    df = pd.read_csv(path)
    df.rename(columns={'Time': 'ds', 'Count': 'original_y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])

    # 归一化
    non_zero = df[df['original_y'] > 0]['original_y']
    y_min, y_max = non_zero.min(), non_zero.max()

    if y_max != y_min:
        df['norm_y'] = df['original_y'].apply(lambda x: (x - y_min) / (y_max - y_min) if x > 0 else 0)
    else:
        df['norm_y'] = df['original_y']

    return df, y_min, y_max


def get_model(df, model_path):
    if os.path.exists(model_path):
        print("📦 加载已保存模型...")
        with open(model_path, 'r') as fin:
            model = model_from_json(fin.read())
    else:
        print("🔧 训练新模型...")
        model = Prophet(
            changepoint_prior_scale=0.8,
            changepoint_range=0.9,
            seasonality_mode='additive'
        )
        model.fit(df.rename(columns={'norm_y': 'y'}))
        with open(model_path, 'w') as fout:
            fout.write(model_to_json(model))
        print(f"✅ 模型保存至 {model_path}")
    return model


def calc_error(row):
    if pd.isna(row['norm_y']):
        return 0
    if row['yhat_lower'] <= row['norm_y'] <= row['yhat_upper']:
        return 0
    return min(abs(row['norm_y'] - row['yhat_lower']), abs(row['norm_y'] - row['yhat_upper']))


def detect_outliers(df, method='zscore', z_thresh=9, iqr_factor=1.5, ratio_thresh=2.0):
    if method == 'zscore':
        mean_err = df['error'].mean()
        std_err = df['error'].std()
        outliers = df[df['error'] > (mean_err + z_thresh * std_err)]

    elif method == 'iqr':
        q1 = df['error'].quantile(0.25)
        q3 = df['error'].quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + iqr_factor * iqr
        outliers = df[df['error'] > threshold]

    elif method == 'deviation_ratio':
        if 'deviation_ratio' not in df.columns:
            df['deviation_ratio'] = df.apply(
                lambda row: 0 if pd.isna(row['norm_y']) or row['yhat_lower'] <= row['norm_y'] <= row['yhat_upper']
                else abs(row['norm_y'] - row['yhat']) / (row['yhat_upper'] - row['yhat_lower'] + 1e-9),
                axis=1
            )
        outliers = df[df['deviation_ratio'] > ratio_thresh]

    else:
        raise ValueError("method 参数必须是 'zscore'、'iqr' 或 'deviation_ratio'")

    return outliers


def plot_forecast(model, forecast, outliers):
    fig1 = model.plot(forecast)
    plt.scatter(outliers['ds'], outliers['norm_y'], color='red', label='Anomaly', s=15, zorder=5)
    plt.legend()
    plt.title('Forecast with Detected Outliers')

    fig2 = model.plot_components(forecast)
    plt.show()


# === 配置 ===
#path = 'data\\35.企业信使\\35.企业信使_merged.csv'
#path = 'data\\31.企网汇款\\31.企网汇款_merged.csv'
#path = 'data\\23.三方平台快捷支付\\23.三方平台快捷支付_merged.csv'
path = 'data\\9.手机银行\\9.手机银行_merged.csv'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

df, y_min, y_max = load_and_preprocess(path)

# 生成模型路径
csv_filename = os.path.basename(path).replace('merged', '')
model_filename = os.path.splitext(csv_filename)[0] + '_model.json'
model_path = os.path.join(model_dir, model_filename)

# 获取模型并预测
model = get_model(df, model_path)
future = model.make_future_dataframe(periods=24 * 12, freq='5min')
forecast = model.predict(future)

# 合并预测结果与原始数据
merged = pd.merge(forecast, df[['ds', 'norm_y']], on='ds', how='left')
merged['error'] = merged.apply(calc_error, axis=1)
cumulative_error = merged['error'].sum()

# 输出误差信息
print(f"🎯 累计误差：{cumulative_error:.2f}")
print(merged[['ds', 'norm_y', 'yhat', 'yhat_lower', 'yhat_upper', 'error']].tail())

# 检测异常
outliers = detect_outliers(merged, method='deviation_ratio')  # 可改为 'iqr' 或 'deviation_ratio'

# 可视化
plot_forecast(model, forecast, outliers)
