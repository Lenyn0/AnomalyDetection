import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（适用于Windows系统）
plt.rcParams['font.family'] = 'SimHei'       # 黑体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={'Time': 'ds', 'Count': 'value'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    df.sort_values('ds', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_scores(df, interval_minutes=30, base_interval=5):
    values = df['value'].values
    n = len(values)

    window_size = interval_minutes // base_interval

    # 增量计算
    delta = (values - np.roll(values, window_size)).astype(float)
    delta[:window_size] = np.nan


    df['delta_30min'] = delta

    # 当前值归一化，避免除0，加一个极小值
    min_val = np.min(values)
    max_val = np.max(values)
    norm_value = (values - min_val) / (max_val - min_val + 1e-9)
    df['norm_value'] = norm_value


    # 计算综合得分，weight由归一化当前值决定
    weight = norm_value
    score = weight* delta
    df['score'] = score

    return df

def plot_all(df):
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axs[0].plot(df['ds'], df['value'], label='原始数据')
    axs[0].set_title('原始数据')
    axs[0].legend()

    axs[1].plot(df['ds'], df['delta_30min'], label='30分钟增量 (delta)', color='orange')
    axs[1].legend()


    axs[2].plot(df['ds'], df['score'], label='综合得分 (加权 z-score)', color='red')
    axs[2].axhline(y=0, color='gray', linestyle='--')
    axs[2].set_title('突增得分')
    axs[2].legend()

    plt.xlabel('时间')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #path = 'data\\35.企业信使\\35.企业信使_merged.csv'
    #path = 'data\\31.企网汇款\\31.企网汇款_merged.csv'
    #path = 'data\\23.三方平台快捷支付\\23.三方平台快捷支付_merged.csv'
    path = 'data\\9.手机银行\\9.手机银行_merged.csv'
    df = load_and_preprocess(path)
    df = calculate_scores(df, interval_minutes=30, base_interval=5)
    plot_all(df)
