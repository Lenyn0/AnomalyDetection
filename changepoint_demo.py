import os
import pandas as pd
import matplotlib.pyplot as plt
from changepoint_online import Focus, Poisson

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={'Time': 'ds', 'Count': 'original_y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    non_zero = df[df['original_y'] > 0]['original_y']
    y_min, y_max = non_zero.min(), non_zero.max()
    df['norm_y'] = df['original_y'].apply(lambda x: (x - y_min) / (y_max - y_min) if x > 0 else 0)
    return df

def realtime_changepoint_detection(df, threshold=10.0):
    detector = Focus(Poisson())
    cps = []
    stats = []

    for idx, row in df.iterrows():
        y = row['norm_y']
        detector.update(y)
        stat = detector.statistic()
        stats.append(stat)

        if stat >= threshold:
            info = detector.changepoint()
            cps.append((row['ds'], info['changepoint'], stat))
            detector = Focus(Poisson())  # 重新初始化

    df['statistic'] = stats
    return cps, df

def plot_results(df, changepoints):
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 绘制时间序列
    axs[0].plot(df['ds'], df['norm_y'], label='Normalized Series')
    for t, cp, stat in changepoints:
        axs[0].axvline(t, color='red', linestyle='--', alpha=0.7)
        axs[0].text(t, df['norm_y'].max()*0.9, f'{stat:.1f}', rotation=90, color='red')
    axs[0].set_title("Time Series with Detected Changepoints")
    axs[0].legend()

    # 绘制 statistic 曲线
    axs[1].plot(df['ds'], df['statistic'], color='purple', label='Statistic')
    axs[1].axhline(y=threshold, color='gray', linestyle='--', label='Threshold')
    axs[1].set_title("Changepoint Statistic Over Time")
    axs[1].legend()

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path = 'data\\35.企业信使\\35.企业信使_merged.csv'
    #path = 'data\\31.企网汇款\\31.企网汇款_merged.csv'
    #path = 'data\\23.三方平台快捷支付\\23.三方平台快捷支付_merged.csv'
    #path = 'data\\9.手机银行\\9.手机银行_merged.csv'
    threshold = 13
    df = load_and_preprocess(path)
    cps, df_with_stat = realtime_changepoint_detection(df, threshold=threshold)
    print("Detected changepoints:", cps)
    plot_results(df_with_stat, cps)
