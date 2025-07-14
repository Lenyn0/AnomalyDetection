import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
import warnings
import streamlit as st
import matplotlib.font_manager as fm
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def extract_number_from_filename(filename):
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def create_time_series_plot(df, time_col, numeric_cols, title):
    if not time_col or len(df) <= 1 or len(numeric_cols) == 0:
        return None
    fig = go.Figure()
    for col in numeric_cols:
        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[col],
            mode='lines+markers',
            name=col,
            line=dict(width=2),
            marker=dict(size=4),
            hovertemplate=f'<b>{col}</b><br>时间: %{{x}}<br>值: %{{y}}<extra></extra>'
        ))
    fig.update_layout(
        title=f'{title} - 时间序列趋势',
        xaxis_title='时间',
        yaxis_title='数值',
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    return fig

def create_box_plot(df, numeric_cols, title):
    if len(numeric_cols) == 0:
        return None
    fig = go.Figure()
    for col in numeric_cols[:5]:
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            hovertemplate=f'<b>{col}</b><br>值: %{{y}}<extra></extra>'
        ))
    fig.update_layout(
        title=f'{title} - 数值分布箱线图',
        yaxis_title='数值',
        height=500,
        showlegend=True
    )
    return fig

def get_available_files(data_path="data"):
    data_path = Path(data_path)
    merged_files = list(data_path.glob("*/*_merged.csv"))
    file_info = []
    for file in merged_files:
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
            file_info.append({
                'path': file,
                'name': file.parent.name,
                'rows': len(df),
                'columns': len(df.columns),
                'sort_key': extract_number_from_filename(file.parent.name)
            })
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")
    file_info.sort(key=lambda x: x['sort_key'])
    return file_info

def main():
    st.set_page_config(page_title="数据可视化分析", layout="wide")
    st.title("📊 数据可视化分析工具")
    st.markdown("选择要分析的业务类型数据")

    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False

    available_files = get_available_files()
    if not available_files:
        st.error("❌ 未找到任何merged.csv文件")
        st.info("请确保data文件夹中存在*_merged.csv文件")
        return

    file_options = {info['name']: info for info in available_files}
    selected_name = st.selectbox(
        "选择要分析的业务类型:",
        options=list(file_options.keys()),
        format_func=lambda x: f"{x} ({file_options[x]['rows']}行, {file_options[x]['columns']}列)"
    )

    if selected_name:
        selected_file = file_options[selected_name]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("数据行数", selected_file['rows'])
        with col2:
            st.metric("数据列数", selected_file['columns'])
        with col3:
            st.metric("文件大小", f"{selected_file['path'].stat().st_size // 1024}KB")

        # 🔍 分析按钮逻辑
        if st.button("🔍 开始分析", type="primary"):
            st.session_state.analyze_clicked = True


        if st.session_state.analyze_clicked:
            with st.spinner("正在读取数据..."):
                try:
                    df = pd.read_csv(selected_file['path'], encoding='utf-8-sig')

                    time_col = None
                    for col in df.columns:
                        if 'time' in col.lower() or 'date' in col.lower():
                            time_col = col
                            break

                    if time_col:
                        df[time_col] = pd.to_datetime(df[time_col])
                        df = df.sort_values(time_col)

                    numeric_cols = df.select_dtypes(include=[np.number]).columns

                    tab1, tab2, tab3, tab4 = st.tabs(["📈 时间序列", "📊 分布图", "📦 箱线图", "📋 数据统计"])

                    with tab1:
                        st.subheader("📈 时间序列分析")
                        if time_col and len(df) > 1 and len(numeric_cols) > 0:
                            min_date = df[time_col].min().date()
                            max_date = df[time_col].max().date()
                            start_date, end_date = st.date_input(
                                "选择日期范围:",
                                value=(min_date, max_date),
                                min_value=min_date,
                                max_value=max_date
                            )
                            mask = (df[time_col].dt.date >= start_date) & (df[time_col].dt.date <= end_date)
                            df_filtered = df.loc[mask]
                            selected_cols = st.multiselect(
                                "选择要显示的数值列:",
                                options=list(numeric_cols),
                                default=list(numeric_cols[:3]),
                                max_selections=5
                            )
                            if selected_cols and not df_filtered.empty:
                                fig = create_time_series_plot(df_filtered, time_col, selected_cols, selected_name)
                                st.plotly_chart(fig, use_container_width=True)
                                st.info(f"📅 筛选时间范围: {df_filtered[time_col].min().strftime('%Y-%m-%d')} 至 {df_filtered[time_col].max().strftime('%Y-%m-%d')}")
                            elif df_filtered.empty:
                                st.warning("❌ 当前筛选时间范围无数据")
                        else:
                            st.warning("⚠️ 没有时间序列数据或数值列")

                    with tab2:
                        st.subheader("📊 数值分布分析")
                        if len(numeric_cols) > 0:
                            selected_col = st.selectbox(
                                "选择要分析的列:",
                                options=list(numeric_cols),
                                index=0
                            )
                            bins = st.slider("直方图分组数:", min_value=10, max_value=50, value=30)
                            fig = go.Figure(data=[go.Histogram(
                                x=df[selected_col].dropna(),
                                nbinsx=bins,
                                name=selected_col,
                                hovertemplate=f'<b>{selected_col}</b><br>区间: %{{x}}<br>频次: %{{y}}<extra></extra>'
                            )])
                            fig.update_layout(
                                title=f'{selected_col} 分布直方图',
                                xaxis_title=selected_col,
                                yaxis_title='频次',
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("均值", f"{df[selected_col].mean():.2f}")
                            with col2:
                                st.metric("标准差", f"{df[selected_col].std():.2f}")
                            with col3:
                                st.metric("最小值", f"{df[selected_col].min():.2f}")
                            with col4:
                                st.metric("最大值", f"{df[selected_col].max():.2f}")
                        else:
                            st.warning("⚠️ 没有数值列")

                    with tab3:
                        st.subheader("📦 箱线图分析")
                        if len(numeric_cols) > 0:
                            fig = create_box_plot(df, numeric_cols, selected_name)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                st.subheader("⚠️ 异常值检测")
                                for col in numeric_cols[:3]:
                                    Q1 = df[col].quantile(0.25)
                                    Q3 = df[col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                                    if len(outliers) > 0:
                                        st.write(f"**{col}**: 发现 {len(outliers)} 个异常值")
                                    else:
                                        st.write(f"**{col}**: 未发现异常值")
                        else:
                            st.warning("⚠️ 没有数值列")

                    with tab4:
                        st.subheader("📋 数据统计")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**基本信息**")
                            st.write(f"- 总行数: {len(df)}")
                            st.write(f"- 总列数: {len(df.columns)}")
                            st.write(f"- 缺失值总数: {df.isnull().sum().sum()}")
                            if time_col:
                                st.write(f"- 时间范围: {df[time_col].min().strftime('%Y-%m-%d')} 至 {df[time_col].max().strftime('%Y-%m-%d')}")
                        with col2:
                            st.markdown("**数值列统计**")
                            if len(numeric_cols) > 0:
                                for col in numeric_cols:
                                    with st.expander(f"📊 {col}"):
                                        st.write(f"- 均值: {df[col].mean():.2f}")
                                        st.write(f"- 标准差: {df[col].std():.2f}")
                                        st.write(f"- 最小值: {df[col].min():.2f}")
                                        st.write(f"- 最大值: {df[col].max():.2f}")
                                        st.write(f"- 缺失值: {df[col].isnull().sum()}")
                            else:
                                st.write("无数值列")
                except Exception as e:
                    st.error(f"❌ 数据读取失败: {e}")

    st.sidebar.title("ℹ️ 使用说明")
    st.sidebar.markdown("""
    1. 选择要分析的业务类型  
    2. 点击"开始分析"按钮  
    3. 使用不同标签页查看各种分析结果
    
    **功能介绍：**
    - 📈 时间序列: 可选时间段的交互式趋势图  
    - 📊 分布图: 数值直方图和统计  
    - 📦 箱线图: 异常值检测和分布  
    - 📋 数据统计: 缺失值和基本汇总
    """)

    st.sidebar.subheader("📁 可用文件")
    for info in available_files:
        st.sidebar.write(f"• {info['name']}")

if __name__ == "__main__":
    main()
