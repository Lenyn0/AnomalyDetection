import pandas as pd
import os
from pathlib import Path

def merge_csvs_in_folder(folder_path, output_filename=None):
    """
    将指定文件夹下的所有CSV文件合并为一个CSV文件
    
    参数:
    folder_path: 要合并CSV文件的文件夹路径
    output_filename: 输出文件名，如果为None则使用文件夹名
    """
    folder_path = Path(folder_path)
    
    # 检查文件夹是否存在
    if not folder_path.exists():
        print(f"文件夹 {folder_path} 不存在")
        return
    
    # 获取文件夹中的所有CSV文件
    csv_files = list(folder_path.glob("*.csv"))
    
    if not csv_files:
        print(f"在文件夹 {folder_path} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # 读取并合并所有CSV文件
    dataframes = []
    for csv_file in sorted(csv_files):  # 按文件名排序
        try:
            df = pd.read_csv(csv_file)
            print(f"读取 {csv_file.name}: {len(df)} 行")
            dataframes.append(df)
        except Exception as e:
            print(f"读取 {csv_file.name} 时出错: {e}")
    
    if not dataframes:
        print("没有成功读取任何CSV文件")
        return
    
    # 合并所有数据框
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # 如果有Time列，按时间排序
    if 'Time' in merged_df.columns:
        merged_df['Time'] = pd.to_datetime(merged_df['Time'])
        merged_df = merged_df.sort_values('Time').reset_index(drop=True)
    
    # 生成输出文件名
    if output_filename is None:
        folder_name = folder_path.name
        output_filename = f"{folder_name}_merged.csv"
    
    # 保存合并后的文件
    output_path = folder_path / output_filename
    merged_df.to_csv(output_path, index=False)
    
    print(f"合并完成！")
    print(f"输出文件: {output_path}")
    print(f"总行数: {len(merged_df)}")
    
    return merged_df

# 使用示例
if __name__ == "__main__":
    # 合并单个文件夹
    folder_to_merge = "data/7.工银信使"
    merged_data = merge_csvs_in_folder(folder_to_merge)
    
    # 如果要合并所有子文件夹，可以使用以下代码
    def merge_all_folders(base_path="data"):
        """合并data文件夹下所有子文件夹的CSV文件"""
        base_path = Path(base_path)
        
        for subfolder in base_path.iterdir():
            if subfolder.is_dir():
                print(f"\n正在处理文件夹: {subfolder.name}")
                merge_csvs_in_folder(subfolder)
    
    #合并所有文件夹
    #merge_all_folders()