import subprocess
import sys

def install_requirements():
    """安装必要的包"""
    packages = ['streamlit', 'pandas', 'matplotlib', 'seaborn', 'numpy','plotly']
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"正在安装 {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def run_streamlit():
    """运行Streamlit应用"""
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'visualize.py'])
    except KeyboardInterrupt:
        print("\n应用已停止")
    except Exception as e:
        print(f"运行出错: {e}")

if __name__ == "__main__":
    print("正在检查依赖...")
    install_requirements()
    print("\n启动Web应用...")
    run_streamlit()