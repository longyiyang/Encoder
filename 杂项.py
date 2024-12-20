import psutil
import subprocess
import time

def get_cpu_usage():
    # 获取 CPU 使用百分比
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage

def get_gpu_usage():
    # 使用 nvidia-smi 获取 GPU 使用百分比
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
    )
    gpu_usage = result.decode('utf-8').strip()  # 解析返回的百分比
    return gpu_usage

def monitor_system():
    try:
        while True:
            # 获取 CPU 使用情况
            cpu_usage = get_cpu_usage()
            # 获取 GPU 使用情况
            gpu_usage = get_gpu_usage()
            
            # 输出 CPU 和 GPU 使用情况
            print(f"CPU Usage: {cpu_usage}% | GPU Usage: {gpu_usage}%")
            
            # 每秒更新一次
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

# 开始实时监控 CPU 和 GPU 使用情况
monitor_system()
