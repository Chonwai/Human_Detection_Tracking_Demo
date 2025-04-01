#!/usr/bin/env python3
"""
CUDA診斷腳本 - 用於檢查NVIDIA Jetson上的CUDA配置
"""

import os
import sys
import subprocess
import platform

def print_section(title):
    """打印帶格式的段落標題"""
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

def run_command(command):
    """執行shell命令並返回輸出"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"命令執行錯誤: {e}")
        if e.stderr:
            print(f"錯誤信息: {e.stderr}")
        return None

def check_system_info():
    """檢查系統基本信息"""
    print_section("系統信息")
    print(f"操作系統: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    
    # 檢查是否為Jetson平台
    if os.path.exists('/etc/nv_tegra_release'):
        print("檢測到NVIDIA Jetson平台")
        jetson_info = run_command('cat /etc/nv_tegra_release')
        if jetson_info:
            print(f"Jetson版本信息: {jetson_info.strip()}")
    else:
        print("未檢測到NVIDIA Jetson平台")
    
    # 檢查處理器架構
    print(f"處理器架構: {platform.machine()}")

def check_cuda_installation():
    """檢查CUDA安裝"""
    print_section("CUDA安裝檢查")
    
    # 檢查CUDA版本
    cuda_version = run_command('nvcc --version')
    if cuda_version:
        print("CUDA編譯器已安裝:")
        print(cuda_version)
    else:
        print("未找到CUDA編譯器(nvcc)，請確認CUDA是否正確安裝")
    
    # 檢查GPU狀態
    nvidia_smi = run_command('nvidia-smi')
    if nvidia_smi:
        print("GPU信息:")
        print(nvidia_smi)
    else:
        tegrastats = run_command('tegrastats')
        if tegrastats:
            print("Tegra狀態信息可用（按Ctrl+C停止）:")
            print("請另開一個終端執行: tegrastats")
        else:
            print("未找到nvidia-smi或tegrastats，無法獲取GPU信息")

def check_pytorch_cuda():
    """檢查PyTorch是否支持CUDA"""
    print_section("PyTorch CUDA支持")
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        # 檢查CUDA可用性
        cuda_available = torch.cuda.is_available()
        print(f"CUDA可用: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '不可用'}")
            print(f"GPU設備數量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  設備 {i}: {torch.cuda.get_device_name(i)}")
                
            # 測試CUDA運算
            print("\n執行簡單的CUDA運算測試...")
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            z = x @ y  # 矩陣乘法
            end.record()
            
            # 等待運算完成
            torch.cuda.synchronize()
            print(f"矩陣乘法(1000x1000)耗時: {start.elapsed_time(end):.2f} ms")
            
        else:
            print("PyTorch無法使用CUDA，請檢查安裝")
            
    except ImportError:
        print("無法導入PyTorch，請確認其已正確安裝")

def provide_solution_suggestions():
    """提供針對Jetson平台的解決方案建議"""
    print_section("解決方案建議")
    
    print("1. 針對Jetson平台安裝正確的PyTorch版本:")
    print("   - 請訪問 https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048")
    print("   - 根據您的JetPack版本選擇對應的PyTorch安裝包")
    
    print("\n2. 確保環境變量正確設置:")
    print("   export PATH=/usr/local/cuda/bin:$PATH")
    print("   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
    
    print("\n3. 修改應用程式的設備選擇邏輯，添加明確的CUDA設備選擇和錯誤處理")

if __name__ == "__main__":
    check_system_info()
    check_cuda_installation()
    check_pytorch_cuda()
    provide_solution_suggestions()
    
    print("\n診斷完成，請根據上述信息分析CUDA配置問題") 