#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
平台識別和設備選擇工具

此模塊提供用於識別運行平台並選擇適當的計算設備的功能。
它為不同平台（Mac/Jetson/標準PC）提供了一致的接口，
使主應用程序代碼能夠以與平台無關的方式運行。
"""

import os
import platform
import logging
import subprocess
from typing import Tuple, Dict, Any, Optional, List

# 設置日誌
logger = logging.getLogger("PlatformUtils")

# 平台類型常量
PLATFORM_MAC = "mac"
PLATFORM_JETSON = "jetson"
PLATFORM_LINUX = "linux"
PLATFORM_WINDOWS = "windows"
PLATFORM_UNKNOWN = "unknown"

# 設備類型常量
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"


def get_platform() -> str:
    """
    識別當前運行平台
    
    返回：
        str: 平台識別符（'mac', 'jetson', 'linux', 'windows', 'unknown'）
    """
    system = platform.system()
    
    if system == "Darwin":
        return PLATFORM_MAC
    elif system == "Linux":
        # 檢查是否為Jetson平台
        if os.path.exists('/etc/nv_tegra_release'):
            return PLATFORM_JETSON
        else:
            return PLATFORM_LINUX
    elif system == "Windows":
        return PLATFORM_WINDOWS
    else:
        return PLATFORM_UNKNOWN


def is_jetson() -> bool:
    """
    檢查是否為Jetson平台
    
    返回：
        bool: 如果在Jetson上運行則為True
    """
    return get_platform() == PLATFORM_JETSON


def is_mac() -> bool:
    """
    檢查是否為Mac平台
    
    返回：
        bool: 如果在Mac上運行則為True
    """
    return get_platform() == PLATFORM_MAC


def is_apple_silicon() -> bool:
    """
    檢查是否為Apple Silicon芯片
    
    返回：
        bool: 如果在Apple Silicon上運行則為True
    """
    if not is_mac():
        return False
    
    return platform.machine() == "arm64"


def get_optimal_device() -> str:
    """
    確定當前平台的最佳計算設備
    
    返回：
        str: 設備標識符 ('cpu', 'cuda', 'mps')
    """
    try:
        # 動態導入torch以避免在導入此模塊時強制要求torch
        import torch
        
        # 對於Jetson和其他支持CUDA的平台
        if torch.cuda.is_available():
            logger.info("CUDA可用，使用GPU加速")
            return DEVICE_CUDA
        
        # 對於Apple Silicon Mac
        elif is_mac() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS可用，使用Apple Silicon加速")
            return DEVICE_MPS
        
        # 對於所有其他平台
        else:
            logger.info("無GPU加速可用，使用CPU")
            return DEVICE_CPU
    
    except ImportError:
        logger.warning("無法導入PyTorch，默認使用CPU")
        return DEVICE_CPU


def get_platform_info() -> Dict[str, Any]:
    """
    獲取當前平台的詳細信息
    
    返回：
        Dict[str, Any]: 包含平台信息的字典
    """
    info = {
        "platform": get_platform(),
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "optimal_device": get_optimal_device()
    }
    
    # 添加Jetson特定信息
    if is_jetson():
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                info["jetson_release"] = f.read().strip()
        except Exception as e:
            info["jetson_release"] = f"Error reading: {str(e)}"
            
        # 嘗試獲取Jetson型號
        try:
            result = subprocess.run(['cat', '/proc/device-tree/model'], 
                                    capture_output=True, text=True, check=False)
            if result.stdout:
                info["jetson_model"] = result.stdout.strip()
        except Exception:
            pass
    
    # 添加PyTorch特定信息
    try:
        import torch
        
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        
        if is_mac() and hasattr(torch.backends, "mps"):
            info["mps_available"] = torch.backends.mps.is_available()
    
    except ImportError:
        info["torch_available"] = False
    
    return info


def print_platform_info() -> None:
    """
    打印格式化的平台信息（用於診斷）
    """
    info = get_platform_info()
    
    print("\n" + "="*50)
    print(f" 系統平台信息")
    print("="*50)
    
    print(f"運行平台: {info['platform']}")
    print(f"操作系統: {info['os']} {info['os_version']}")
    print(f"系統架構: {info['architecture']}")
    print(f"Python版本: {info['python_version']}")
    print(f"最佳計算設備: {info['optimal_device']}")
    
    # PyTorch信息
    if "torch_version" in info:
        print("\nPyTorch信息:")
        print(f"PyTorch版本: {info['torch_version']}")
        print(f"CUDA可用: {info.get('cuda_available', False)}")
        
        if info.get('cuda_available', False):
            print(f"CUDA版本: {info.get('cuda_version', 'unknown')}")
            print(f"GPU設備數量: {info.get('cuda_device_count', 0)}")
            print(f"GPU設備名稱: {info.get('cuda_device_name', 'unknown')}")
        
        if "mps_available" in info:
            print(f"MPS可用: {info['mps_available']}")
    else:
        print("\nPyTorch未安裝或無法導入")
    
    # Jetson特定信息
    if is_jetson():
        print("\nJetson特定信息:")
        if "jetson_model" in info:
            print(f"Jetson型號: {info['jetson_model']}")
        if "jetson_release" in info:
            print(f"JetPack發行版: {info['jetson_release']}")
    
    print("="*50 + "\n")


# 如果作為腳本執行，打印平台診斷信息
if __name__ == "__main__":
    print_platform_info() 