#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
系統配置文件
包含所有可配置的參數和常量
"""
from pathlib import Path
import platform
import os
import torch

# 項目根目錄
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 數據路徑配置
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "output"

# 檢測配置
DETECTION_CONF = 0.5  # 檢測置信度閾值
DETECTION_IOU = 0.45  # NMS IOU閾值
DETECTION_CLASSES = [0]  # 只檢測人類 (COCO類別)

# 追蹤配置
TRACKING_MAX_AGE = 30  # 最大追蹤幀數
TRACKING_MIN_HITS = 3  # 最小確認幀數
TRACKING_IOU_THRESH = 0.3  # 追蹤IOU閾值

# 系統配置
MAX_FPS = 30  # 最大FPS限制
PROCESS_WIDTH = 640  # 處理寬度
PROCESS_HEIGHT = 480  # 處理高度

# 平台檢測增強版 - 更準確地檢測Jetson和其他平台
PLATFORM = platform.system()

# 檢測是否為NVIDIA Jetson平台
IS_JETSON = os.path.exists('/etc/nv_tegra_release')

# 優化的設備選擇邏輯
def get_optimal_device():
    """
    根據可用硬件選擇最佳計算設備
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# 設置設備
DEVICE = get_optimal_device()

# 針對Jetson平台的特殊設置
if IS_JETSON:
    # Jetson平台特定設置
    ENABLE_TENSORRT = True
    # 如果可用，優先使用較小的模型
    DEFAULT_MODEL = "yolov10n.pt"
    # 處理分辨率降低以提高性能
    PROCESS_WIDTH = 512
    PROCESS_HEIGHT = 384
    # Jetson上降低默認FPS以避免過載
    MAX_FPS = 15
elif PLATFORM == "Darwin":  # MacOS
    ENABLE_TENSORRT = False
    DEFAULT_MODEL = "yolov10m.pt"
else:  # 其他平台(Windows/Linux桌面)
    ENABLE_TENSORRT = False
    DEFAULT_MODEL = "yolov10m.pt"

# 記錄當前配置
print(f"平台: {PLATFORM}{' (Jetson)' if IS_JETSON else ''}")
print(f"選擇的計算設備: {DEVICE}")
print(f"TensorRT加速: {'啟用' if ENABLE_TENSORRT else '禁用'}")

# 創建必要的目錄
for dir_path in [VIDEO_DIR, MODEL_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True) 