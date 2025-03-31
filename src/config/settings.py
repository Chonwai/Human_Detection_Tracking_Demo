#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
系統配置文件
包含所有可配置的參數和常量
"""
from pathlib import Path
import platform

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

# 平台特定配置
PLATFORM = platform.system()
if PLATFORM == "Darwin":  # MacOS
    DEVICE = "mps"  # Apple Metal
    ENABLE_TENSORRT = False
elif PLATFORM == "Linux":  # Jetson
    DEVICE = "cuda"
    ENABLE_TENSORRT = True
else:
    DEVICE = "cpu"
    ENABLE_TENSORRT = False

# 創建必要的目錄
for dir_path in [VIDEO_DIR, MODEL_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True) 