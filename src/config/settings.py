#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
系統配置文件
包含所有可配置的參數和常量
"""
from pathlib import Path
import os
import logging

# 導入平台識別工具
from core.platform_utils import (
    get_platform, 
    get_optimal_device, 
    is_jetson, 
    is_mac,
    PLATFORM_MAC,
    PLATFORM_JETSON,
    DEVICE_CPU, 
    DEVICE_CUDA, 
    DEVICE_MPS
)

# 設置日誌
logger = logging.getLogger("Settings")

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

# 系統配置 - 預設值（可能被平台特定設置覆蓋）
MAX_FPS = 30  # 最大FPS限制
PROCESS_WIDTH = 640  # 處理寬度
PROCESS_HEIGHT = 480  # 處理高度

# 獲取平台
PLATFORM = get_platform()
logger.info(f"檢測到平台: {PLATFORM}")

# 檢測設備
DEVICE = get_optimal_device()
logger.info(f"選擇計算設備: {DEVICE}")

# 平台特定配置
# 1. Jetson平台優化
if is_jetson():
    logger.info("應用Jetson平台特定設置")
    # Jetson平台特定設置
    ENABLE_TENSORRT = True
    # 如果可用，優先使用較小的模型
    DEFAULT_MODEL = "yolov10n.pt"
    # 處理分辨率降低以提高性能
    PROCESS_WIDTH = 512
    PROCESS_HEIGHT = 384
    # Jetson上降低默認FPS以避免過載
    MAX_FPS = 15
    # Jetson優化選項 - 可通過UI覆蓋
    USE_HALF_PRECISION = True
    BATCH_SIZE = 1
    
# 2. Mac平台優化
elif is_mac():
    logger.info("應用Mac平台特定設置")
    ENABLE_TENSORRT = False
    DEFAULT_MODEL = "yolov10m.pt"  # Mac上可以使用中等大小的模型
    # 對於Apple Silicon可以使用較高的處理分辨率
    if DEVICE == DEVICE_MPS:
        logger.info("檢測到Apple Silicon，啟用MPS加速")
        PROCESS_WIDTH = 720
        PROCESS_HEIGHT = 540
    
# 3. 其他平台(Windows/Linux桌面)
else:
    logger.info("應用標準桌面平台設置")
    ENABLE_TENSORRT = False
    DEFAULT_MODEL = "yolov10m.pt"

# 記錄當前配置
logger.info(f"處理分辨率: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
logger.info(f"最大FPS: {MAX_FPS}")
logger.info(f"默認模型: {DEFAULT_MODEL}")
logger.info(f"TensorRT加速: {'啟用' if ENABLE_TENSORRT else '禁用'}")

# 創建必要的目錄
for dir_path in [VIDEO_DIR, MODEL_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True) 