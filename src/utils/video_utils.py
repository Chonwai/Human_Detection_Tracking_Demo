#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
視頻處理工具函數
用於處理不同來源的視頻輸入
"""
import cv2
import numpy as np
from typing import Union, Tuple, Optional, Generator
from pathlib import Path
import tempfile
import os
import time # 確保 time 模組已導入

def get_video_source(
    source: Union[str, int, Path],
    target_width: Optional[int] = None,
    target_height: Optional[int] = None
) -> cv2.VideoCapture:
    """
    創建視頻捕獲對象
    
    Args:
        source: 視頻源，可以是文件路徑、攝像頭ID或RTSP URL
        target_width: 目標寬度，如果指定則調整視頻寬度
        target_height: 目標高度，如果指定則調整視頻高度
        
    Returns:
        OpenCV VideoCapture對象
    """
    # 處理臨時上傳的文件
    if hasattr(source, 'read'):  # 檢查是否為文件對象
        # 處理Streamlit上傳的文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(source.read())
        temp_file.close()
        source = temp_file.name
    
    # 創建捕獲對象
    cap = cv2.VideoCapture(source)
    
    # 檢查捕獲對象是否成功創建
    if not cap.isOpened():
        raise ValueError(f"無法打開視頻源: {source}")
    
    # 設置分辨率（如果指定）
    if target_width and target_height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    
    return cap

def get_video_info(cap: cv2.VideoCapture) -> dict:
    """
    獲取視頻信息
    
    Args:
        cap: OpenCV VideoCapture對象
        
    Returns:
        包含視頻信息的字典
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count
    }

def video_frame_generator(
    cap: cv2.VideoCapture,
    max_fps: Optional[float] = None
) -> Generator[Tuple[np.ndarray, float], None, None]:
    """
    視頻幀生成器
    
    Args:
        cap: OpenCV VideoCapture對象
        max_fps: 最大FPS限制，如果指定，會控制生成速率
        
    Yields:
        元組 (frame, timestamp)：當前幀和時間戳 (秒，相對於影片開始)
    """
    # import time # time 已經在模組頂部導入
    
    if max_fps and max_fps > 0: # 確保 max_fps > 0 以避免 ZeroDivisionError
        frame_interval_control = 1.0 / max_fps
    else:
        frame_interval_control = 0 # 不進行FPS控制
    
    last_yield_time = time.time() # 用於FPS控制
    
    while cap.isOpened():
        # FPS 控制邏輯 (基於實際處理時間，非影片時間)
        if frame_interval_control > 0:
            current_process_time = time.time()
            elapsed_since_last_yield = current_process_time - last_yield_time
            if elapsed_since_last_yield < frame_interval_control:
                sleep_duration = frame_interval_control - elapsed_since_last_yield
                if sleep_duration > 0: # 避免 time.sleep(負數)
                    time.sleep(sleep_duration)
        
        # 讀取下一幀
        ret, frame = cap.read()
        
        if not ret:
            break # 影片結束或讀取錯誤
        
        # 獲取當前幀在影片中的時間戳 (毫秒)
        video_timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        # 轉換為秒
        video_timestamp_sec = video_timestamp_msec / 1000.0
        
        yield frame, video_timestamp_sec
        
        last_yield_time = time.time() # 更新上次產生的時間，用於FPS控制
    
    # 釋放資源
    cap.release() 