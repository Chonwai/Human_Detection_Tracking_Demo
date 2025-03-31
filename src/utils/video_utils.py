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
        元組 (frame, timestamp)：當前幀和時間戳
    """
    import time
    
    if max_fps:
        frame_time = 1.0 / max_fps
    else:
        frame_time = 0
    
    prev_time = time.time()
    
    while cap.isOpened():
        # 控制幀率
        if max_fps:
            elapsed = time.time() - prev_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
        
        # 讀取下一幀
        ret, frame = cap.read()
        curr_time = time.time()
        
        if not ret:
            break
        
        yield frame, curr_time
        
        prev_time = curr_time
    
    # 釋放資源
    cap.release() 