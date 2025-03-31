#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
圖像增強工具函數
提供各種圖像預處理和增強方法，用於提高檢測性能
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

def apply_histogram_equalization(frame: np.ndarray, method: str = 'clahe', clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    對圖像應用直方圖均衡化
    
    Args:
        frame: 輸入圖像 (BGR格式)
        method: 均衡化方法，可選 'basic'（基本均衡化）或 'clahe'（對比度受限自適應直方圖均衡化）
        clip_limit: CLAHE的對比度限制，較高的值給出更強的對比度
        tile_size: CLAHE的網格大小，較小的值處理更局部的細節
    
    Returns:
        增強後的圖像
    """
    # 檢查輸入圖像有效性
    if frame is None or frame.size == 0:
        return frame
    
    # 檢查是否為彩色圖像
    is_color = len(frame.shape) == 3 and frame.shape[2] == 3
    
    # 創建輸出圖像的副本
    enhanced_frame = frame.copy()
    
    if method == 'basic':
        # 基本直方圖均衡化
        if is_color:
            # 轉換到 LAB 色彩空間 (僅對亮度通道進行均衡化)
            lab = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 對亮度通道應用直方圖均衡化
            l_eq = cv2.equalizeHist(l)
            
            # 合併通道
            lab_eq = cv2.merge([l_eq, a, b])
            
            # 轉換回 BGR 色彩空間
            enhanced_frame = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        else:
            # 灰度圖像直接應用均衡化
            enhanced_frame = cv2.equalizeHist(enhanced_frame)
    
    elif method == 'clahe':
        # 對比度受限自適應直方圖均衡化 (CLAHE)
        # 對於舞蹈表演場景，CLAHE通常效果更好，因為它能在保持局部對比度的同時減少噪聲放大
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        
        if is_color:
            # 轉換到 LAB 色彩空間
            lab = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 對亮度通道應用 CLAHE
            l_clahe = clahe.apply(l)
            
            # 合併通道
            lab_clahe = cv2.merge([l_clahe, a, b])
            
            # 轉換回 BGR 色彩空間
            enhanced_frame = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            # 灰度圖像直接應用 CLAHE
            enhanced_frame = clahe.apply(enhanced_frame)
    
    return enhanced_frame

def apply_gamma_correction(frame: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    應用伽瑪校正調整圖像亮度
    
    Args:
        frame: 輸入圖像
        gamma: 伽瑪值，小於1使圖像變亮，大於1使圖像變暗
    
    Returns:
        校正後的圖像
    """
    # 防止除以零
    if gamma <= 0:
        gamma = 0.01
    
    # 創建查找表
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    # 應用伽瑪校正
    return cv2.LUT(frame, table)

def enhance_image(frame: np.ndarray, methods: List[Dict[str, Any]] = None) -> np.ndarray:
    """
    應用多種圖像增強方法
    
    Args:
        frame: 輸入圖像
        methods: 要應用的方法列表，例如 [{'name': 'histogram_equalization', 'params': {'method': 'clahe'}}]
    
    Returns:
        增強後的圖像
    """
    if methods is None:
        # 默認使用 CLAHE 直方圖均衡化
        methods = [{'name': 'histogram_equalization', 'params': {'method': 'clahe'}}]
    
    enhanced_frame = frame.copy()
    
    for method in methods:
        name = method.get('name', '')
        params = method.get('params', {})
        
        if name == 'histogram_equalization':
            eq_method = params.get('method', 'clahe')
            clip_limit = params.get('clip_limit', 2.0)
            tile_size = params.get('tile_size', (8, 8))
            enhanced_frame = apply_histogram_equalization(enhanced_frame, eq_method, clip_limit, tile_size)
        
        elif name == 'gamma_correction':
            gamma = params.get('gamma', 1.0)
            enhanced_frame = apply_gamma_correction(enhanced_frame, gamma)
    
    return enhanced_frame 