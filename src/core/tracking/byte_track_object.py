#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ByteTrack追蹤對象類
用於表示和管理單個追蹤目標
"""
import numpy as np
from typing import List, Tuple, Optional
import cv2

class STrack:
    """
    單個追蹤目標表示類
    包含目標位置、運動狀態和ID等屬性
    """
    _next_id = 1  # 類變量，用於分配唯一ID
    
    def __init__(self, bbox: np.ndarray, score: float, cls: int, feat: Optional[np.ndarray] = None):
        """
        初始化追蹤對象
        
        Args:
            bbox: 邊界框 [x1, y1, x2, y2]
            score: 檢測置信度
            cls: 類別ID
            feat: 外觀特徵向量（可選）
        """
        # 轉換為中心點-尺寸格式以便卡爾曼濾波
        # [center_x, center_y, width, height]
        self.xyxy = bbox
        self.xywh = self._xyxy_to_xywh(bbox)
        self.score = score
        self.cls = cls
        self.feature = feat
        
        # 卡爾曼濾波器狀態
        self.mean = np.zeros(8)  # [x, y, w, h, vx, vy, vw, vh]
        self.covariance = np.eye(8) * 100.0  # 初始協方差矩陣
        
        # 以中心點和尺寸初始化卡爾曼狀態
        self.mean[:4] = self.xywh
        
        # 追蹤狀態管理
        self.id = -1  # 未分配ID
        self.time_since_update = 0  # 自上次更新後經過的幀數
        self.hits = 0  # 連續檢測次數
        self.age = 0  # 總追蹤幀數
        
        # 追蹤歷史軌跡
        self.trajectory = []  # 儲存中心點歷史位置
        
    def update(self, new_track: 'STrack', frame_id: int) -> None:
        """
        用新的檢測結果更新追蹤對象
        
        Args:
            new_track: 新的追蹤對象
            frame_id: 當前幀ID
        """
        self.xyxy = new_track.xyxy
        self.xywh = new_track.xywh
        self.score = new_track.score
        
        # 更新卡爾曼濾波器
        self._update_kalman(self.xywh)
        
        # 更新狀態計數器
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
        
        # 更新軌跡
        self._update_trajectory()
    
    def predict(self) -> None:
        """
        預測下一幀的位置
        使用卡爾曼濾波進行狀態預測
        """
        # 簡化的卡爾曼預測
        # 實際系統中應使用完整的卡爾曼濾波器
        if self.time_since_update > 0:
            # 使用速度分量預測新位置
            self.mean[0] += self.mean[4]  # x += vx
            self.mean[1] += self.mean[5]  # y += vy
            
        # 更新卡爾曼濾波器狀態
        self.time_since_update += 1
        self.age += 1
        
        # 從卡爾曼狀態更新邊界框
        self.xywh = self.mean[:4].copy()
        self.xyxy = self._xywh_to_xyxy(self.xywh)
        
        # 更新軌跡
        self._update_trajectory()
    
    def assign_id(self) -> None:
        """
        分配唯一的追蹤ID
        """
        self.id = STrack._next_id
        STrack._next_id += 1
    
    def _update_trajectory(self) -> None:
        """
        更新軌跡記錄
        添加當前中心點位置到軌跡歷史
        """
        center_x = (self.xyxy[0] + self.xyxy[2]) / 2
        center_y = (self.xyxy[1] + self.xyxy[3]) / 2
        self.trajectory.append((int(center_x), int(center_y)))
        
        # 最多保留30個歷史位置
        if len(self.trajectory) > 30:
            self.trajectory = self.trajectory[-30:]
    
    def _update_kalman(self, measurement: np.ndarray) -> None:
        """
        使用測量值更新卡爾曼濾波器
        
        Args:
            measurement: 測量值 [x, y, w, h]
        """
        # 簡化的卡爾曼更新
        # 實際系統中應使用完整的卡爾曼濾波器
        if self.hits == 1:  # 首次檢測，直接設置
            self.mean[:4] = measurement
        else:
            # 計算速度（當前位置 - 上一位置）
            self.mean[4:6] = (measurement[0:2] - self.mean[0:2])
            
            # 更新位置
            self.mean[:4] = measurement
    
    def _xyxy_to_xywh(self, bbox: np.ndarray) -> np.ndarray:
        """
        將xyxy格式轉換為xywh格式
        
        Args:
            bbox: [x1, y1, x2, y2] 格式的邊界框
            
        Returns:
            [center_x, center_y, width, height] 格式的邊界框
        """
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return np.array([center_x, center_y, width, height])
    
    def _xywh_to_xyxy(self, bbox: np.ndarray) -> np.ndarray:
        """
        將xywh格式轉換為xyxy格式
        
        Args:
            bbox: [center_x, center_y, width, height] 格式的邊界框
            
        Returns:
            [x1, y1, x2, y2] 格式的邊界框
        """
        x1 = bbox[0] - bbox[2] / 2
        y1 = bbox[1] - bbox[3] / 2
        x2 = bbox[0] + bbox[2] / 2
        y2 = bbox[1] + bbox[3] / 2
        return np.array([x1, y1, x2, y2]) 