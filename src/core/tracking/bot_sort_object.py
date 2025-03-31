#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BoT-SORT追蹤對象類
用於表示和管理單個追蹤目標，支持外觀特徵和運動預測
"""
import numpy as np
from typing import List, Tuple, Optional
import cv2

class BoTTrack:
    """
    單個追蹤目標表示類
    包含目標位置、運動狀態、外觀特徵和ID等屬性
    """
    _next_id = 1  # 類變量，用於分配唯一ID
    
    def __init__(self, bbox: np.ndarray, score: float, cls: int, feat: Optional[np.ndarray] = None):
        """
        初始化追蹤對象
        
        Args:
            bbox: 邊界框 [x1, y1, x2, y2]
            score: 檢測置信度
            cls: 類別ID
            feat: 外觀特徵向量
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
        
        # 增強卡爾曼參數 (BoT-SORT的擴展)
        # 使用更複雜的運動模型
        self.velocity_direction = None  # 速度方向
        self.velocity_magnitude = 0     # 速度大小
        
        # 特徵歷史
        self.feature_history = []  # 保存歷史特徵以提高匹配穩定性
        self.max_feature_history = 10  # 最多保留的歷史特徵數量
        
        # 添加特徵
        if feat is not None:
            self.feature_history.append(feat)
        
    def update(self, new_track: 'BoTTrack', frame_id: int) -> None:
        """
        用新的檢測結果更新追蹤對象
        
        Args:
            new_track: 新的追蹤對象
            frame_id: 當前幀ID
        """
        self.xyxy = new_track.xyxy
        self.xywh = new_track.xywh
        self.score = new_track.score
        
        # 更新特徵
        if new_track.feature is not None:
            self.feature = new_track.feature
            self.feature_history.append(new_track.feature)
            # 限制歷史特徵數量
            if len(self.feature_history) > self.max_feature_history:
                self.feature_history = self.feature_history[-self.max_feature_history:]
        
        # 更新卡爾曼濾波器
        self._update_kalman(self.xywh)
        
        # 更新狀態計數器
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
        
        # 更新軌跡
        self._update_trajectory()
        
        # 更新速度方向和大小
        self._update_velocity()
    
    def predict(self) -> None:
        """
        預測下一幀的位置
        使用增強的卡爾曼濾波進行狀態預測
        """
        # 增強的卡爾曼預測
        if self.time_since_update > 0:
            # 考慮加速度和阻尼效應
            # 在長期未匹配的情況下降低速度預測的權重
            damping = 0.8 ** min(self.time_since_update, 5)
            
            # 使用速度分量預測新位置
            self.mean[0] += self.mean[4] * damping  # x += vx
            self.mean[1] += self.mean[5] * damping  # y += vy
            
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
        self.id = BoTTrack._next_id
        BoTTrack._next_id += 1
    
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
    
    def _update_velocity(self) -> None:
        """
        更新速度方向和大小
        用於改進預測和匹配
        """
        if len(self.trajectory) >= 2:
            # 計算最近兩個點的速度
            curr_x, curr_y = self.trajectory[-1]
            prev_x, prev_y = self.trajectory[-2]
            
            # 確保使用浮點數計算
            dx = float(curr_x - prev_x)
            dy = float(curr_y - prev_y)
            
            # 計算速度大小
            self.velocity_magnitude = np.sqrt(dx**2 + dy**2)
            
            # 計算速度方向（弧度）
            if self.velocity_magnitude > 0:
                self.velocity_direction = np.arctan2(dy, dx)
    
    def get_feature(self) -> Optional[np.ndarray]:
        """
        獲取最新的特徵向量
        
        Returns:
            特徵向量或None (如果沒有特徵)
        """
        return self.feature
    
    def get_merged_feature(self) -> Optional[np.ndarray]:
        """
        獲取合併的歷史特徵
        通過平均所有歷史特徵來提高穩定性
        
        Returns:
            合併的特徵向量或None (如果沒有特徵)
        """
        if not self.feature_history:
            return None
            
        # 平均所有歷史特徵
        merged = np.mean(self.feature_history, axis=0)
        # 標準化
        norm = np.linalg.norm(merged)
        if norm > 0:
            merged = merged / norm
        
        return merged
    
    def _update_kalman(self, measurement: np.ndarray) -> None:
        """
        使用測量值更新卡爾曼濾波器
        BoT-SORT使用更穩健的卡爾曼更新
        
        Args:
            measurement: 測量值 [x, y, w, h]
        """
        # 增強的卡爾曼更新
        if self.hits == 1:  # 首次檢測，直接設置
            self.mean[:4] = measurement
        else:
            # 計算速度（當前位置 - 上一位置）
            new_velocity = measurement[0:2] - self.mean[0:2]
            
            # 平滑速度更新（考慮歷史速度）
            alpha = 0.6  # 新測量的權重，根據情況調整
            self.mean[4:6] = alpha * new_velocity + (1 - alpha) * self.mean[4:6]
            
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