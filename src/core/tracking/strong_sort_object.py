#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StrongSORT追蹤對象類
用於表示和管理單個追蹤目標，支持增強的運動估計和外觀特徵匹配
"""
import numpy as np
from typing import List, Tuple, Optional
import cv2

class StrongTrack:
    """
    單個追蹤目標表示類
    包含目標位置、非規則形狀參數化、外觀特徵和ID等屬性
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
        # 增強的狀態向量，包含更多參數以支持非規則形狀
        # [x, y, w, h, vx, vy, vw, vh, aspect_ratio]
        self.mean = np.zeros(9)  # 增加aspect_ratio參數
        self.covariance = np.eye(9) * 100.0  # 初始協方差矩陣
        
        # 以中心點和尺寸初始化卡爾曼狀態
        self.mean[:4] = self.xywh
        # 設置初始長寬比
        self.mean[8] = self.xywh[3] / (self.xywh[2] + 1e-6)
        
        # 追蹤狀態管理
        self.id = -1  # 未分配ID
        self.time_since_update = 0  # 自上次更新後經過的幀數
        self.hits = 0  # 連續檢測次數
        self.age = 0  # 總追蹤幀數
        
        # 追蹤歷史軌跡
        self.trajectory = []  # 儲存中心點歷史位置
        self.feature_history = []  # 保存歷史特徵
        self.max_feature_history = 100  # 最多保留的歷史特徵數量
        
        # StrongSORT特有的狀態
        self.ema_alpha = 0.9  # 指數移動平均參數
        self.ema_features = None  # 特徵的指數移動平均
        self.aspect_history = []  # 長寬比歷史
        self.max_aspect_history = 50
        
        # 保存原始特徵
        if feat is not None:
            self.feature_history.append(feat)
            self.ema_features = feat.copy()
        
    def update(self, new_track: 'StrongTrack', frame_id: int) -> None:
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
            # 更新當前特徵
            self.feature = new_track.feature
            
            # 添加到特徵歷史
            self.feature_history.append(new_track.feature)
            if len(self.feature_history) > self.max_feature_history:
                self.feature_history = self.feature_history[-self.max_feature_history:]
            
            # 更新特徵的指數移動平均
            if self.ema_features is None:
                self.ema_features = new_track.feature.copy()
            else:
                self.ema_features = self.ema_alpha * self.ema_features + \
                                  (1 - self.ema_alpha) * new_track.feature
        
        # 更新卡爾曼濾波器
        self._update_kalman(self.xywh)
        
        # 更新長寬比歷史
        aspect_ratio = self.xywh[3] / (self.xywh[2] + 1e-6)
        self.aspect_history.append(aspect_ratio)
        if len(self.aspect_history) > self.max_aspect_history:
            self.aspect_history = self.aspect_history[-self.max_aspect_history:]
        
        # 更新狀態計數器
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
        
        # 更新軌跡
        self._update_trajectory()
    
    def predict(self) -> None:
        """
        預測下一幀的位置
        使用增強的卡爾曼濾波進行狀態預測
        """
        # 增強的卡爾曼預測，考慮長寬比變化
        if self.time_since_update > 0:
            # 考慮阻尼效應，隨著未匹配時間增加而減小速度影響
            damping = 0.7 ** min(self.time_since_update, 3)
            
            # 使用速度分量預測新位置，加入阻尼
            self.mean[0] += self.mean[4] * damping  # x += vx
            self.mean[1] += self.mean[5] * damping  # y += vy
            
            # 寬度和高度考慮速度
            self.mean[2] += self.mean[6] * damping * 0.5  # w += vw
            self.mean[3] += self.mean[7] * damping * 0.5  # h += vh
            
            # 確保寬高為正值
            self.mean[2] = max(10, self.mean[2])  # 最小寬度10像素
            self.mean[3] = max(10, self.mean[3])  # 最小高度10像素
        
        # 更新狀態計數器
        self.time_since_update += 1
        self.age += 1
        
        # 從卡爾曼狀態更新邊界框
        self.xywh = self.mean[:4].copy()
        self.xyxy = self._xywh_to_xyxy(self.xywh)
        
        # 更新軌跡
        self._update_trajectory()
    
    def update_shape(self) -> None:
        """
        更新形狀參數
        StrongSORT特有的非規則形狀參數化 (NSA)
        """
        if len(self.aspect_history) > 2:
            # 計算長寬比的平均值
            mean_aspect = np.mean(self.aspect_history[-10:])
            
            # 更新長寬比狀態
            self.mean[8] = (0.7 * self.mean[8] + 0.3 * mean_aspect)
            
            # 使用長寬比狀態調整邊界框形狀
            # 保持面積不變，調整長寬比
            area = float(self.mean[2] * self.mean[3])  # 確保面積為浮點數
            new_w = np.sqrt(area / self.mean[8])
            new_h = area / new_w
            
            # 平滑過渡
            alpha = 0.3  # 調整比例
            self.mean[2] = (1 - alpha) * self.mean[2] + alpha * new_w
            self.mean[3] = (1 - alpha) * self.mean[3] + alpha * new_h
            
            # 更新邊界框
            self.xywh = self.mean[:4].copy()
            self.xyxy = self._xywh_to_xyxy(self.xywh)
    
    def assign_id(self) -> None:
        """
        分配唯一的追蹤ID
        """
        self.id = StrongTrack._next_id
        StrongTrack._next_id += 1
    
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
    
    def get_feature(self) -> Optional[np.ndarray]:
        """
        獲取特徵向量
        優先返回指數移動平均特徵，提高穩定性
        
        Returns:
            特徵向量或None (如果沒有特徵)
        """
        if self.ema_features is not None:
            return self.ema_features
        elif self.feature is not None:
            return self.feature
        elif len(self.feature_history) > 0:
            # 如果沒有EMA特徵，返回最新的特徵
            return self.feature_history[-1]
        else:
            return None
    
    def _update_kalman(self, measurement: np.ndarray) -> None:
        """
        使用測量值更新卡爾曼濾波器
        StrongSORT使用更穩健的更新方法
        
        Args:
            measurement: 測量值 [x, y, w, h]
        """
        # 增強的卡爾曼更新
        if self.hits == 1:  # 首次檢測，直接設置
            self.mean[:4] = measurement
        else:
            # 計算位置和尺寸變化
            pos_diff = measurement[0:2] - self.mean[0:2]
            size_diff = measurement[2:4] - self.mean[2:4]
            
            # 自適應更新權重
            # 根據檢測質量和時間調整權重
            alpha_pos = min(0.8, max(0.2, 1.0 / (1.0 + self.time_since_update)))
            alpha_size = min(0.7, max(0.1, 1.0 / (1.0 + self.time_since_update)))
            
            # 更新速度估計
            self.mean[4:6] = alpha_pos * pos_diff + (1 - alpha_pos) * self.mean[4:6]
            self.mean[6:8] = alpha_size * size_diff + (1 - alpha_size) * self.mean[6:8]
            
            # 更新位置和尺寸
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