#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
追蹤器基類
定義所有追蹤器必須實現的基本接口
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class TrackerBase(ABC):
    """
    追蹤器基類
    所有追蹤器實現必須繼承此類並實現所需的方法
    """
    
    @abstractmethod
    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        更新追蹤器狀態
        
        Args:
            detections: 檢測結果列表，每個元素是一個包含'bbox'、'confidence'等鍵的字典
            frame: 當前幀圖像
            
        Returns:
            更新後的追蹤結果列表，每個元素是一個包含'track_id'、'bbox'等鍵的字典
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        重置追蹤器狀態
        通常在處理新視頻或重新開始追蹤時調用
        """
        pass 