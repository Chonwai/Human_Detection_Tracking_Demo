#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
檢測器基類
定義檢測器的基本接口
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict, Any

class DetectorBase(ABC):
    """
    檢測器基類，定義所有檢測器必須實現的接口
    """
    
    @abstractmethod
    def load_model(self) -> None:
        """
        加載模型
        """
        pass
    
    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        圖像預處理
        
        Args:
            frame: 輸入圖像
            
        Returns:
            處理後的圖像
        """
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        執行檢測
        
        Args:
            frame: 輸入圖像
            
        Returns:
            檢測結果列表和處理後的圖像
        """
        pass
    
    @abstractmethod
    def postprocess(self, 
                   detections: List[Dict[str, Any]], 
                   frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        後處理檢測結果
        
        Args:
            detections: 檢測結果列表
            frame: 原始圖像
            
        Returns:
            處理後的檢測結果和標註後的圖像
        """
        pass 