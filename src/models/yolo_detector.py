#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8檢測器實現
基於Ultralytics YOLOv8的人體檢測器
"""
import time
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path

from ultralytics import YOLO
import torch

from core.detector_base import DetectorBase
from config.settings import DETECTION_CONF, DETECTION_IOU, DETECTION_CLASSES, DEVICE, MODEL_DIR, PROJECT_ROOT

class YOLODetector(DetectorBase):
    """
    YOLOv8檢測器類
    實現基於YOLOv8的物體檢測
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",  # 默認使用YOLOv8-nano模型
        conf_threshold: float = DETECTION_CONF,
        iou_threshold: float = DETECTION_IOU,
        classes: List[int] = DETECTION_CLASSES,
        device: str = DEVICE
    ):
        """
        初始化YOLOv8檢測器
        
        Args:
            model_name: 模型名稱或路徑
            conf_threshold: 置信度閾值
            iou_threshold: NMS IOU閾值
            classes: 要檢測的類別ID列表 (COCO格式，0表示人)
            device: 計算設備 ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.device = device
        
        # 模型路徑
        self.model_path = self._get_model_path()
        
        # 模型實例
        self.model = None
        
        # 加載模型
        self.load_model()
    
    def _get_model_path(self) -> Path:
        """
        獲取模型路徑
        如果是預設模型名稱，優先從模型目錄讀取；如果是本地路徑，則使用該路徑
        
        Returns:
            模型路徑
        """
        # 檢查模型目錄中是否已有該模型
        model_dir_path = MODEL_DIR / self.model_name
        if model_dir_path.exists():
            print(f"從模型目錄加載模型: {model_dir_path}")
            return model_dir_path
        
        # 向後兼容：檢查根目錄中是否有該模型
        root_model_path = Path(PROJECT_ROOT) / self.model_name
        if root_model_path.exists():
            print(f"警告: 從項目根目錄加載模型，建議將模型移至 {MODEL_DIR}")
            return root_model_path
        
        # 檢查是否是完整路徑
        if Path(self.model_name).exists():
            return Path(self.model_name)
        
        # 返回模型名，YOLO庫會自動下載
        print(f"未找到本地模型，嘗試使用 YOLO 庫下載模型: {self.model_name}")
        return self.model_name
    
    def load_model(self) -> None:
        """
        加載YOLO模型
        """
        try:
            # 使用Ultralytics YOLO庫加載模型
            self.model = YOLO(self.model_path)
            
            # 設置設備
            if self.device != 'cpu':
                if self.device == 'mps' and torch.backends.mps.is_available():
                    # 對於Apple Silicon，使用MPS
                    self.model.to(self.device)
                elif self.device == 'cuda' and torch.cuda.is_available():
                    # 對於NVIDIA GPU，使用CUDA
                    self.model.to(self.device)
                else:
                    # 如果指定設備不可用，回退到CPU
                    self.device = 'cpu'
                    print(f"警告: {self.device} 不可用，使用CPU替代。")
            
            print(f"成功加載模型 {self.model_name} 到 {self.device} 設備")
        except Exception as e:
            print(f"加載模型時出錯: {e}")
            raise
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        圖像預處理
        這裡簡單實現，因為YOLOv8庫會自動處理大部分預處理
        
        Args:
            frame: 輸入圖像
            
        Returns:
            處理後的圖像
        """
        # 如果是BGR格式（OpenCV默認），轉換為RGB
        if frame.shape[2] == 3:
            # 檢查是否需要BGR到RGB轉換
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            processed_frame = frame
            
        return processed_frame
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        執行檢測
        
        Args:
            frame: 輸入圖像
            
        Returns:
            檢測結果列表和原始圖像
        """
        start_time = time.time()
        
        # 預處理（實際上YOLOv8庫內部會處理）
        processed_frame = self.preprocess(frame)
        
        # 使用YOLOv8進行推理
        results = self.model.predict(
            source=processed_frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False  # 禁用冗長輸出
        )
        
        # 解析結果
        detections = []
        
        if results and len(results) > 0:
            # 獲取第一幀結果
            result = results[0]
            
            # 獲取邊界框、置信度和類別
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2格式
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            # 封裝檢測結果
            for i, box in enumerate(boxes):
                detection = {
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(confs[i]),
                    'class_id': int(classes[i]),
                    'class_name': result.names[int(classes[i])],
                    'time': time.time() - start_time
                }
                detections.append(detection)
        
        return detections, frame
    
    def postprocess(self, 
                   detections: List[Dict[str, Any]], 
                   frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        後處理檢測結果
        在這個簡單實現中，僅將結果傳遞給可視化函數
        
        Args:
            detections: 檢測結果列表
            frame: 原始圖像
            
        Returns:
            處理後的檢測結果和標註後的圖像
        """
        # 在實際項目中，這裡可以添加更多後處理邏輯
        # 例如，過濾特定區域的檢測結果，合併重疊檢測等
        
        # 創建結果圖像
        result_frame = frame.copy()
        
        # 繪製檢測框
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # 檢測框
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 置信度和類別
            label = det['class_name'] + ": " + str(round(det['confidence'], 2))
            cv2.putText(
                result_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        return detections, result_frame 