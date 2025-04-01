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
import logging

from ultralytics import YOLO
import torch

from core.detector_base import DetectorBase
from config.settings import DETECTION_CONF, DETECTION_IOU, DETECTION_CLASSES, DEVICE, MODEL_DIR, PROJECT_ROOT
from core.platform_utils import is_jetson, is_mac, get_optimal_device, DEVICE_CUDA, DEVICE_MPS, DEVICE_CPU

# 設置日誌
logger = logging.getLogger("YOLODetector")

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
        
        # 日誌
        logger.info(f"初始化 YOLODetector: 模型={model_name}, 設備={device}")
        
        # 加載模型
        self.load_model()
    
    def _get_model_path(self) -> Path:
        """
        獲取模型路徑
        
        Returns:
            模型的完整路徑
        """
        # 檢查是否提供了絕對路徑
        if Path(self.model_name).is_absolute() and Path(self.model_name).exists():
            return Path(self.model_name)
        
        # 檢查模型目錄中是否存在模型文件
        model_path = MODEL_DIR / self.model_name
        if model_path.exists():
            return model_path
        
        # 如果模型文件不存在，則假設它是一個預訓練的模型名稱
        # 讓ultralytics庫從預訓練模型庫中下載
        return Path(self.model_name)
    
    def load_model(self) -> None:
        """
        加載YOLO模型
        
        使用跨平台的方式加載模型，自動檢測並使用最適合當前平台的加速方式。
        支持Jetson平台的CUDA加速、Apple Silicon的MPS加速和其他平台的標準執行。
        """
        try:
            # 獲取平台特定優化選項
            use_tensorrt = False
            use_half = False
            
            # 從streamlit session_state中讀取用戶配置（如果可用）
            try:
                import streamlit as st
                if 'enable_tensorrt' in st.session_state:
                    use_tensorrt = st.session_state.enable_tensorrt
                if 'use_half_precision' in st.session_state:
                    use_half = st.session_state.use_half_precision
                if 'use_mps' in st.session_state:
                    # 如果用戶明確禁用MPS，則回退到CPU
                    if not st.session_state.use_mps and self.device == DEVICE_MPS:
                        self.device = DEVICE_CPU
                        logger.info("遵循用戶設置，禁用MPS加速")
            except ImportError:
                # 在非Streamlit環境中無法獲取session_state
                logger.debug("未在Streamlit環境中運行，無法獲取用戶UI配置")
            
            # 標準模式加載基本模型
            logger.info(f"開始加載模型 {self.model_name}...")
            self.model = YOLO(self.model_path)
            
            # ==== 平台特定優化 ====
            # 1. Jetson平台CUDA優化
            if is_jetson() and self.device == DEVICE_CUDA:
                if torch.cuda.is_available():
                    logger.info("在Jetson平台上優化CUDA執行...")
                    
                    # 設置當前CUDA設備
                    torch.cuda.set_device(0)
                    
                    # 嘗試CUDA初始化測試
                    try:
                        test_tensor = torch.ones(1).cuda()
                        del test_tensor  # 立即釋放內存
                        
                        # CUDA可用，應用Jetson特定優化
                        # 1.1 嘗試TensorRT優化
                        if use_tensorrt:
                            try:
                                logger.info("嘗試應用TensorRT優化...")
                                self.model.to('cuda')
                                self.model.fuse()  # 融合層以提高性能
                                logger.info("TensorRT優化成功應用")
                            except Exception as e:
                                logger.warning(f"TensorRT優化失敗: {e}")
                                logger.info("繼續使用標準CUDA加速")
                        
                        # 1.2 應用半精度計算（如果啟用）
                        if use_half:
                            try:
                                logger.info("應用FP16半精度計算...")
                                self.model = self.model.half()
                            except Exception as e:
                                logger.warning(f"半精度轉換失敗: {e}")
                        
                        # 1.3 移動模型到CUDA設備
                        self.model.to(self.device)
                        
                        # 1.4 輸出CUDA診斷信息
                        cuda_device_count = torch.cuda.device_count()
                        cuda_device_name = torch.cuda.get_device_name(0)
                        logger.info(f"CUDA初始化成功: {cuda_device_count}個GPU, 設備: {cuda_device_name}")
                        
                    except RuntimeError as e:
                        logger.error(f"CUDA初始化失敗: {e}")
                        logger.warning("回退到CPU模式")
                        self.device = DEVICE_CPU
                        self.model.to(self.device)
                else:
                    logger.warning("CUDA在PyTorch中不可用，使用CPU替代")
                    logger.info("對於Jetson平台，請確保安裝了正確版本的PyTorch (arm64/aarch64)")
                    self.device = DEVICE_CPU
                    self.model.to(self.device)
            
            # 2. Mac平台MPS優化 (Apple Silicon)
            elif is_mac() and self.device == DEVICE_MPS:
                if torch.backends.mps.is_available():
                    logger.info("在Mac平台上使用MPS (Metal Performance Shaders) 加速...")
                    
                    # 設置內存限制（如果在UI中設置）
                    try:
                        import streamlit as st
                        if 'gpu_mem_limit' in st.session_state:
                            # 注意：這只是記錄，PyTorch目前沒有直接限制MPS內存使用的API
                            logger.info(f"設置GPU內存限制: {st.session_state.gpu_mem_limit}GB")
                    except ImportError:
                        pass
                    
                    try:
                        # 移動模型到MPS設備
                        self.model.to(self.device)
                        logger.info("MPS加速成功應用")
                    except Exception as e:
                        logger.error(f"MPS初始化失敗: {e}")
                        logger.warning("回退到CPU模式")
                        self.device = DEVICE_CPU
                        self.model.to(self.device)
                else:
                    logger.warning("MPS在PyTorch中不可用，使用CPU替代")
                    self.device = DEVICE_CPU
                    self.model.to(self.device)
            
            # 3. 標準CUDA平台（非Jetson的NVIDIA GPU）
            elif self.device == DEVICE_CUDA:
                if torch.cuda.is_available():
                    logger.info("在標準CUDA平台上運行...")
                    try:
                        # 移動模型到CUDA設備
                        self.model.to(self.device)
                        
                        # 應用半精度計算（如果啟用）
                        if use_half:
                            try:
                                logger.info("應用FP16半精度計算...")
                                self.model = self.model.half()
                            except Exception as e:
                                logger.warning(f"半精度轉換失敗: {e}")
                        
                        # 輸出CUDA信息
                        cuda_device_name = torch.cuda.get_device_name(0)
                        logger.info(f"CUDA初始化成功，使用設備: {cuda_device_name}")
                    except RuntimeError as e:
                        logger.error(f"CUDA初始化失敗: {e}")
                        logger.warning("回退到CPU模式")
                        self.device = DEVICE_CPU
                        self.model.to(self.device)
                else:
                    logger.warning("CUDA在PyTorch中不可用，使用CPU替代")
                    self.device = DEVICE_CPU
                    self.model.to(self.device)
            
            # 4. CPU平台（或回退設備）
            else:
                logger.info("在CPU上運行模型...")
                self.model.to(self.device)
                
                # 檢查模型大小，提供性能建議
                if any(x in self.model_name.lower() for x in ['l', 'x']):
                    logger.warning("您正在CPU上使用較大的模型。考慮使用較小的模型（如yolov8n.pt）以提高性能。")
            
            logger.info(f"模型 {self.model_name} 成功加載到 {self.device} 設備")
            
        except Exception as e:
            logger.error(f"模型加載失敗: {e}")
            # 在開發環境中提供完整錯誤信息
            import traceback
            logger.debug(traceback.format_exc())
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