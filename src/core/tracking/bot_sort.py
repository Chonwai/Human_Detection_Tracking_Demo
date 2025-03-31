#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BoT-SORT追蹤器實現
結合ByteTrack的關聯策略和視覺外觀特徵匹配的多目標追蹤器
"""
import numpy as np
import cv2  # 添加缺少的OpenCV導入
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import linear_sum_assignment

from core.tracker_base import TrackerBase
from core.tracking.bot_sort_object import BoTTrack
from config.settings import TRACKING_MAX_AGE, TRACKING_MIN_HITS, TRACKING_IOU_THRESH

class BoTSort(TrackerBase):
    """
    BoT-SORT追蹤器
    結合運動預測、視覺外觀特徵和攝像機運動補償的多目標追蹤器
    
    論文參考：BoT-SORT: Robust Associations Multi-Pedestrian Tracking
    """
    
    def __init__(
        self,
        max_age: int = TRACKING_MAX_AGE,
        min_hits: int = 1,
        iou_threshold: float = TRACKING_IOU_THRESH,
        high_threshold: float = 0.6,  # 較高的高置信度閾值
        low_threshold: float = 0.1,   # 相同的低置信度閾值
        match_threshold: float = 0.7, # ReID特徵匹配閾值
        use_appearance: bool = True,  # 是否使用外觀特徵
        use_cmc: bool = True,         # 是否使用攝像機運動補償
    ):
        """
        初始化BoT-SORT追蹤器
        
        Args:
            max_age: 最大追蹤年齡，超過此值的軌跡將被刪除
            min_hits: 最小檢測次數，達到此值後軌跡才會被激活
            iou_threshold: IoU匹配閾值
            high_threshold: 高置信度閾值，用於第一階段匹配
            low_threshold: 低置信度閾值，用於第二階段匹配
            match_threshold: ReID特徵匹配閾值
            use_appearance: 是否使用外觀特徵進行匹配
            use_cmc: 是否使用攝像機運動補償
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.use_appearance = use_appearance
        self.use_cmc = use_cmc
        
        # 追蹤狀態
        self.frame_id = 0
        self.tracks = []          # 活躍的追蹤軌跡
        self.lost_tracks = []     # 暫時丟失的軌跡
        self.removed_tracks = []  # 已刪除的軌跡
        
        # 特徵提取器 (簡化版，實際使用時應加載預訓練模型)
        self.feature_extractor = None
        
        # 上一幀的關鍵點 (用於攝像機運動補償)
        self.prev_keypoints = None
        self.prev_descriptors = None
        
    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        使用新的檢測結果更新追蹤器
        
        Args:
            detections: 檢測結果列表
            frame: 當前幀圖像
            
        Returns:
            追蹤結果列表，每個元素是一個包含'track_id'、'bbox'等鍵的字典
        """
        self.frame_id += 1
        
        # 攝像機運動補償 (如果啟用)
        if self.use_cmc and frame is not None:
            self._apply_camera_motion_compensation(frame)
        
        # 轉換檢測結果為BoTTrack對象
        detection_tracks = []
        for det in detections:
            bbox = np.array(det['bbox'])  # [x1, y1, x2, y2]
            score = det['confidence']
            cls = det.get('class_id', 0)  # 預設為人類 (class_id=0)
            
            # 提取外觀特徵 (如果啟用)
            feat = None
            if self.use_appearance and frame is not None:
                feat = self._extract_features(frame, bbox)
            
            # 創建BoTTrack對象
            bot_track = BoTTrack(bbox, score, cls, feat)
            detection_tracks.append(bot_track)
        
        # 分為高置信度和低置信度檢測
        high_score_tracks = [t for t in detection_tracks if t.score >= self.high_threshold]
        low_score_tracks = [t for t in detection_tracks if t.score < self.high_threshold and t.score >= self.low_threshold]
        
        # 活躍軌跡預測
        tracked_tracks = []
        for track in self.tracks:
            track.predict()
            tracked_tracks.append(track)
        
        # 第一階段關聯：高置信度檢測與活躍軌跡匹配
        matches_a, unmatched_tracks_a, unmatched_detections_a = \
            self._associate(tracked_tracks, high_score_tracks, frame)
        
        # 更新匹配的軌跡
        for track_idx, det_idx in matches_a:
            self.tracks[track_idx].update(high_score_tracks[det_idx], self.frame_id)
        
        # 第二階段關聯：未匹配的軌跡與低置信度檢測匹配
        r_tracked_tracks = [self.tracks[i] for i in unmatched_tracks_a]
        matches_b, unmatched_tracks_b, unmatched_detections_b = \
            self._associate(r_tracked_tracks, low_score_tracks, frame, use_appearance=False)  # 低置信度不使用外觀特徵
        
        # 更新匹配的軌跡
        for track_idx, det_idx in matches_b:
            r_tracked_tracks[track_idx].update(low_score_tracks[det_idx], self.frame_id)
        
        # 處理未匹配的軌跡：加入丟失列表
        for i in unmatched_tracks_b:
            track = r_tracked_tracks[i]
            if track.time_since_update <= self.max_age:
                self.lost_tracks.append(track)
        
        # 處理未匹配的檢測：創建新軌跡
        for i in unmatched_detections_a:
            track = high_score_tracks[i]
            if track.score >= self.high_threshold:  # 只為高置信度檢測創建新軌跡
                track.assign_id()  # 分配新ID
                self.tracks.append(track)
        
        # 第三階段關聯：嘗試恢復丟失的軌跡
        # 這是BoT-SORT的一個關鍵改進
        if len(self.lost_tracks) > 0 and len(unmatched_detections_a) > 0:
            r_detections = [high_score_tracks[i] for i in unmatched_detections_a]
            matches_c, _, _ = self._associate(
                self.lost_tracks, r_detections, frame,
                iou_threshold=0.3,  # 對丟失軌跡使用較低的IoU閾值
                use_appearance=self.use_appearance,
                appearance_weight=0.7  # 丟失軌跡恢復時增加外觀特徵的權重
            )
            
            # 恢復匹配的軌跡
            for lost_idx, det_idx in matches_c:
                lost_track = self.lost_tracks[lost_idx]
                det_track = r_detections[det_idx]
                lost_track.update(det_track, self.frame_id)
                
                # 從丟失列表移回活躍列表
                self.tracks.append(lost_track)
                self.lost_tracks[lost_idx] = None
            
            # 清理已恢復的軌跡
            self.lost_tracks = [t for t in self.lost_tracks if t is not None]
        
        # 更新丟失軌跡
        lost_idx = []
        for i, track in enumerate(self.lost_tracks):
            track.predict()
            if track.time_since_update > self.max_age:
                lost_idx.append(i)
                self.removed_tracks.append(track)
        
        # 刪除過期的丟失軌跡
        for idx in reversed(lost_idx):
            self.lost_tracks.pop(idx)
        
        # 整理返回結果
        tracked_results = []
        for track in self.tracks:
            if track.time_since_update == 0 and (track.hits >= self.min_hits or self.frame_id <= self.min_hits):
                result = {
                    'track_id': track.id,
                    'bbox': track.xyxy.tolist(),
                    'confidence': track.score,
                    'class_id': track.cls,
                    'trajectory': track.trajectory
                }
                tracked_results.append(result)
        
        return tracked_results
    
    def reset(self) -> None:
        """
        重置追蹤器狀態
        """
        self.frame_id = 0
        self.tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        BoTTrack._next_id = 1  # 重置ID計數器
        
        # 重置攝像機運動補償狀態
        self.prev_keypoints = None
        self.prev_descriptors = None
    
    def _associate(
        self, 
        tracks: List[BoTTrack], 
        detections: List[BoTTrack],
        frame: Optional[np.ndarray] = None,
        iou_threshold: Optional[float] = None,
        use_appearance: Optional[bool] = None,
        appearance_weight: float = 0.5
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        將檢測結果與現有軌跡進行關聯
        
        Args:
            tracks: 現有軌跡列表
            detections: 檢測結果列表
            frame: 當前幀圖像，用於特徵提取
            iou_threshold: 可選的IoU閾值，不指定則使用類屬性
            use_appearance: 是否使用外觀特徵，不指定則使用類屬性
            appearance_weight: 外觀特徵的權重 (0~1之間)
            
        Returns:
            matches: 匹配的軌跡和檢測索引對 [(track_idx, det_idx), ...]
            unmatched_tracks: 未匹配的軌跡索引列表
            unmatched_detections: 未匹配的檢測索引列表
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
            
        if use_appearance is None:
            use_appearance = self.use_appearance
        
        # 計算IoU矩陣
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t, track in enumerate(tracks):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self._iou(track.xyxy, detection.xyxy)
        
        # 如果啟用外觀特徵，則計算外觀相似度矩陣並結合
        if use_appearance:
            appearance_matrix = np.zeros_like(iou_matrix)
            valid_appearance = False
            
            for t, track in enumerate(tracks):
                if track.feature is not None:
                    for d, detection in enumerate(detections):
                        if detection.feature is not None:
                            # 計算特徵向量的餘弦相似度
                            appearance_matrix[t, d] = self._cosine_similarity(track.feature, detection.feature)
                            valid_appearance = True
            
            # 如果有有效的外觀特徵，融合兩個矩陣
            if valid_appearance:
                # 融合IoU和外觀特徵
                matrix = iou_matrix * (1 - appearance_weight) + appearance_matrix * appearance_weight
            else:
                matrix = iou_matrix
        else:
            matrix = iou_matrix
        
        # 使用匈牙利算法進行最優匹配
        if min(matrix.shape) > 0:
            # 將相似度轉換為代價矩陣 (1-相似度)
            cost_matrix = 1.0 - matrix
            
            # 使用scipy的匈牙利算法求解
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
            
            # 根據閾值過濾匹配結果
            matches = []
            for t, d in zip(track_indices, detection_indices):
                if matrix[t, d] >= iou_threshold:  # 使用融合矩陣或IoU矩陣判斷
                    matches.append((t, d))
                else:
                    track_indices = np.append(track_indices, t)
                    detection_indices = np.append(detection_indices, d)
            
            # 未匹配的軌跡和檢測
            unmatched_tracks = list(set(range(len(tracks))) - set(track_indices))
            unmatched_detections = list(set(range(len(detections))) - set(detection_indices))
        else:
            # 如果沒有軌跡或檢測，則所有都是未匹配
            matches = []
            unmatched_tracks = list(range(len(tracks)))
            unmatched_detections = list(range(len(detections)))
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _apply_camera_motion_compensation(self, frame: np.ndarray) -> None:
        """
        應用攝像機運動補償
        對於移動攝像機場景，補償攝像機運動對目標軌跡的影響
        
        Args:
            frame: 當前幀圖像
        """
        if frame is None:
            return
            
        # 轉換為灰度圖
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 檢測特徵點
        # 使用ORB特徵提取器，速度較快
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # 如果是第一幀或上一幀沒有特徵點，則保存並返回
        if self.prev_keypoints is None or self.prev_descriptors is None:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return
            
        # 如果特徵點太少，則跳過補償
        if len(keypoints) < 10 or len(self.prev_keypoints) < 10:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return
            
        # 特徵匹配
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(self.prev_descriptors, descriptors)
        
        # 如果匹配點太少，則跳過補償
        if len(matches) < 10:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return
            
        # 獲取匹配點坐標
        prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches])
        
        # 計算變換矩陣 (僅考慮平移和旋轉)
        transform_matrix, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        
        if transform_matrix is None:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return
            
        # 應用變換矩陣到軌跡
        # 對活躍軌跡的最近位置進行補償
        for track in self.tracks:
            if track.trajectory:
                # 獲取最後一個軌跡點
                x, y = track.trajectory[-1]
                
                # 變換坐標
                pt = np.array([x, y, 1])  # 齊次坐標
                new_pt = np.dot(transform_matrix, pt)
                
                # 更新軌跡點
                track.trajectory[-1] = (int(new_pt[0]), int(new_pt[1]))
                
                # 更新邊界框
                center_x = (track.xyxy[0] + track.xyxy[2]) / 2
                center_y = (track.xyxy[1] + track.xyxy[3]) / 2
                width = track.xyxy[2] - track.xyxy[0]
                height = track.xyxy[3] - track.xyxy[1]
                
                new_center = np.dot(transform_matrix, np.array([center_x, center_y, 1]))
                
                # 更新邊界框
                track.xyxy[0] = new_center[0] - width / 2
                track.xyxy[1] = new_center[1] - height / 2
                track.xyxy[2] = new_center[0] + width / 2
                track.xyxy[3] = new_center[1] + height / 2
                
                # 更新卡爾曼濾波器狀態
                track.xywh = track._xyxy_to_xywh(track.xyxy)
                track.mean[:4] = track.xywh
        
        # 更新上一幀的特徵點和描述符
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
    
    def _extract_features(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        提取邊界框區域的視覺特徵
        
        Args:
            frame: 完整幀圖像
            bbox: 邊界框 [x1, y1, x2, y2]
            
        Returns:
            特徵向量 (如果特徵提取失敗則返回None)
        """
        # 簡易實現，在實際應用中應使用預訓練的ReID模型
        try:
            # 裁剪邊界框區域
            x1, y1, x2, y2 = bbox.astype(int)
            # 防止越界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            crop = frame[y1:y2, x1:x2]
            
            # 調整大小並標準化
            resized = cv2.resize(crop, (64, 128))
            
            # 簡易特徵提取（實際中應使用深度學習模型）
            # 這裡僅作為示範，使用顏色直方圖作為特徵
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            
            # 連接直方圖特徵
            feature = np.concatenate([h_hist, s_hist, v_hist]).flatten()
            # 標準化
            feature = feature / (np.sum(feature) + 1e-10)
            
            return feature
        except Exception as e:
            # print(f"特徵提取錯誤: {e}")
            return None
    
    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        計算兩個特徵向量的餘弦相似度
        
        Args:
            feat1: 第一個特徵向量
            feat2: 第二個特徵向量
            
        Returns:
            餘弦相似度 (0~1之間)
        """
        # 防止除以零
        norm1 = np.linalg.norm(feat1) + 1e-10
        norm2 = np.linalg.norm(feat2) + 1e-10
        
        # 計算餘弦相似度
        return np.dot(feat1, feat2) / (norm1 * norm2)
    
    def _iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        計算兩個邊界框的IoU
        
        Args:
            bbox1: 第一個邊界框 [x1, y1, x2, y2]
            bbox2: 第二個邊界框 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        # 計算相交區域
        xx1 = max(bbox1[0], bbox2[0])
        yy1 = max(bbox1[1], bbox2[1])
        xx2 = min(bbox1[2], bbox2[2])
        yy2 = min(bbox1[3], bbox2[3])
        
        # 檢查是否有重疊
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        inter_area = w * h
        
        # 計算各自面積
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # 計算IoU
        iou = inter_area / (area1 + area2 - inter_area + 1e-6)
        return float(iou) 