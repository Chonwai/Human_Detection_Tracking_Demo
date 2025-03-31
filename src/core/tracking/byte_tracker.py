#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ByteTrack追蹤器實現
基於ByteTrack算法的多目標追蹤器
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.optimize import linear_sum_assignment

from core.tracker_base import TrackerBase
from core.tracking.byte_track_object import STrack
from config.settings import TRACKING_MAX_AGE, TRACKING_MIN_HITS, TRACKING_IOU_THRESH

class ByteTracker(TrackerBase):
    """
    ByteTrack追蹤器
    實現基於ByteTrack算法的多目標追蹤
    """
    
    def __init__(
        self,
        max_age: int = TRACKING_MAX_AGE,
        min_hits: int = 1,  # 改為默認值1，使新檢測立即成為活躍軌跡
        iou_threshold: float = TRACKING_IOU_THRESH,
        high_threshold: float = 0.3,  # 默認降低高置信度閾值
        low_threshold: float = 0.1    # 默認降低低置信度閾值
    ):
        """
        初始化ByteTrack追蹤器
        
        Args:
            max_age: 最大追蹤年齡，超過此值的軌跡將被刪除
            min_hits: 最小檢測次數，達到此值後軌跡才會被激活
            iou_threshold: IoU匹配閾值
            high_threshold: 高置信度閾值，用於第一階段匹配
            low_threshold: 低置信度閾值，用於第二階段匹配
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_threshold = high_threshold  # 新增
        self.low_threshold = low_threshold    # 新增
        
        # 追蹤狀態
        self.frame_id = 0
        self.tracks = []  # 活躍的追蹤軌跡
        self.lost_tracks = []  # 暫時丟失的軌跡
        self.removed_tracks = []  # 已刪除的軌跡
        
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
        
        # 轉換檢測結果為STrack對象
        detection_tracks = []
        for det in detections:
            bbox = np.array(det['bbox'])  # [x1, y1, x2, y2]
            score = det['confidence']
            cls = det.get('class_id', 0)  # 預設為人類 (class_id=0)
            
            # 創建STrack對象
            strack = STrack(bbox, score, cls)
            detection_tracks.append(strack)
        
        # 分為高置信度和低置信度檢測
        # 降低高置信度閾值，確保更多檢測能進入第一階段匹配
        high_score_tracks = [t for t in detection_tracks if t.score >= self.high_threshold]
        low_score_tracks = [t for t in detection_tracks if t.score < self.high_threshold and t.score >= self.low_threshold]
        
        # 活躍軌跡預測
        tracked_tracks = []
        for track in self.tracks:
            track.predict()
            tracked_tracks.append(track)
        
        # 第一階段關聯：高置信度檢測與活躍軌跡匹配
        matches_a, unmatched_tracks_a, unmatched_detections_a = \
            self._associate_detections_to_tracks(tracked_tracks, high_score_tracks)
        
        # 更新匹配的軌跡
        for track_idx, det_idx in matches_a:
            self.tracks[track_idx].update(high_score_tracks[det_idx], self.frame_id)
        
        # 第二階段關聯：未匹配的軌跡與低置信度檢測匹配
        r_tracked_tracks = [self.tracks[i] for i in unmatched_tracks_a]
        matches_b, unmatched_tracks_b, unmatched_detections_b = \
            self._associate_detections_to_tracks(r_tracked_tracks, low_score_tracks)
        
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
        STrack._next_id = 1  # 重置ID計數器
    
    def _associate_detections_to_tracks(
        self, 
        tracks: List[STrack], 
        detections: List[STrack]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        將檢測結果與現有軌跡進行關聯
        
        Args:
            tracks: 現有軌跡列表
            detections: 檢測結果列表
            
        Returns:
            matches: 匹配的軌跡和檢測索引對 [(track_idx, det_idx), ...]
            unmatched_tracks: 未匹配的軌跡索引列表
            unmatched_detections: 未匹配的檢測索引列表
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # 計算IoU矩陣
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t, track in enumerate(tracks):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self._iou(track.xyxy, detection.xyxy)
        
        # 使用匈牙利算法進行最優匹配
        if min(iou_matrix.shape) > 0:
            # 將IoU轉換為代價矩陣（1-IoU）
            cost_matrix = 1.0 - iou_matrix
            
            # 使用scipy的匈牙利算法求解
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
            
            # 根據閾值過濾匹配結果
            matches = []
            for t, d in zip(track_indices, detection_indices):
                if iou_matrix[t, d] >= self.iou_threshold:
                    matches.append((t, d))
                else:
                    # IoU低於閾值視為未匹配
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