#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StrongSORT追蹤器實現
基於DeepSORT的改進版本，添加了更強的外觀特徵模型和運動預測
"""
import numpy as np
import cv2  # 添加缺少的OpenCV導入
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import linear_sum_assignment

from core.tracker_base import TrackerBase
from core.tracking.strong_sort_object import StrongTrack
from config.settings import TRACKING_MAX_AGE, TRACKING_MIN_HITS, TRACKING_IOU_THRESH

class StrongSort(TrackerBase):
    """
    StrongSORT追蹤器
    結合高級外觀模型與ECC(增強相關係數最大化)相機運動補償的強大追蹤器
    
    論文參考：StrongSORT: Make DeepSORT Great Again
    """
    
    def __init__(
        self,
        max_age: int = TRACKING_MAX_AGE,
        min_hits: int = 1,
        iou_threshold: float = TRACKING_IOU_THRESH,
        reid_threshold: float = 0.25,      # ReID距離閾值
        use_appearance: bool = True,       # 是否使用外觀特徵
        use_ecc: bool = True,              # 是否使用ECC相機運動補償
        appearance_weight: float = 0.75,   # 外觀特徵權重 (0~1)
    ):
        """
        初始化StrongSORT追蹤器
        
        Args:
            max_age: 最大追蹤年齡，超過此值的軌跡將被刪除
            min_hits: 最小檢測次數，達到此值後軌跡才會被激活
            iou_threshold: IoU匹配閾值
            reid_threshold: ReID特徵距離閾值，大於此閾值的匹配被拒絕
            use_appearance: 是否使用外觀特徵進行匹配
            use_ecc: 是否使用ECC相機運動補償
            appearance_weight: 外觀特徵權重 (0~1)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold
        self.use_appearance = use_appearance
        self.use_ecc = use_ecc
        self.appearance_weight = appearance_weight
        
        # 追蹤狀態
        self.frame_id = 0
        self.tracks = []  # 活躍的追蹤軌跡
        self.lost_tracks = []  # 暫時丟失的軌跡
        self.removed_tracks = []  # 已刪除的軌跡
        
        # 特徵提取器 (簡化版，實際使用時應加載預訓練模型)
        self.feature_extractor = None
        
        # ECC相機運動補償相關
        self.prev_gray = None  # 上一幀灰度圖像
        self.warp_matrix = np.eye(2, 3, dtype=np.float32)  # 變換矩陣
        
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
        
        # 相機運動補償 (如果啟用)
        if self.use_ecc and frame is not None:
            self._apply_ecc_motion_compensation(frame)
        
        # 轉換檢測結果為StrongTrack對象
        detection_tracks = []
        for det in detections:
            bbox = np.array(det['bbox'])  # [x1, y1, x2, y2]
            score = det['confidence']
            cls = det.get('class_id', 0)  # 預設為人類 (class_id=0)
            
            # 提取外觀特徵 (如果啟用)
            feat = None
            if self.use_appearance and frame is not None:
                feat = self._extract_features(frame, bbox)
            
            # 創建StrongTrack對象
            strong_track = StrongTrack(bbox, score, cls, feat)
            detection_tracks.append(strong_track)
        
        # 當前活躍軌跡預測下一步
        tracked_tracks = []
        for track in self.tracks:
            track.predict()
            tracked_tracks.append(track)
        
        # 執行關聯
        # StrongSORT使用級聯匹配策略，首先使用外觀特徵進行高置信度匹配，然後再使用IoU匹配處理剩餘檢測
        
        # 第一階段：外觀匹配
        # 將當前軌跡與新檢測進行關聯，使用外觀特徵和馬氏距離
        matches_a, unmatched_tracks_a, unmatched_detections_a = \
            self._associate(tracked_tracks, detection_tracks, frame, "appearance")
        
        # 更新匹配的軌跡
        for track_idx, det_idx in matches_a:
            self.tracks[track_idx].update(detection_tracks[det_idx], self.frame_id)
        
        # 第二階段：IoU匹配
        # 對未匹配的軌跡和檢測使用IoU進行關聯
        r_tracked_tracks = [self.tracks[i] for i in unmatched_tracks_a]
        r_detection_tracks = [detection_tracks[i] for i in unmatched_detections_a]
        
        matches_b, unmatched_tracks_b, unmatched_detections_b = \
            self._associate(r_tracked_tracks, r_detection_tracks, frame, "iou")
        
        # 更新匹配的軌跡
        for track_idx, det_idx in matches_b:
            r_tracked_tracks[track_idx].update(r_detection_tracks[det_idx], self.frame_id)
        
        # 處理未匹配的軌跡：加入丟失列表
        for i in unmatched_tracks_b:
            track = r_tracked_tracks[i]
            if track.time_since_update <= self.max_age:
                self.lost_tracks.append(track)
        
        # 處理未匹配的檢測：創建新軌跡
        remaining_detections = [r_detection_tracks[i] for i in unmatched_detections_b]
        for track in remaining_detections:
            if track.score >= 0.5:  # 只為高置信度檢測創建新軌跡
                track.assign_id()  # 分配新ID
                self.tracks.append(track)
        
        # 嘗試恢復丟失的軌跡
        # StrongSORT特性: 使用外觀特徵恢復長期丟失的軌跡
        if len(self.lost_tracks) > 0 and len(remaining_detections) > 0:
            matches_c, _, _ = self._associate(
                self.lost_tracks, remaining_detections, frame, 
                "appearance",
                reid_threshold=self.reid_threshold * 1.5  # 對丟失軌跡使用更寬鬆的閾值
            )
            
            # 恢復匹配的軌跡
            for lost_idx, det_idx in matches_c:
                lost_track = self.lost_tracks[lost_idx]
                det_track = remaining_detections[det_idx]
                lost_track.update(det_track, self.frame_id)
                
                # 從丟失列表移回活躍列表
                self.tracks.append(lost_track)
                self.lost_tracks[lost_idx] = None
            
            # 清理已恢復的軌跡
            self.lost_tracks = [t for t in self.lost_tracks if t is not None]
        
        # 更新所有丟失軌跡
        lost_idx = []
        for i, track in enumerate(self.lost_tracks):
            track.predict()
            if track.time_since_update > self.max_age:
                lost_idx.append(i)
                self.removed_tracks.append(track)
        
        # 刪除過期的丟失軌跡
        for idx in reversed(lost_idx):
            self.lost_tracks.pop(idx)
        
        # 更新卡爾曼濾波器狀態
        # StrongSORT特性: 使用NSA (非規則形狀參數化)改進卡爾曼狀態估計
        for track in self.tracks:
            track.update_shape()
        
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
        StrongTrack._next_id = 1  # 重置ID計數器
        
        # 重置ECC狀態
        self.prev_gray = None
        self.warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    def _associate(
        self, 
        tracks: List[StrongTrack], 
        detections: List[StrongTrack],
        frame: Optional[np.ndarray] = None,
        method: str = "hybrid",  # "iou", "appearance", or "hybrid"
        reid_threshold: Optional[float] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        將檢測結果與現有軌跡進行關聯
        
        Args:
            tracks: 現有軌跡列表
            detections: 檢測結果列表
            frame: 當前幀圖像
            method: 匹配方法 ("iou", "appearance", "hybrid")
            reid_threshold: ReID距離閾值，不指定則使用類屬性
            
        Returns:
            matches: 匹配的軌跡和檢測索引對 [(track_idx, det_idx), ...]
            unmatched_tracks: 未匹配的軌跡索引列表
            unmatched_detections: 未匹配的檢測索引列表
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        if reid_threshold is None:
            reid_threshold = self.reid_threshold
            
        # 初始化矩陣
        costs = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        
        # 計算不同的距離矩陣
        if method == "iou" or method == "hybrid":
            # 計算IoU距離矩陣 (1-IoU)
            for t, track in enumerate(tracks):
                for d, detection in enumerate(detections):
                    iou_val = self._iou(track.xyxy, detection.xyxy)
                    costs[t, d] = 1.0 - iou_val  # IoU距離 = 1-IoU
        
        if method == "appearance" or method == "hybrid":
            # 計算外觀特徵距離矩陣
            if self.use_appearance:
                for t, track in enumerate(tracks):
                    track_feat = track.get_feature()
                    if track_feat is None:
                        continue
                        
                    for d, detection in enumerate(detections):
                        det_feat = detection.get_feature()
                        if det_feat is None:
                            continue
                            
                        # 計算特徵向量的歐氏距離 (可選用餘弦距離)
                        appearance_dist = np.linalg.norm(track_feat - det_feat)
                        
                        if method == "appearance":
                            costs[t, d] = appearance_dist
                        else:  # hybrid
                            # 加權組合IoU和外觀特徵
                            iou_dist = costs[t, d]
                            costs[t, d] = self.appearance_weight * appearance_dist + \
                                         (1 - self.appearance_weight) * iou_dist
        
        # 使用匈牙利算法進行分配
        if np.isfinite(costs).any():
            row_indices, col_indices = linear_sum_assignment(costs)
            
            # 過濾匹配結果
            matches = []
            for r, c in zip(row_indices, col_indices):
                # 僅保留足夠接近的匹配
                if method == "iou" and costs[r, c] <= 1.0 - self.iou_threshold:
                    matches.append((r, c))
                elif method == "appearance" and costs[r, c] <= reid_threshold:
                    matches.append((r, c))
                elif method == "hybrid" and costs[r, c] <= reid_threshold:
                    # 對於混合模式，使用合併閾值
                    matches.append((r, c))
                else:
                    row_indices = np.append(row_indices, r)
                    col_indices = np.append(col_indices, c)
            
            # 計算未匹配的軌跡和檢測
            unmatched_tracks = list(set(range(len(tracks))) - set(row_indices))
            unmatched_detections = list(set(range(len(detections))) - set(col_indices))
        else:
            # 如果所有距離都是無效的，則全部視為未匹配
            matches = []
            unmatched_tracks = list(range(len(tracks)))
            unmatched_detections = list(range(len(detections)))
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _apply_ecc_motion_compensation(self, frame: np.ndarray) -> None:
        """
        應用ECC (增強相關係數最大化) 相機運動補償
        StrongSORT的一個關鍵改進，提高在移動相機場景的表現
        
        Args:
            frame: 當前幀圖像
        """
        if frame is None:
            return
            
        # 轉換為灰度圖
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 如果是第一幀，初始化並返回
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return
            
        # 設置ECC參數
        # 使用仿射變換 (translation + rotation + scale)
        warp_mode = cv2.MOTION_AFFINE
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001)
        
        try:
            # 執行ECC算法計算變換矩陣
            _, self.warp_matrix = cv2.findTransformECC(
                self.prev_gray, 
                curr_gray, 
                self.warp_matrix, 
                warp_mode, 
                criteria,
                None,
                1  # RANSAC初始估計
            )
            
            # 應用變換到所有軌跡
            for track in self.tracks:
                self._warp_track(track)
                
            for track in self.lost_tracks:
                self._warp_track(track)
        except Exception as e:
            # 如果ECC失敗，使用單位矩陣
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # 更新上一幀灰度圖
        self.prev_gray = curr_gray
    
    def _warp_track(self, track: StrongTrack) -> None:
        """
        對單個軌跡應用變換矩陣
        
        Args:
            track: 要應用變換的軌跡
        """
        if not track.trajectory:
            return
            
        # 應用到最近的軌跡點
        x, y = track.trajectory[-1]
        pt = np.array([[x], [y], [1]])
        new_pt = np.dot(self.warp_matrix, pt)
        track.trajectory[-1] = (int(new_pt[0][0]), int(new_pt[1][0]))
        
        # 應用到邊界框
        # 變換邊界框的四個角點
        x1, y1, x2, y2 = track.xyxy
        points = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ]).T
        
        # 應用變換
        new_points = np.dot(self.warp_matrix, points)
        
        # 計算新的邊界框 (最小外接矩形)
        min_x = np.min(new_points[0])
        min_y = np.min(new_points[1])
        max_x = np.max(new_points[0])
        max_y = np.max(new_points[1])
        
        # 更新邊界框
        track.xyxy = np.array([min_x, min_y, max_x, max_y])
        track.xywh = track._xyxy_to_xywh(track.xyxy)
        
        # 更新卡爾曼濾波器狀態
        track.mean[:4] = track.xywh
    
    def _extract_features(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        提取邊界框區域的視覺特徵
        在實際應用中應使用預訓練的ReID模型
        
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
            
            # StrongSORT中會使用高級特徵提取器，這裡用簡化版本
            # 調整大小並標準化
            resized = cv2.resize(crop, (64, 128))
            
            # 使用HOG特徵代替深度特徵 (僅用於演示)
            hog = cv2.HOGDescriptor()
            feature = hog.compute(resized)
            
            # 標準化
            if np.sum(feature) > 0:
                feature = feature / np.linalg.norm(feature)
            
            return feature.flatten()
        except Exception as e:
            # print(f"特徵提取錯誤: {e}")
            return None
    
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