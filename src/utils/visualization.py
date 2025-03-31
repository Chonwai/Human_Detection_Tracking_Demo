#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
視覺化工具函數
用於繪製檢測框、軌跡等
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import random


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    在圖像上繪製檢測框

    Args:
        frame: 輸入圖像
        detections: 檢測結果列表
        color: BGR顏色元組
        thickness: 線條粗細

    Returns:
        標註後的圖像
    """
    output_frame = frame.copy()

    for det in detections:
        # 獲取邊界框座標
        x1, y1, x2, y2 = map(int, det["bbox"])

        # 繪製邊界框
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)

        # 如果有置信度，則顯示
        if "confidence" in det:
            conf_text = f"{det['confidence']:.2f}"
            cv2.putText(
                output_frame,
                conf_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

        # 如果有追蹤ID，則顯示
        if "track_id" in det:
            id_text = f"ID: {det['track_id']}"
            cv2.putText(
                output_frame,
                id_text,
                (x1, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

    return output_frame


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
    show_bbox: bool = True,
    show_id: bool = True,
    show_trajectory: bool = True,
    id_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    show_all_detections: bool = True,  # 新增參數控制是否顯示所有檢測
) -> Tuple[np.ndarray, Dict]:
    """
    在圖像上繪製追蹤結果，包括邊界框、ID和軌跡

    Args:
        frame: 輸入圖像
        tracks: 追蹤結果列表
        show_bbox: 是否顯示邊界框
        show_id: 是否顯示追蹤ID
        show_trajectory: 是否顯示軌跡
        id_colors: ID到顏色的映射字典，如果為None則自動生成
        show_all_detections: 是否顯示所有檢測結果，包括未被追蹤的

    Returns:
        標註後的圖像和更新的ID顏色映射
    """
    output_frame = frame.copy()

    # 如果沒有提供顏色映射，則創建隨機顏色
    if id_colors is None:
        id_colors = {}

    # 首先檢查輸入是否為有效的列表/字典
    if not isinstance(tracks, (list, tuple)):
        # 如果不是列表或元組，嘗試轉換為列表
        try:
            tracks = [tracks]
        except:
            # 如果轉換失敗，返回原始幀
            return output_frame, id_colors

    # 分離有追蹤ID和無追蹤ID的結果
    tracked = []
    untracked = []

    for track in tracks:
        if (
            isinstance(track, dict)
            and "track_id" in track
            and track.get("track_id", -1) != -1
        ):
            tracked.append(track)
        else:
            untracked.append(track)

    # 先繪製未被追蹤的檢測結果（如果啟用）
    if show_all_detections:
        for det in untracked:
            if "bbox" in det:
                x1, y1, x2, y2 = map(int, det["bbox"])

                # 繪製傳統目標檢測框（綠色）
                if show_bbox:
                    # 繪製半透明背景框
                    overlay = output_frame.copy()
                    cv2.rectangle(
                        overlay, (x1, y1), (x2, y2), (0, 200, 0), -1
                    )  # 填充內部
                    output_frame = cv2.addWeighted(overlay, 0.2, output_frame, 0.8, 0)

                    # 繪製邊界框
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

                # 顯示類別標籤和置信度
                label_text = ""
                if "class_name" in det:
                    label_text = det["class_name"]
                elif "class_id" in det:
                    # 預設為人類
                    label_text = "person"

                if "confidence" in det:
                    conf = det["confidence"]
                    if label_text:
                        label_text += " " + str(round(conf, 2))
                    else:
                        label_text = str(round(conf, 2))

                if label_text:
                    # 計算文本大小以適應標籤背景
                    label_size, baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )

                    # 繪製標籤背景
                    cv2.rectangle(
                        output_frame,
                        (x1, y1 - label_size[1] - 5),
                        (x1 + label_size[0], y1),
                        (0, 200, 0),
                        -1,
                    )

                    # 繪製標籤文字
                    cv2.putText(
                        output_frame,
                        label_text,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),  # 白色文字
                        1,
                    )

    # 使用單一顏色避免閃爍
    single_color = (0, 200, 0)  # 綠色

    # 繪製有追蹤ID的結果
    for track in tracked:
        track_id = track.get("track_id", -1)

        if "bbox" in track:
            x1, y1, x2, y2 = map(int, track["bbox"])

            # 繪製邊界框
            if show_bbox:
                # 繪製半透明背景框
                overlay = output_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), single_color, -1)  # 填充內部
                output_frame = cv2.addWeighted(overlay, 0.2, output_frame, 0.8, 0)

                # 繪製邊界框
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), single_color, 2)

            # 準備標籤文本
            label_parts = []

            # 如果顯示ID，添加ID信息
            if show_id:
                label_parts.append("ID: " + str(track_id))

            # 添加類別標籤
            if "class_name" in track:
                label_parts.append(track["class_name"])
            elif "class_id" in track:
                # 預設為人類
                label_parts.append("person")

            # 添加置信度
            if "confidence" in track:
                label_parts.append(str(round(track["confidence"], 2)))

            # 組合標籤文本
            label_text = " ".join(label_parts)

            if label_text:
                # 計算文本大小以適應標籤背景
                label_size, baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # 繪製標籤背景
                cv2.rectangle(
                    output_frame,
                    (x1, y1 - label_size[1] - 5),
                    (x1 + label_size[0], y1),
                    single_color,
                    -1,
                )

                # 繪製標籤文字
                cv2.putText(
                    output_frame,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # 白色文字
                    1,
                )

            # 繪製軌跡
            if (
                show_trajectory
                and "trajectory" in track
                and len(track["trajectory"]) > 1
            ):
                for i in range(1, len(track["trajectory"])):
                    # 繪製軌跡線段
                    p1 = track["trajectory"][i - 1]
                    p2 = track["trajectory"][i]
                    cv2.line(output_frame, p1, p2, single_color, 2)

    return output_frame, id_colors


def draw_stats(
    frame: np.ndarray,
    person_count: int = 0,
    fps: float = 0.0,
    position: str = "top-left",
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    在圖像上繪製統計信息

    Args:
        frame: 輸入圖像
        person_count: 檢測到的人數，默認為0
        fps: 處理幀率，默認為0.0
        position: 顯示位置 ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        bg_color: 背景顏色
        text_color: 文字顏色

    Returns:
        添加統計信息後的圖像
    """
    # 檢查輸入
    if frame is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)  # 返回空白畫面

    output_frame = frame.copy()
    h, w = output_frame.shape[:2]

    # 準備統計文本
    stats_text = "People: " + str(person_count) + " | FPS: " + str(round(fps, 1))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # 計算文本大小
    (text_w, text_h), baseline = cv2.getTextSize(
        stats_text, font, font_scale, thickness
    )

    # 確定位置
    if position == "top-left":
        x, y = 10, 30
    elif position == "top-right":
        x, y = w - text_w - 10, 30
    elif position == "bottom-left":
        x, y = 10, h - 10
    elif position == "bottom-right":
        x, y = w - text_w - 10, h - 10
    else:  # 默認左上角
        x, y = 10, 30

    # 繪製背景矩形
    cv2.rectangle(
        output_frame,
        (x - 5, y - text_h - 5),
        (x + text_w + 5, y + 5),
        bg_color,
        -1,  # 填充矩形
    )

    # 繪製文本
    cv2.putText(
        output_frame, stats_text, (x, y), font, font_scale, text_color, thickness
    )

    return output_frame


def create_heatmap(
    tracks: List[Dict[str, Any]], frame_shape: Tuple[int, int], alpha: float = 0.5
) -> np.ndarray:
    """
    根據軌跡創建熱力圖

    Args:
        tracks: 追蹤結果列表
        frame_shape: 幀的形狀 (height, width)
        alpha: 熱力圖透明度

    Returns:
        熱力圖
    """
    height, width = frame_shape

    # 創建累積熱力圖
    heatmap = np.zeros((height, width), dtype=np.uint8)

    # 為每個軌跡添加熱點
    for track in tracks:
        if "trajectory" in track:
            for x, y in track["trajectory"]:
                if 0 <= x < width and 0 <= y < height:
                    # 在軌跡點周圍添加高斯模糊熱點
                    cv2.circle(heatmap, (x, y), 10, 255, -1)

    # 應用高斯模糊
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # 轉換為熱力圖顏色
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap_color


def draw_all_detections(
    frame: np.ndarray,
    results: List[Dict[str, Any]],
    show_bbox: bool = True,
    show_id: bool = True,
    show_trajectory: bool = True,
    id_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    在圖像上同時繪製追蹤結果和檢測結果

    Args:
        frame: 輸入圖像
        results: 結果列表，包含追蹤和檢測結果
        show_bbox: 是否顯示邊界框
        show_id: 是否顯示追蹤ID
        show_trajectory: 是否顯示軌跡
        id_colors: ID到顏色的映射字典，如果為None則自動生成

    Returns:
        標註後的圖像和更新的顏色映射
    """
    output_frame = frame.copy()

    # 如果沒有提供顏色映射，則創建
    if id_colors is None:
        id_colors = {}

    # 先繪製未追蹤的檢測結果（灰色框）
    for result in results:
        # 如果結果被標記為未追蹤（或沒有track_id），則使用灰色框
        if result.get("is_tracked", False) == False or "track_id" not in result:
            x1, y1, x2, y2 = map(int, result["bbox"])

            # 灰色框表示未追蹤的檢測
            if show_bbox:
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (128, 128, 128), 2)

            # 顯示置信度
            if "confidence" in result:
                conf_text = f"{result['confidence']:.2f}"
                cv2.putText(
                    output_frame,
                    conf_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (128, 128, 128),
                    2,
                )

    # 再繪製已追蹤的結果（彩色框）
    for result in results:
        if result.get("is_tracked", True) == True and "track_id" in result:
            track_id = result["track_id"]

            # 為新ID生成顏色
            if track_id not in id_colors:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                id_colors[track_id] = (b, g, r)  # OpenCV使用BGR格式

            color = id_colors[track_id]
            x1, y1, x2, y2 = map(int, result["bbox"])

            # 繪製邊界框
            if show_bbox:
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            # 繪製ID
            if show_id:
                id_text = f"ID: {track_id}"
                cv2.putText(
                    output_frame,
                    id_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # 如果有置信度，也顯示
                if "confidence" in result:
                    conf_text = f"{result['confidence']:.2f}"
                    cv2.putText(
                        output_frame,
                        conf_text,
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

            # 繪製軌跡
            if (
                show_trajectory
                and "trajectory" in result
                and len(result["trajectory"]) > 1
            ):
                for i in range(1, len(result["trajectory"])):
                    # 繪製軌跡線段
                    p1 = result["trajectory"][i - 1]
                    p2 = result["trajectory"][i]
                    cv2.line(output_frame, p1, p2, color, 2)

    return output_frame, id_colors
