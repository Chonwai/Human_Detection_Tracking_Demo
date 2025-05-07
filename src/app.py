#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主應用入口
整合所有組件並提供Streamlit界面
"""

import streamlit as st
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
import time
import tempfile
import plotly.graph_objects as go
from collections import deque
import os

from config.settings import *
from config.logging_config import setup_logging
from models.yolo_detector import YOLODetector
from core.tracking.byte_tracker import ByteTracker
from core.tracking.bot_sort import BoTSort
from core.tracking.strong_sort import StrongSort
from utils.video_utils import get_video_source, video_frame_generator, get_video_info
from utils.visualization import (
    draw_tracks,
    draw_stats,
    create_heatmap,
    draw_all_detections,
)
from utils.image_enhancement import enhance_image, apply_histogram_equalization
from core.platform_utils import is_jetson, is_mac, get_platform_info
from utils.script_handler import load_script_from_uploaded_file

# 設置頁面配置
st.set_page_config(
    page_title="人體檢測與追蹤系統",
    page_icon="🎯",
    layout="wide",
)

# 初始化日誌
logger = setup_logging()

# 會話狀態初始化
if "processing" not in st.session_state:
    st.session_state.processing = False
if "stats" not in st.session_state:
    st.session_state.stats = {
        "person_count": [],
        "timestamps": [],
        "fps": [],
        "track_count": [],  # 添加追蹤ID計數
    }
if "id_colors" not in st.session_state:
    st.session_state.id_colors = {}  # 追蹤ID對應的顏色映射
if "all_tracks" not in st.session_state:
    st.session_state.all_tracks = []  # 存儲所有軌跡，用於熱力圖
if "heatmap_enabled" not in st.session_state:
    st.session_state.heatmap_enabled = False  # 熱力圖開關
# 新增用於實時統計的狀態變量 (即時更新的關鍵)
if "current_stats" not in st.session_state:
    st.session_state.current_stats = {
        "person_count": 0,
        "track_count": 0,
        "fps": 0.0,
        "frame_count": 0,
        "last_update_time": 0,
    }  # 當前幀的統計數據，用於即時顯示
if "update_interval" not in st.session_state:
    st.session_state.update_interval = 0.5  # 統計更新間隔(秒)

# 視頻處理狀態保存 - 新增：用於恢復視頻處理的完整狀態
if "video_state" not in st.session_state:
    st.session_state.video_state = {
        "source": None,                 # 當前視頻源
        "frame_position": 0,            # 當前處理到的幀位置
        "detector_config": None,        # 檢測器配置
        "tracker_config": None,         # 追蹤器配置
        "display_config": None,         # 顯示配置
        "enhancement_config": None      # 圖像增強配置
    }

# 處理連續性標記 - 新增：確保處理的連續性
if "processing_continuity" not in st.session_state:
    st.session_state.processing_continuity = {
        "is_continuation": False,       # 是否為繼續處理
        "last_frame": None,             # 上次處理的幀
        "detector": None,               # 保存的檢測器對象
        "tracker": None                 # 保存的追蹤器對象
    }

# 新增劇本相關的會話狀態初始化
if "script_file_name" not in st.session_state:
    st.session_state.script_file_name = None
if "parsed_script" not in st.session_state:
    st.session_state.parsed_script = None
if "script_error_message" not in st.session_state:
    st.session_state.script_error_message = None
if "last_script_load_success" not in st.session_state:
    st.session_state.last_script_load_success = False
if "active_display_cues" not in st.session_state: # 用於存儲當前幀要顯示的 cues
    st.session_state.active_display_cues = []

def main():
    """
    主應用入口函數
    """
    # 檢查是否從暫停狀態恢復（從st.rerun()返回）
    if "temp_processing_state" in st.session_state and st.session_state.temp_processing_state:
        # 恢復處理狀態 - 增強版：標記為連續處理，確保不重新初始化視頻源
        st.session_state.processing = True
        st.session_state.processing_continuity["is_continuation"] = True  # 標記為連續處理
        # 清除臨時狀態 - 確保立即重置，避免無限循環
        st.session_state.temp_processing_state = False
        
        # 簡化循環保護：不使用阻塞式延遲，減少卡頓感
        current_time = time.time()
        if "last_rerun_time" not in st.session_state:
            st.session_state.last_rerun_time = 0
        
        # 僅記錄日誌，不進行阻塞式延遲
        st.session_state.last_rerun_time = current_time
        logger.info("從UI更新中恢復，繼續視頻處理")

    # ==== 函數定義區開始 ====
    # 定義用於更新統計區塊的函數
    def update_stats_display(container, tracking_enabled=False):
        """
        更新統計顯示面板
        
        Args:
            container: streamlit容器，用於顯示統計信息
            tracking_enabled: 布爾值，是否啟用了追蹤功能
        """
        # 檢查容器是否存在
        if container is None:
            logger.warning("統計容器未初始化")
            return
            
        # 獲取當前統計數據
        current_count = st.session_state.current_stats["person_count"]
        current_fps = st.session_state.current_stats["fps"]
        current_tracks = st.session_state.current_stats["track_count"]
        frame_count = st.session_state.current_stats["frame_count"]
        
        # 使用Streamlit原生組件而非HTML - 避免HTML渲染問題
        # 使用列（columns）布局來創建卡片樣式
        cols = container.columns(2)
        
        with cols[0]:
            st.metric(label="檢測人數", value=current_count)
            st.metric(label="FPS", value=f"{current_fps:.1f}")
            
        with cols[1]:
            if tracking_enabled:
                st.metric(label="追蹤ID數量", value=current_tracks)
            st.metric(label="已處理幀數", value=frame_count)
    
    # 定義圖表更新函數
    def update_trend_charts(trend_chart, fps_chart, track_chart=None, tracking_enabled=False):
        """
        更新趨勢圖表
        
        Args:
            trend_chart: streamlit容器，用於顯示人數趨勢圖
            fps_chart: streamlit容器，用於顯示FPS趨勢圖
            track_chart: streamlit容器，用於顯示軌跡分析圖（可選）
            tracking_enabled: 布爾值，是否啟用了追蹤功能
        """
        # 檢查圖表容器是否存在
        if trend_chart is None or fps_chart is None:
            logger.warning("圖表容器未初始化")
            return
            
        # 生成當前時間戳作為key的一部分，確保每次更新都有唯一的key
        current_timestamp = int(time.time() * 1000)
            
        if st.session_state.stats["timestamps"]:
            # 創建相對時間軸
            relative_times = [
                t - st.session_state.stats["timestamps"][0]
                for t in st.session_state.stats["timestamps"]
            ]
            
            # 人數趨勢圖
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(
                    x=relative_times,
                    y=st.session_state.stats["person_count"],
                    mode="lines",
                    name="人數",
                    line=dict(color="#1e40af", width=2),
                )
            )
            
            # 如果啟用追蹤，也顯示追蹤ID數量
            if tracking_enabled and "track_count" in st.session_state.stats:
                fig1.add_trace(
                    go.Scatter(
                        x=relative_times,
                        y=st.session_state.stats["track_count"],
                        mode="lines",
                        name="追蹤ID",
                        line=dict(color="#d97706", width=2, dash="dot"),
                    )
                )
            
            fig1.update_layout(
                xaxis_title="時間 (秒)",
                yaxis_title="數量",
                height=200,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                font=dict(size=10),
            )
            # 使用當前時間戳作為key的一部分，確保唯一性
            trend_chart.plotly_chart(fig1, use_container_width=True, key=f"person_trend_chart_{current_timestamp}")
            
            # FPS趨勢圖
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=relative_times,
                    y=st.session_state.stats["fps"],
                    mode="lines",
                    name="FPS",
                    line=dict(color="#047857", width=2),
                )
            )
            fig2.update_layout(
                xaxis_title="時間 (秒)",
                yaxis_title="FPS",
                height=200,
                margin=dict(l=0, r=0, t=10, b=0),
                font=dict(size=10),
            )
            # 使用當前時間戳作為key的一部分，確保唯一性
            fps_chart.plotly_chart(fig2, use_container_width=True, key=f"fps_trend_chart_{current_timestamp}")
            
            # 軌跡分析（如果啟用追蹤）
            if tracking_enabled and track_chart and len(st.session_state.all_tracks) > 0:
                # 計算各軌跡持續時間
                track_durations = {}
                for track in st.session_state.all_tracks:
                    track_id = track.get("track_id")
                    if track_id is not None and "trajectory" in track:
                        if track_id not in track_durations:
                            track_durations[track_id] = 0
                        track_durations[track_id] += 1
                
                # 顯示軌跡持續時間圖表
                if track_durations:
                    # 只取前8個最長的軌跡
                    sorted_durations = sorted(
                        track_durations.items(), key=lambda x: x[1], reverse=True
                    )[:8]
                    track_ids = ["ID " + str(id) for id, _ in sorted_durations]
                    durations = [duration for _, duration in sorted_durations]
                    
                    fig3 = go.Figure()
                    fig3.add_trace(
                        go.Bar(
                            x=track_ids,
                            y=durations,
                            marker_color="#4f46e5",
                        )
                    )
                    fig3.update_layout(
                        xaxis_title="追蹤ID",
                        yaxis_title="持續幀數",
                        height=200,
                        margin=dict(l=0, r=0, t=10, b=0),
                        font=dict(size=10),
                    )
                    # 使用當前時間戳作為key的一部分，確保唯一性
                    track_chart.plotly_chart(fig3, use_container_width=True, key=f"track_duration_chart_{current_timestamp}")
    # ==== 函數定義區結束 ====

    # 添加自定義CSS以支持深色模式
    st.markdown(
        """
    <style>
    /* 深色模式下的卡片樣式 */
    [data-testid="stAppViewContainer"] .stApp.stApp--dark div[style*="background-color: #f0f2f6"] {
        background-color: #2c3e50 !important;
        color: deepblue !important;
    }
    
    /* 確保深色模式下圖表標籤更加可見 */
    [data-testid="stAppViewContainer"] .stApp.stApp--dark .js-plotly-plot .plotly .gtitle, 
    [data-testid="stAppViewContainer"] .stApp.stApp--dark .js-plotly-plot .plotly .xtitle,
    [data-testid="stAppViewContainer"] .stApp.stApp--dark .js-plotly-plot .plotly .ytitle {
        fill: #ffffff !important;
    }
    
    /* 使趨勢線在深色背景下更清晰 */
    [data-testid="stAppViewContainer"] .stApp.stApp--dark .js-plotly-plot .plotly .scatter .lines path {
        stroke-width: 3px !important;
    }

    /* 自定義統計卡片樣式 */
    .stat-card {
        padding: 10px;
        border-radius: 8px;
        color: white;
        margin-bottom: 10px;
        text-align: center;
    }
    .person-card { background-color: #1e3a8a; }
    .track-card { background-color: #065f46; }
    .fps-card { background-color: #9d174d; }
    .frame-card { background-color: #4338ca; }
    .stat-label { 
        margin: 0; 
        font-size: 0.85rem; 
        opacity: 0.9;
    }
    .stat-value { 
        margin: 0; 
        font-weight: bold; 
        font-size: 1.5rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 初始化顯示統計信息
    if "current_stats" not in st.session_state:
        st.session_state.current_stats = {
            "person_count": 0,
            "track_count": 0,
            "fps": 0.0,
            "frame_count": 0,
            "last_update_time": time.time(),
        }
        
    # 重要：解決變數作用域問題 - 在主函數作用域定義UI容器，確保在各處理區域都能訪問
    stats_container = None
    trend_chart = None  
    fps_chart = None
    track_chart = None

    # 顯示標題
    st.title("人體檢測與追蹤系統")

    # 側邊欄配置
    with st.sidebar:
        st.header("系統配置")

        # 選擇輸入源
        source_type = st.radio(
            "選擇輸入源", ["視頻文件", "攝像頭", "RTSP串流", "示例視頻"]
        )
        
        # === 新增配置選項 ===
        with st.expander("高級設置", expanded=False):
            # 添加完整頁面刷新設置
            enable_full_refresh = st.checkbox("啟用定期頁面刷新", value=False, 
                                             help="定期重載頁面以優化長時間運行性能。若應用變得緩慢可嘗試啟用此選項。")
            
            if enable_full_refresh:
                refresh_interval = st.slider("頁面刷新間隔（秒）", 
                                           min_value=10, max_value=120, value=30, step=10,
                                           help="設置多久自動刷新一次頁面。時間越長，視頻處理越連續，但可能影響長時間穩定性。")
            else:
                refresh_interval = 999999  # 實際上禁用刷新
                
            st.caption("注意：頁面刷新可能導致視頻處理短暫中斷，但有助於釋放內存和保持長時間運行穩定。")
            
            # 添加Jetson平台特定優化選項
            # 獲取平台信息
            platform_info = get_platform_info()
            
            # Jetson平台優化選項
            if is_jetson():
                st.markdown("### Jetson平台優化")
                
                # 處理分辨率調整（較低分辨率可提高性能）
                resolution_scale = st.slider(
                    "處理分辨率縮放", 
                    min_value=0.25, 
                    max_value=1.0, 
                    value=0.5, 
                    step=0.05,
                    help="降低處理分辨率可顯著提高性能，但可能影響檢測精度"
                )
                
                # TensorRT加速選項
                enable_tensorrt = st.checkbox(
                    "啟用TensorRT加速", 
                    value=True,
                    help="使用NVIDIA TensorRT加速推理。首次啟用需要較長時間優化模型。"
                )
                
                # 批處理大小選項
                batch_size = st.radio(
                    "批處理大小",
                    options=[1, 2, 4],
                    index=0,
                    help="較大的批處理可能提高吞吐量，但會增加延遲"
                )
                
                # 半精度計算選項
                use_half_precision = st.checkbox(
                    "使用FP16半精度",
                    value=True,
                    help="使用半精度浮點數可顯著提高性能，但可能略微降低精度"
                )
                
                st.info("⚠️ 提示：在Jetson平台上，建議使用較小的模型（如yolov10n）和較低的處理分辨率以獲得最佳性能。")
                
                # 設置環境變量（這裡只是記錄，實際應該在啟動腳本中設置）
                st.caption("請確保已在執行環境中設置以下環境變量：")
                st.code("export OPENBLAS_CORETYPE=ARMV8")
            
            # Mac平台優化選項    
            elif is_mac():
                st.markdown("### Mac平台優化")
                
                # GPU內存限制選項
                gpu_mem_limit = st.slider(
                    "GPU內存限制 (GB)", 
                    min_value=1.0, 
                    max_value=8.0, 
                    value=4.0, 
                    step=0.5,
                    help="限制PyTorch使用的GPU內存量，可防止系統內存不足"
                )
                
                # 啟用MPS加速選項
                if platform_info.get("mps_available", False):
                    use_mps = st.checkbox(
                        "使用Apple Silicon加速 (MPS)", 
                        value=True,
                        help="啟用Metal Performance Shaders (MPS) 來加速模型推理"
                    )
                    
                    st.info("⚠️ 提示：在某些情況下，如果遇到MPS相關錯誤，您可能需要禁用MPS加速。")
                
                # 電源管理選項
                power_mode = st.radio(
                    "電源模式",
                    options=["平衡", "高性能", "節能"],
                    index=0,
                    help="高性能模式會消耗更多電池，但處理速度更快"
                )
            
            # 通用性能優化選項（適用於所有平台）
            st.markdown("### 通用優化")
            
            # 線程設置
            worker_threads = st.slider(
                "工作線程數", 
                min_value=1, 
                max_value=16, 
                value=4, 
                step=1,
                help="用於視頻處理的線程數量。更多線程可能提高性能，但會增加CPU使用率"
            )
            
            # 緩存設置
            enable_cache = st.checkbox(
                "啟用模型緩存", 
                value=True,
                help="緩存模型以加速重複推理，但會使用更多內存"
            )
            
            # 存儲設置到session_state
            for key, value in {
                "worker_threads": worker_threads,
                "enable_cache": enable_cache,
            }.items():
                if key not in st.session_state or st.session_state[key] != value:
                    st.session_state[key] = value
                    
            # 針對特定平台存儲設置
            if is_jetson():
                for key, value in {
                    "resolution_scale": resolution_scale,
                    "enable_tensorrt": enable_tensorrt,
                    "batch_size": batch_size,
                    "use_half_precision": use_half_precision
                }.items():
                    if key not in st.session_state or st.session_state[key] != value:
                        st.session_state[key] = value
            elif is_mac():
                for key, value in {
                    "gpu_mem_limit": gpu_mem_limit,
                    "power_mode": power_mode
                }.items():
                    if key not in st.session_state or st.session_state[key] != value:
                        st.session_state[key] = value
                
                if platform_info.get("mps_available", False):
                    if "use_mps" not in st.session_state or st.session_state.use_mps != use_mps:
                        st.session_state.use_mps = use_mps

        # 輸入源選擇
        video_source = None
        if source_type == "視頻文件":
            video_file = st.file_uploader("上傳視頻文件", type=["mp4", "avi", "mov"])
            if video_file:
                video_source = video_file

        elif source_type == "攝像頭":
            camera_id = st.number_input("攝像頭ID", min_value=0, value=0)
            video_source = int(camera_id)

        elif source_type == "RTSP串流":
            rtsp_url = st.text_input("RTSP URL")
            if rtsp_url:
                video_source = rtsp_url

        else:  # 示例視頻
            # 檢查示例視頻目錄
            example_dir = VIDEO_DIR
            example_dir.mkdir(parents=True, exist_ok=True)

            example_files = list(example_dir.glob("*.mp4")) + list(
                example_dir.glob("*.avi")
            )

            if not example_files:
                st.warning("示例視頻目錄為空。請先上傳視頻到 'data/videos/' 目錄。")
            else:
                example_options = [f.name for f in example_files]
                selected_example = st.selectbox("選擇示例視頻", example_options)
                if selected_example:
                    video_source = example_dir / selected_example

        # 檢測配置
        st.subheader("檢測配置")

        # 模型選擇 - 從模型目錄獲取可用模型
        available_models = []
        # 獲取可用模型列表
        for model_file in MODEL_DIR.glob("*.pt"):
            available_models.append(model_file.name)
        
        # 如果模型目錄為空，添加默認選項
        if not available_models:
            available_models = [
                "yolov10n.pt",
                "yolov10m.pt",
                "yolov10l.pt",
                "yolov10x.pt",
                "yolo12m.pt",
                "yolo12l.pt",
                "yolo12x.pt",
            ]
            st.warning("未在模型目錄中找到預訓練模型。請確保模型文件已放置在 data/models/ 目錄中。")
        
        # 按模型大小排序（n < s < m < l < x）
        def model_size_key(model_name):
            if 'n' in model_name: return 0
            if 's' in model_name: return 1
            if 'm' in model_name: return 2
            if 'l' in model_name: return 3
            if 'x' in model_name: return 4
            return 5
        
        available_models.sort(key=model_size_key)
        
        selected_model = st.selectbox("選擇模型", available_models)

        # 置信度閾值
        conf_threshold = st.slider(
            "置信度閾值", min_value=0.1, max_value=1.0, value=DETECTION_CONF, step=0.05
        )

        # 追蹤配置
        st.subheader("追蹤配置")
        tracking_enabled = st.checkbox("啟用追蹤", value=False)  # 默認關閉追蹤

        if tracking_enabled:
            # 追蹤器選擇
            tracker_options = [
                "ByteTracker (快速且穩定)",
                "BoT-SORT (結合視覺特徵)",
                "StrongSORT (強大運動補償)",
            ]
            selected_tracker = st.selectbox(
                "選擇追蹤器",
                tracker_options,
                index=0,
                help="不同追蹤器適用於不同場景：ByteTracker速度快，BoT-SORT在擁擠場景表現好，StrongSORT在相機移動場景效果佳",
            )

            # 通用追蹤參數
            min_hits = st.slider("最小確認幀數", min_value=1, max_value=10, value=1)
            max_age = st.slider(
                "最大丟失幀數", min_value=1, max_value=50, value=TRACKING_MAX_AGE
            )

            # 根據選擇的追蹤器顯示不同的高級參數
            with st.expander("高級追蹤參數"):
                iou_threshold = st.slider(
                    "IoU匹配閾值",
                    min_value=0.1,
                    max_value=0.9,
                    value=TRACKING_IOU_THRESH,
                    step=0.05,
                )

                # ByteTracker特有參數
                if selected_tracker == tracker_options[0]:  # ByteTracker
                    high_threshold = st.slider(
                        "高置信度閾值",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.3,
                        step=0.05,
                    )
                    low_threshold = st.slider(
                        "低置信度閾值",
                        min_value=0.05,
                        max_value=0.5,
                        value=0.1,
                        step=0.05,
                    )

                # BoT-SORT特有參數
                elif selected_tracker == tracker_options[1]:  # BoT-SORT
                    high_threshold = st.slider(
                        "高置信度閾值",
                        min_value=0.2,
                        max_value=0.9,
                        value=0.6,
                        step=0.05,
                    )
                    low_threshold = st.slider(
                        "低置信度閾值",
                        min_value=0.05,
                        max_value=0.5,
                        value=0.1,
                        step=0.05,
                    )
                    match_threshold = st.slider(
                        "特徵匹配閾值",
                        min_value=0.3,
                        max_value=0.9,
                        value=0.7,
                        step=0.05,
                    )
                    use_appearance = st.checkbox("使用外觀特徵", value=True)
                    use_cmc = st.checkbox("使用相機運動補償", value=True)

                # StrongSORT特有參數
                elif selected_tracker == tracker_options[2]:  # StrongSORT
                    reid_threshold = st.slider(
                        "ReID距離閾值",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.25,
                        step=0.05,
                    )
                    appearance_weight = st.slider(
                        "外觀特徵權重",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.75,
                        step=0.05,
                    )
                    use_appearance = st.checkbox("使用外觀特徵", value=True)
                    use_ecc = st.checkbox("使用ECC運動補償", value=True)

        # 顯示配置
        st.subheader("顯示配置")
        show_bbox = st.checkbox("顯示邊界框", value=True)
        show_id = st.checkbox("顯示ID", value=True)
        show_trajectory = st.checkbox("顯示軌跡", value=True)
        heatmap_enabled = st.checkbox("顯示熱力圖", value=False)
        if heatmap_enabled != st.session_state.heatmap_enabled:
            st.session_state.heatmap_enabled = heatmap_enabled

        # 圖像增強配置
        st.subheader("圖像增強配置")
        enable_enhancement = st.checkbox("啟用圖像增強", value=False)
        show_comparison = (
            st.checkbox("顯示增強效果對比", value=False)
            if enable_enhancement
            else False
        )

        if enable_enhancement:
            enhancement_method = st.selectbox(
                "增強方法",
                ["CLAHE (自適應直方圖均衡化)", "基本直方圖均衡化", "伽瑪校正"],
                index=0,
            )

            # 根據選擇的方法顯示相應的參數設置
            if enhancement_method == "伽瑪校正":
                gamma_value = st.slider(
                    "伽瑪值",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    help="調整圖像亮度，<1使圖像變亮，>1使圖像變暗",
                )
            elif enhancement_method == "CLAHE (自適應直方圖均衡化)":
                clip_limit = st.slider(
                    "對比度限制",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    help="較高的值給出更強的對比度",
                )
                tile_size = st.slider(
                    "網格大小",
                    min_value=1,
                    max_value=16,
                    value=8,
                    step=1,
                    help="較小的值處理更局部的細節",
                )

        # 性能控制
        st.subheader("性能設置")
        max_fps = st.slider("最大FPS", min_value=1, max_value=60, value=30)

        # 開始/停止按鈕
        if not st.session_state.processing:
            if st.button("開始檢測"):
                if video_source is not None:
                    # 重置統計
                    st.session_state.stats = {
                        "person_count": [],
                        "timestamps": [],
                        "fps": [],
                        "track_count": [],
                    }
                    st.session_state.id_colors = {}
                    st.session_state.all_tracks = []
                    st.session_state.processing = True
                    st.rerun()
                else:
                    st.error("請選擇有效的視頻源")
        else:
            if st.button("停止檢測"):
                st.session_state.processing = False
                st.rerun()

        with st.sidebar.expander("📝 劇本與提示設定", expanded=True):
            # 使用唯一的 key "script_uploader"
            uploaded_script_file = st.file_uploader("上傳劇本 JSON 檔案", type=["json"], key="script_uploader_key")

            if uploaded_script_file is not None:
                # 檢查是否是同一個檔案的重複處理 (可選優化，如果沒有按鈕觸發的話)
                # if st.session_state.get("script_file_name") != uploaded_script_file.name or not st.session_state.get("last_script_load_success"):
                
                logger.info(f"偵測到上傳的劇本檔案: {uploaded_script_file.name}")
                # 注意：UploadedFile 對象在 rerun 後可能會重置或行為不一致，
                # 理想情況下，應該在它首次出現時處理，然後依賴 session_state。
                # 為了簡化，這裡每次 rerender 且 file_uploader 有值時都可能重新處理，
                # 這在 Streamlit 中是常見模式，但若劇本大或處理耗時，需優化。
                # 假設 load_script_from_uploaded_file 接受 file-like object
                
                # 創建一個臨時副本以避免 "read of closed file" 錯誤，如果 load_script... 多次讀取
                # from io import BytesIO
                # file_buffer = BytesIO(uploaded_script_file.getvalue())
                # setattr(file_buffer, 'name', uploaded_script_file.name)
                # parsed_data, error_msg = load_script_from_uploaded_file(file_buffer)
                
                # 直接傳遞 uploaded_file_obj, script_handler 應處理它
                parsed_data, error_msg = load_script_from_uploaded_file(uploaded_script_file)

                st.session_state.parsed_script = parsed_data
                st.session_state.script_error_message = error_msg
                st.session_state.script_file_name = uploaded_script_file.name
                if parsed_data:
                    st.session_state.last_script_load_success = True
                    logger.info(f"劇本 '{uploaded_script_file.name}' 成功載入。")
                else:
                    st.session_state.last_script_load_success = False
                    logger.error(f"劇本 '{uploaded_script_file.name}' 載入失敗: {error_msg}")
                    
            # 在 uploader下方顯示載入狀態 (仍在 expander 內)
            if st.session_state.get('script_file_name'):
                if st.session_state.get('last_script_load_success'):
                    st.success(f"劇本 '{st.session_state.script_file_name}' 已成功載入。")
                elif st.session_state.get('script_error_message'):
                    # 使用 st.warning 或 st.error，並確保訊息換行正確顯示
                    st.error(f"載入劇本 '{st.session_state.script_file_name}' 失敗:\\n{st.session_state.script_error_message}")
            else:
                st.info("請上傳劇本檔案以啟用智慧提示功能。")

    # 主界面
    # 分為兩列：左側視頻顯示，右側統計信息
    col1, col2 = st.columns([4, 1])  # 修改比例讓視頻區域更寬

    with col1:
        # 視頻顯示區域
        video_placeholder = st.empty()

        # 如果啟用了對比顯示，創建兩列佈局
        if show_comparison:
            comp_col1, comp_col2 = st.columns(2)
            original_placeholder = comp_col1.empty()
            enhanced_placeholder = comp_col2.empty()

        # 熱力圖顯示區域（如果啟用）
        if st.session_state.heatmap_enabled:
            heatmap_placeholder = st.empty()

    # 右側統計區域 - 初始化右側UI元素
    with col2:
        st.subheader("即時統計")
        
        # 創建空容器用於實時更新統計數據
        stats_container = st.empty()  # 更新變數名稱，避免混淆
        
        # 趨勢圖表容器
        st.subheader("人數趨勢")
        trend_chart = st.empty()
        
        st.subheader("性能趨勢")
        fps_chart = st.empty()
        
        # 僅當啟用追蹤時創建軌跡分析圖表容器
        if tracking_enabled:
            st.subheader("軌跡分析")
            track_chart = st.empty()
        
        st.subheader("💡 智慧提示") # 在右側欄的統計信息下方添加
        cues_display_container = st.empty() # 使用 empty() 或 container()
        
    # 初始顯示統計資訊和圖表
    update_stats_display(stats_container, tracking_enabled)
    update_trend_charts(trend_chart, fps_chart, track_chart, tracking_enabled)
    
    # 如果正在處理視頻，則啟動檢測流程
    if st.session_state.processing and video_source is not None:
        # 將當前視頻源保存到狀態中 - 用於恢復處理
        st.session_state.video_state["source"] = video_source
        
        # 創建檢測器 - 優化：如果是繼續處理，重用之前的檢測器
        if st.session_state.processing_continuity["is_continuation"] and st.session_state.processing_continuity["detector"]:
            detector = st.session_state.processing_continuity["detector"]
            logger.info("重用之前的檢測器")
        else:
            detector = YOLODetector(
                model_name=selected_model, conf_threshold=conf_threshold
            )
            # 保存檢測器配置
            st.session_state.video_state["detector_config"] = {
                "model_name": selected_model,
                "conf_threshold": conf_threshold
            }
            
            logger.info("創建新檢測器: " + selected_model)

        # 創建追蹤器（如果啟用）- 優化：如果是繼續處理，重用之前的追蹤器
        tracker = None
        if tracking_enabled:
            if st.session_state.processing_continuity["is_continuation"] and st.session_state.processing_continuity["tracker"]:
                tracker = st.session_state.processing_continuity["tracker"]
                logger.info("重用之前的追蹤器")
            else:
                # 根據選擇的追蹤器創建相應的實例
                if selected_tracker == tracker_options[0]:  # ByteTracker
                    tracker = ByteTracker(
                        max_age=max_age,
                        min_hits=min_hits,
                        iou_threshold=iou_threshold,
                        high_threshold=high_threshold,
                        low_threshold=low_threshold,
                    )
                elif selected_tracker == tracker_options[1]:  # BoT-SORT
                    tracker = BoTSort(
                        max_age=max_age,
                        min_hits=min_hits,
                        iou_threshold=iou_threshold,
                        high_threshold=high_threshold,
                        low_threshold=low_threshold,
                        match_threshold=match_threshold,
                        use_appearance=use_appearance,
                        use_cmc=use_cmc,
                    )
                elif selected_tracker == tracker_options[2]:  # StrongSORT
                    tracker = StrongSort(
                        max_age=max_age,
                        min_hits=min_hits,
                        iou_threshold=iou_threshold,
                        reid_threshold=reid_threshold,
                        use_appearance=use_appearance,
                        use_ecc=use_ecc,
                        appearance_weight=appearance_weight,
                    )
                # 保存追蹤器配置
                st.session_state.video_state["tracker_config"] = {
                    "selected_tracker": selected_tracker,
                    "max_age": max_age,
                    "min_hits": min_hits,
                    "iou_threshold": iou_threshold
                }
                logger.info("創建新追蹤器: " + selected_tracker)

        # 保存顯示配置和增強配置
        st.session_state.video_state["display_config"] = {
            "show_bbox": show_bbox,
            "show_id": show_id,
            "show_trajectory": show_trajectory,
            "heatmap_enabled": heatmap_enabled
        }
        
        if enable_enhancement:
            st.session_state.video_state["enhancement_config"] = {
                "method": enhancement_method,
                "params": {
                    "gamma_value": gamma_value if enhancement_method == "伽瑪校正" else None,
                    "clip_limit": clip_limit if enhancement_method == "CLAHE (自適應直方圖均衡化)" else None,
                    "tile_size": tile_size if enhancement_method == "CLAHE (自適應直方圖均衡化)" else None
                }
            }

        try:
            # 打開視頻源 - 優化：根據處理狀態決定是新建還是繼續
            is_continuation = st.session_state.processing_continuity["is_continuation"]
            
            # 獲取視頻源
            cap = get_video_source(video_source)
            
            # 如果是繼續處理，設置到上次處理的位置
            if is_continuation and st.session_state.video_state["frame_position"] > 0:
                frame_position = st.session_state.video_state["frame_position"]
                logger.info("設置視頻到之前的位置: " + str(frame_position))
                
                # 檢查設置的幀位置是否超出視頻範圍
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if frame_position >= total_frames - 10:  # 留10幀的緩衝
                    logger.warning(f"幀位置 {frame_position} 接近視頻結束 {total_frames}，重置為開頭")
                    frame_position = 0
                
                # 設置視頻位置
                success = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                
                # 簡化驗證，減少不必要的日誌
                if not success:
                    logger.warning(f"視頻位置設置失敗，請求位置: {frame_position}")

            # 重置連續處理標記，下次rerun時會設置
            st.session_state.processing_continuity["is_continuation"] = False

            # 獲取視頻信息
            video_info = get_video_info(cap)
            
            # 直接從視頻信息獲取幀尺寸（如有）
            frame_height = video_info.get("height", 480)
            frame_width = video_info.get("width", 640)
            
            # 如果無法從視頻信息獲取尺寸，則嘗試讀取一幀（簡化邏輯）
            if (frame_height == 480 and frame_width == 640) and not is_continuation:
                # 保存當前位置，僅在必要時讀取幀獲取尺寸
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                ret, test_frame = cap.read()
                if ret:
                    frame_height, frame_width = test_frame.shape[:2]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                # 不再輸出警告，減少日誌量
            
            # 確保當前統計數據正確初始化
            if is_continuation and "person_count" in st.session_state.current_stats:
                # 繼續處理時，保留現有的統計數據
                pass
            else:
                # 新處理或無現有數據時重置
                st.session_state.current_stats = {
                    "person_count": 0,
                    "track_count": 0,
                    "fps": 0.0,
                    "frame_count": 0,
                    "last_update_time": time.time(),
                }

            # 處理視頻幀
            for frame, timestamp in video_frame_generator(cap, max_fps):
                if not st.session_state.processing:
                    break

                # 記錄開始時間
                start_time = time.time()

                # 保存當前幀位置 - 關鍵：確保rerun後能從正確位置繼續
                current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                st.session_state.video_state["frame_position"] = current_frame_pos
                
                # 保存原始幀用於對比顯示
                original_frame = frame.copy()

                # 當前幀計數加1（確保在處理開始時初始化）
                if st.session_state.current_stats["frame_count"] == 0:
                    # 首次運行時重設統計數據
                    st.session_state.current_stats = {
                        "person_count": 0,
                        "track_count": 0,
                        "fps": 0.0,
                        "frame_count": 0,
                        "last_update_time": time.time(),
                    }
                st.session_state.current_stats["frame_count"] += 1

                # 應用圖像增強（如果啟用）
                if enable_enhancement:
                    if enhancement_method == "CLAHE (自適應直方圖均衡化)":
                        # 應用CLAHE增強
                        frame = apply_histogram_equalization(
                            frame,
                            method="clahe",
                            clip_limit=clip_limit,
                            tile_size=(tile_size, tile_size),
                        )
                    elif enhancement_method == "基本直方圖均衡化":
                        # 應用基本直方圖均衡化
                        frame = apply_histogram_equalization(frame, method="basic")
                    elif enhancement_method == "伽瑪校正":
                        # 應用伽瑪校正
                        methods = [
                            {
                                "name": "gamma_correction",
                                "params": {"gamma": gamma_value},
                            }
                        ]
                        frame = enhance_image(frame, methods)

                # 執行檢測
                detections, _ = detector.detect(frame)

                # 計算檢測到的人數
                person_count = len(detections)

                # 執行追蹤（如果啟用）
                tracks = []
                track_count = 0
                if tracker and tracking_enabled:
                    tracks = tracker.update(detections, frame)
                    track_count = len(tracks)

                    # 將當前幀的軌跡添加到全局追蹤列表
                    if tracks:
                        for track in tracks:
                            if track not in st.session_state.all_tracks:
                                st.session_state.all_tracks.append(track)
                else:
                    # 如果未啟用追蹤，將檢測結果作為軌跡使用
                    # 為每個檢測結果添加臨時ID和位置信息
                    for i, det in enumerate(detections):
                        # 不再分配臨時track_id，只保留必要信息
                        # det['track_id'] = i  # 移除臨時ID
                        # 計算中心點位置用於音樂調整
                        x1, y1, x2, y2 = det["bbox"]
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        det["center"] = (center_x, center_y)
                        # 添加簡單的軌跡記錄（僅當前幀）
                        det["trajectory"] = [(int(center_x), int(center_y))]
                        # 確保有class_name欄位，默認為"person"
                        if "class_name" not in det and "class_id" in det:
                            # 根據COCO數據集，0是人類
                            if det["class_id"] == 0:
                                det["class_name"] = "person"
                            else:
                                det["class_name"] = "class_" + str(det["class_id"])
                        elif "class_name" not in det:
                            det["class_name"] = "person"  # 預設為人類

                # 更新統計 - 確保變數都已定義
                process_time = time.time() - start_time
                fps = 1.0 / process_time if process_time > 0 else 0

                # 更新當前統計數據（用於實時顯示）- 即時統計的數據來源
                st.session_state.current_stats["person_count"] = person_count  # 更新當前人數
                st.session_state.current_stats["track_count"] = track_count    # 更新追蹤ID數
                st.session_state.current_stats["fps"] = fps                    # 更新當前幀率
                
                # 存儲統計數據至歷史記錄 - 用於趨勢圖表
                st.session_state.stats["person_count"].append(person_count)
                st.session_state.stats["track_count"].append(track_count)
                st.session_state.stats["timestamps"].append(timestamp)
                st.session_state.stats["fps"].append(fps)

                # 關鍵改進：直接更新UI而不依賴rerun
                # 實時更新統計面板 - 每幀更新，確保即時性
                update_stats_display(stats_container, tracking_enabled)
                
                # 定期更新圖表 - 圖表更新較重，降低頻率以提高性能
                current_time = time.time()
                if "last_chart_update_time" not in st.session_state:
                    st.session_state.last_chart_update_time = 0
                
                # 每2秒更新一次圖表
                if current_time - st.session_state.last_chart_update_time >= 2.0:
                    update_trend_charts(trend_chart, fps_chart, track_chart, tracking_enabled)
                    st.session_state.last_chart_update_time = current_time
                
                # 檢查是否需要定期完全刷新頁面
                if "last_rerun_time" not in st.session_state:
                    st.session_state.last_rerun_time = 0
                    
                # 使用用戶配置的刷新間隔 - 如果啟用了刷新功能
                # 注意：refresh_interval在enable_full_refresh為False時為999999秒（實際上禁用刷新）
                if current_time - st.session_state.last_rerun_time >= refresh_interval:
                    # 僅在啟用定期刷新時執行rerun
                    if enable_full_refresh:
                        # 保存處理狀態，確保rerun後能繼續處理
                        st.session_state.temp_processing_state = True
                        st.session_state.processing_continuity["detector"] = detector
                        st.session_state.processing_continuity["tracker"] = tracker
                        st.session_state.processing_continuity["last_frame"] = frame.copy()
                        
                        # 保存當前幀位置，以便恢復
                        next_frame_position = current_frame_pos + 1
                        st.session_state.video_state["frame_position"] = next_frame_position
                        
                        st.session_state.last_rerun_time = current_time
                        logger.info(f"執行定期頁面重載，間隔{refresh_interval}秒，當前位置: {next_frame_position}")
                        st.rerun()
                    else:
                        # 僅更新時間戳但不執行rerun
                        st.session_state.last_rerun_time = current_time
                
                # === 關鍵修復：視頻顯示代碼 ===
                # 繪製結果
                if tracking_enabled and len(tracks) > 0:
                    # 追蹤模式：顯示所有追蹤結果
                    result_frame, st.session_state.id_colors = draw_tracks(
                        frame,
                        tracks,
                        show_bbox=show_bbox,
                        show_id=show_id,
                        show_trajectory=show_trajectory,
                        id_colors=st.session_state.id_colors,
                        show_all_detections=True,
                    )
                else:
                    # 僅檢測模式：顯示所有檢測結果，但不包含ID和軌跡
                    result_frame, _ = draw_tracks(
                        frame,
                        detections,
                        show_bbox=show_bbox,
                        show_id=False,  # 不顯示ID
                        show_trajectory=False,  # 不顯示軌跡
                    )

                # 添加統計信息到畫面
                result_frame = draw_stats(
                    result_frame, person_count, fps, position="top-right"
                )

                # 顯示結果 - 確保每幀都及時更新
                if show_comparison:
                    # 在原始幀上添加標題
                    cv2.putText(
                        original_frame,
                        "原始圖像",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    # 在結果幀上添加標題
                    cv2.putText(
                        result_frame,
                        "增強後圖像",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    # 顯示原始幀和增強後的幀
                    original_placeholder.image(
                        original_frame, channels="BGR", use_container_width=True
                    )
                    enhanced_placeholder.image(
                        result_frame, channels="BGR", use_container_width=True
                    )
                else:
                    # 只顯示結果幀
                    video_placeholder.image(
                        result_frame, channels="BGR", use_container_width=True
                    )

                # 更新熱力圖（如果啟用）
                if (
                    st.session_state.heatmap_enabled
                    and len(st.session_state.all_tracks) > 0
                ):
                    heatmap = create_heatmap(
                        st.session_state.all_tracks,
                        (frame_height, frame_width),
                        alpha=0.5,
                    )

                    # 疊加熱力圖和原始幀
                    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
                    heatmap_placeholder.image(
                        overlay, channels="BGR", use_container_width=True
                    )

                # --------------------------------------------------------------------
                # 智慧提示系統 - 核心邏輯整合點 (階段 1 和 後續階段2)
                # --------------------------------------------------------------------
                parsed_script_data = st.session_state.get('parsed_script')
                script_loaded_successfully = st.session_state.get('last_script_load_success', False)
                current_display_cues = [] # 本幀要顯示的cues

                if script_loaded_successfully and parsed_script_data:
                    # logger.debug(f"Script loaded. Timestamp: {timestamp}, Person Count: {person_count}")
                    # 階段 2 的核心匹配邏輯將在此處實現
                    # 現在是臨時佔位/初步測試邏輯：
                    for event in parsed_script_data:
                        event_time_start = event.get('time_start', float('inf'))
                        event_time_end = event.get('time_end', float('-inf'))
                        
                        # 時間匹配
                        if timestamp >= event_time_start and timestamp < event_time_end:
                            condition = event.get('trigger_condition', {})
                            cond_type = condition.get('type')
                            cond_op = condition.get('operator')
                            cond_val = condition.get('value')

                            # 人數條件匹配 (簡化版，後續階段會用 _check_trigger_condition 輔助函數)
                            event_triggered_by_person_count = False
                            if cond_type == 'person_count':
                                if cond_op == '==' and person_count == cond_val:
                                    event_triggered_by_person_count = True
                                elif cond_op == '>=' and person_count >= cond_val:
                                    event_triggered_by_person_count = True
                                elif cond_op == '<=' and person_count <= cond_val:
                                    event_triggered_by_person_count = True
                                elif cond_op == '>' and person_count > cond_val:
                                    event_triggered_by_person_count = True
                                elif cond_op == '<' and person_count < cond_val:
                                    event_triggered_by_person_count = True
                            
                            if event_triggered_by_person_count:
                                # logger.info(f"Event '{event.get('event_id', 'N/A')}' triggered at {timestamp:.2f}s with {person_count} people.")
                                for cue_obj in event.get('predicted_cues', []):
                                    # 階段 3 將會更詳細地處理 offset 和顯示格式
                                    # 目前只做簡單顯示
                                    cue_desc = cue_obj.get('cue_description', 'N/A')
                                    cue_offset = cue_obj.get('offset', 0)
                                    current_display_cues.append(
                                        f"事件 '{event.get('event_id', event.get('description', 'N/A')[:15])}': "
                                        f"預計 {cue_offset:.1f}s 後觸發 '{cue_desc[:30]}...' (觸發於 {timestamp:.1f}s, 人數 {person_count})"
                                    )
                
                # 更新 session_state 中的 active_display_cues，以便UI部分讀取
                st.session_state.active_display_cues = current_display_cues
                # --------------------------------------------------------------------
                # 智慧提示系統 - 邏輯結束
                # --------------------------------------------------------------------

                # 更新智慧提示UI (每一輪 UI 更新都會執行)
                # 這個位置要確保 cues_display_container 已經被定義 (即在 col2 中)
                # 並且 st.session_state.active_display_cues 已經被視訊處理迴圈更新
                
                # 這段邏輯應該放在主函數作用域下，確保每次streamlit rerun時都會更新提示區域
                # 但它依賴於 video_frame_generator 循環內部對 st.session_state.active_display_cues 的更新
                
                _active_cues_to_show = st.session_state.get('active_display_cues', [])
                _script_is_loaded = st.session_state.get('last_script_load_success', False)
                _parsed_script_exists = st.session_state.get('parsed_script') is not None
                _script_file_attempted = st.session_state.get('script_file_name') is not None

                # 更新 cues_display_container 的內容
                # 注意：cues_display_container 的定義需要在這段代碼執行前
                # 通常它在 st.columns 分配的列中定義 st.empty()
                
                # 我們將這段更新邏輯移到 col2 初始化 cues_display_container 之後，確保它每次都能更新
                # （實際上，Streamlit 的 st.empty() 或 st.container() 內容更新應該在其定義的列/塊內）
                # 因此，這段更新顯示的邏輯，正確的位置是在定義了 cues_display_container 之後，
                # 並且能夠訪問到 st.session_state.active_display_cues 的最新值。
                # 在 Streamlit 的執行模型中，腳本從頭到尾執行，所以可以放在定義 col2 之後的任何地方，
                # 或者，如果 cues_display_container 是在 col2 中，那麼就在 col2 的 with 塊中更新。

                # 已將 cues_display_container 的更新邏輯放置在 col2 的 st.subheader("💡 智慧提示") 之後
                # 以下是該更新邏輯的精煉版本，應放置在 `with col2:` 塊的末尾，或緊隨 `cues_display_container = st.empty()` 之後

                # (在 with col2: 內部，緊隨 cues_display_container = st.empty() 之後)
                # 這段代碼會被執行，前提是 st.session_state.active_display_cues 被主循環更新了
                # logger.debug(f"Updating cues display. Active cues: {st.session_state.get('active_display_cues')}")
                if _script_is_loaded and _parsed_script_exists:
                    if _active_cues_to_show:
                        cues_text_md = "\\n".join([f"- {cue_str}" for cue_str in _active_cues_to_show])
                        cues_display_container.markdown(f"**預測的技術提示:**\\n{cues_text_md}")
                    else:
                        # 只有在視頻處理中（st.session_state.processing 為 True）且沒有 cues 時，才顯示 "目前無提示"
                        # 否則在視頻未開始時，這個訊息也可能出現
                        if st.session_state.get("processing", False):
                             cues_display_container.info("目前無進行中的預測技術提示。")
                        else:
                             cues_display_container.info("開始視頻處理以查看提示。")
                elif _script_file_attempted and not _script_is_loaded: # 嘗試過載入但失敗
                    # 錯誤訊息已在側邊欄顯示，這裡可以不重複或顯示簡短提示
                    cues_display_container.warning("劇本載入失敗，無法提供提示。請檢查側邊欄錯誤訊息。")
                else: # 從未嘗試載入劇本
                    cues_display_container.info("請在側邊欄上傳劇本檔案以啟用智慧提示。")

            # 處理完成
            st.session_state.processing = False

        except Exception as e:
            logger.error("處理視頻時出錯: " + str(e))
            st.error("處理視頻時出錯: " + str(e))
            st.session_state.processing = False
    else:
        # 顯示佔位圖像
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder_img,
            "請選擇視頻源並點擊'開始檢測'",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        video_placeholder.image(
            placeholder_img, channels="BGR", use_container_width=True
        )

if __name__ == "__main__":
    main()
