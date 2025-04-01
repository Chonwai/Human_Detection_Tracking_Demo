#!/bin/bash
# Jetson平台優化啟動腳本

# 顯示腳本標題
echo "================================"
echo "人體檢測與追蹤系統 - Jetson啟動腳本"
echo "================================"

# 設置CUDA和PyTorch相關環境變量
echo "正在設置環境變量..."
export OPENBLAS_CORETYPE=ARMV8
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# 檢查是否安裝了正確版本的PyTorch
echo "檢查PyTorch安裝..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')" || { 
    echo "PyTorch導入失敗或不可用。請檢查安裝。"; 
    echo "請參考README中的Jetson安裝指南。";
    exit 1; 
}

# 檢查可用的YOLO模型
echo "檢查模型文件..."
if [ ! -d "data/models" ]; then
    mkdir -p data/models
    echo "已創建data/models目錄，請確保下載模型文件。"
fi

MODEL_COUNT=$(ls -1 data/models/*.pt 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "警告: 未找到YOLO模型文件。請將模型文件（.pt）放置在data/models目錄中。"
    echo "建議優先使用yolov10n.pt或其他nano/small模型以獲得最佳性能。"
fi

# 運行診斷腳本
echo "運行CUDA診斷..."
python scripts/check_cuda.py

# 優化Jetson性能
echo "應用Jetson性能優化..."

# 啟用最大性能模式 (需要sudo權限)
if [ $(id -u) -eq 0 ]; then
    echo "設置最大性能模式..."
    
    # 如果是Jetson Nano，設置6W或10W電源模式
    if [ -f /sys/devices/platform/p3448_battery/power_supply/battery/current_now ]; then
        echo "檢測到Jetson Nano。設置10W高性能模式..."
        /usr/sbin/nvpmodel -m 0
        echo 1 > /sys/devices/system/cpu/cpu0/online
        echo 1 > /sys/devices/system/cpu/cpu1/online
        echo 1 > /sys/devices/system/cpu/cpu2/online
        echo 1 > /sys/devices/system/cpu/cpu3/online
    fi
    
    # 設置最大GPU和CPU頻率
    if [ -f /sys/devices/gpu.0/devfreq/17000000.gp10b/min_freq ]; then
        cat /sys/devices/gpu.0/devfreq/17000000.gp10b/max_freq > /sys/devices/gpu.0/devfreq/17000000.gp10b/min_freq
        echo "已設置GPU最大頻率"
    fi
    
    # 對於TX1/TX2/Xavier/Orin等其他Jetson設備
    if [ -f /sys/kernel/debug/tegra_cpufreq/cpu_freq_min ]; then
        echo "Jetson設備性能優化..."
        # 最大化CPU性能
        cat /sys/kernel/debug/tegra_cpufreq/cpu_freq_max > /sys/kernel/debug/tegra_cpufreq/cpu_freq_min
    fi
    
    # 嘗試運行jetson_clocks腳本 (如果存在)
    if [ -f /usr/bin/jetson_clocks ]; then
        echo "運行jetson_clocks設置最大性能..."
        /usr/bin/jetson_clocks
    fi
else
    echo "注意：需要root權限才能進行最大性能設置。請考慮使用sudo運行。"
fi

# 清理緩存，釋放內存
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

# 顯示系統信息
echo "系統信息："
free -h
nvidia-smi 2>/dev/null || echo "nvidia-smi不可用，使用tegrastats檢查GPU"

# 啟動應用
echo "啟動人體檢測與追蹤系統..."
echo "應用將在http://localhost:8501中運行"
echo "================================"

# 運行應用
streamlit run src/app.py 