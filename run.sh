#!/bin/bash
# 智能啟動腳本 - 可在Mac和Jetson平台上運行
# 用法: ./run.sh [--setup]

# 設置錯誤處理
set -e

# 顯示標題
echo "================================================"
echo "     人體檢測與追蹤系統 - 智能啟動腳本          "
echo "================================================"

# 檢測平台
PLATFORM="unknown"
IS_JETSON=false

# macOS檢測
if [[ "$(uname)" == "Darwin" ]]; then
    PLATFORM="mac"
    echo "檢測到Mac平台"
    
    # 檢查是否為Apple Silicon
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo "檢測到Apple Silicon芯片"
    else
        echo "檢測到Intel芯片"
    fi
# Linux檢測
elif [[ "$(uname)" == "Linux" ]]; then
    echo "檢測到Linux平台"
    
    # 檢查是否為Jetson
    if [[ -f "/etc/nv_tegra_release" ]]; then
        PLATFORM="jetson"
        IS_JETSON=true
        echo "檢測到NVIDIA Jetson平台"
        cat /etc/nv_tegra_release
    else
        echo "檢測到標準Linux平台"
        PLATFORM="linux"
    fi
# Windows檢測
elif [[ "$(uname)" == "MINGW"* ]] || [[ "$(uname)" == "MSYS"* ]]; then
    PLATFORM="windows"
    echo "檢測到Windows平台"
else
    echo "未知平台: $(uname)"
fi

# 處理命令行參數
SETUP_MODE=false
for arg in "$@"; do
    case $arg in
        --setup)
            SETUP_MODE=true
            shift
            ;;
        *)
            # 未知參數
            ;;
    esac
done

# 如果是設置模式，運行環境設置腳本
if [[ "$SETUP_MODE" == "true" ]]; then
    echo "運行環境設置腳本..."
    ./scripts/setup_env.sh
    exit 0
fi

# 檢查虛擬環境是否存在
if [[ ! -d "venv" ]]; then
    echo "未找到虛擬環境。是否要創建它？[y/N]"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "運行環境設置腳本..."
        ./scripts/setup_env.sh
    else
        echo "跳過虛擬環境創建。"
    fi
fi

# 激活虛擬環境（如果存在）
if [[ -d "venv" ]]; then
    echo "激活虛擬環境..."
    if [[ "$PLATFORM" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
fi

# 平台特定啟動準備
if [[ "$PLATFORM" == "jetson" ]]; then
    echo "準備Jetson平台..."
    
    # 設置環境變量
    export OPENBLAS_CORETYPE=ARMV8
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=0
    
    # 檢查CUDA可用性
    echo "檢查CUDA可用性..."
    if python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"; then
        echo "CUDA檢查通過"
    else
        echo "警告: CUDA可能不可用。請確保已安裝正確的PyTorch版本。"
    fi
    
    # 優化系統性能（如果有root權限）
    if [[ "$(id -u)" -eq 0 ]]; then
        echo "優化系統性能..."
        
        # 嘗試運行jetson_clocks腳本 (如果存在)
        if [ -f /usr/bin/jetson_clocks ]; then
            echo "運行jetson_clocks設置最大性能..."
            /usr/bin/jetson_clocks
        fi
        
        # 清理緩存
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    else
        echo "提示: 以root身份運行可獲得最佳性能 (sudo ./run.sh)"
    fi
    
elif [[ "$PLATFORM" == "mac" ]]; then
    echo "準備Mac平台..."
    
    # 檢查MPS可用性
    echo "檢查MPS可用性..."
    python -c "import torch; print(f'MPS可用: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
    
    # 建議關閉其他應用以提升性能
    echo "提示: 關閉其他應用程序可以提高性能。"
fi

# 啟動應用
echo ""
echo "啟動人體檢測與追蹤系統..."
echo "應用將在http://localhost:8501中運行"
echo "================================================"

# 運行Streamlit應用
streamlit run src/app.py 