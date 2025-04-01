#!/bin/bash
# 智能環境設置腳本 - 自動識別平台並配置適當的環境
# 用法: ./scripts/setup_env.sh

set -e  # 遇到錯誤時停止

# 顯示標題
echo "================================================"
echo "    智能人體檢測系統 - 跨平台環境設置腳本       "
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

echo ""
echo "正在設置 $PLATFORM 環境..."

# 創建虛擬環境（可選）
if [[ ! -d "venv" ]]; then
    echo "創建虛擬環境..."
    python -m venv venv
    echo "虛擬環境已創建"
fi

# 激活虛擬環境
if [[ "$PLATFORM" == "windows" ]]; then
    echo "激活Windows虛擬環境..."
    source venv/Scripts/activate
else
    echo "激活虛擬環境..."
    source venv/bin/activate
fi

# 安裝通用依賴
echo "安裝通用依賴..."
pip install -r requirements.txt

# 基於平台安裝特定依賴
if [[ "$PLATFORM" == "mac" ]]; then
    echo "安裝Mac特定依賴..."
    pip install -r requirements_mac.txt
    
    # 檢查PyTorch是否已安裝並支持MPS
    echo "檢查PyTorch MPS支持..."
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'MPS可用: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
    
elif [[ "$PLATFORM" == "jetson" ]]; then
    echo "對於Jetson平台，請按照requirements_jetson.txt中的說明手動安裝PyTorch/torchvision"
    echo "其他Jetson依賴將被安裝..."
    pip install jetson-stats
    
    # 檢查PyTorch CUDA支持
    echo "檢查PyTorch CUDA支持..."
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')" || echo "PyTorch未正確安裝或缺少CUDA支持"
    
    # 設置Jetson特定環境變量
    echo "設置Jetson特定環境變量..."
    echo 'export OPENBLAS_CORETYPE=ARMV8' >> venv/bin/activate
    echo 'export PYTHONUNBUFFERED=1' >> venv/bin/activate
    echo "環境變量已設置"
elif [[ "$PLATFORM" == "linux" ]]; then
    echo "安裝標準Linux依賴..."
    # 這裡可以添加Linux特定依賴
    pip install -r requirements.txt
fi

echo ""
echo "環境設置完成！"
echo "要激活環境，請運行: source venv/bin/activate"
echo "================================================" 