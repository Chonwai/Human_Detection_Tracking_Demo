# 人體檢測與追蹤系統

智能人體檢測與追蹤系統，基於YOLO系列模型實現，支持多種輸入源和追蹤算法。

## 項目結構

```
human_detection_tracking/
├── src/                    # 源代碼目錄
│   ├── core/              # 核心功能實現
│   ├── utils/             # 工具函數
│   ├── models/            # 模型相關代碼
│   ├── config/            # 配置文件
│   └── interface/         # 用戶界面代碼
├── tests/                 # 測試代碼
├── data/                  # 數據目錄
│   ├── videos/           # 測試視頻
│   ├── models/           # 模型文件
│   └── output/           # 輸出結果
├── docs/                 # 文檔
└── scripts/              # 工具腳本
```

## 安裝與運行

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 安裝 Git LFS

由於本項目包含大型模型文件，需要使用 Git LFS (Large File Storage) 管理，請按照以下步驟安裝：

**macOS**:
```bash
brew install git-lfs
```

**Ubuntu/Debian**:
```bash
sudo apt-get install git-lfs
```

**Windows**:
- 下載並安裝 Git LFS: https://git-lfs.com/

安裝後，在項目目錄中初始化 Git LFS:
```bash
git lfs install
```

### 3. 運行應用

```bash
python src/app.py
```

## 模型路徑說明

所有YOLO模型文件應存放在 `data/models/` 目錄下，推薦使用的模型有：
- YOLOv10-nano: 輕量級模型，適合低算力設備
- YOLOv10-medium: 平衡速度和精度的中等模型
- YOLOv10/YOLOv12-large/xlarge: 高精度模型，需要較高算力