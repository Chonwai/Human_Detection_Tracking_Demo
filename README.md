# Human Detection and Tracking System

An intelligent human detection and tracking system based on YOLO series models, supporting multiple input sources and tracking algorithms.

## Project Structure

```
human_detection_tracking/
├── src/                    # Source code directory
│   ├── core/              # Core functionality implementation
│   ├── utils/             # Utility functions
│   ├── models/            # Model-related code
│   ├── config/            # Configuration files
│   └── interface/         # User interface code
├── tests/                 # Test code
├── data/                  # Data directory
│   ├── videos/           # Test videos
│   ├── models/           # Model files
│   └── output/           # Output results
├── docs/                 # Documentation
└── scripts/              # Utility scripts
```

## Installation and Running

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download YOLO Models

Make sure all YOLO model files are placed in the `data/models/` directory.

Recommended models include:
- YOLOv10-nano: Lightweight model suitable for devices with limited computing power
- YOLOv10-medium: Medium model balancing speed and accuracy
- YOLOv10/YOLOv12-large/xlarge: High accuracy models requiring more computing power

### 3. Running the Application

To start the application, use the Streamlit command:

```bash
streamlit run src/app.py
```

The application will be accessible through your web browser at `http://localhost:8501` by default.

### 4. Running on NVIDIA Jetson

When deploying on NVIDIA Jetson platforms, special considerations are needed to achieve optimal performance:

#### Install the Correct PyTorch Version

NVIDIA Jetson requires a specific PyTorch version built for its ARM architecture. Do not use the standard pip-installed PyTorch:

```bash
# First remove any existing PyTorch installations
pip uninstall torch torchvision

# Install the Jetson-compatible PyTorch version
# Check https://forums.developer.nvidia.com/t/pytorch-for-jetson/ for the latest version
```

You can find the latest compatible wheels for your JetPack version at NVIDIA's forums. For example:

```bash
# For JetPack 5.1.1 (L4T R35.3.1)
wget https://nvidia.box.com/shared/static/i8pukei49fgbf1qp2i8nsi113dtj2xj0.whl -O torch-2.1.0-cp310-cp310-linux_aarch64.whl
pip install torch-2.1.0-cp310-cp310-linux_aarch64.whl

# Install compatible torchvision
git clone --branch v0.16.0 https://github.com/pytorch/vision torchvision
cd torchvision
python setup.py install
```

#### Performance Optimization Tips

1. Use lightweight models (nano or small) for better performance.
2. Run the diagnostic script to verify CUDA is working properly:
   ```bash
   python scripts/check_cuda.py
   ```
3. Set appropriate environment variables:
   ```bash
   export OPENBLAS_CORETYPE=ARMV8
   ```
4. Enabling TensorRT optimization can significantly improve performance
5. Lower the resolution and reduce the maximum FPS in the settings to achieve smoother performance

## Features

- **Multiple Input Sources**: Support for video files, webcams, RTSP streams, and demo videos
- **Tracking Algorithms**: Integration with ByteTracker, BoT-SORT, and StrongSORT
- **Real-time Statistics**: Display of detected person count, FPS, and tracking information
- **Visualization**: Bounding boxes, IDs, trajectories, and heatmaps
- **Image Enhancement**: CLAHE, basic histogram equalization, and gamma correction

## Customization

The application provides various configuration options through the sidebar:
- Input source selection
- Detection model and confidence threshold
- Tracking algorithm and parameters
- Display options (bounding boxes, IDs, trajectories, heatmaps)
- Image enhancement methods
- Performance settings (FPS limit)

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Streamlit
- NumPy
- Other dependencies listed in requirements.txt