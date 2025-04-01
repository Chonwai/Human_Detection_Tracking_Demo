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

## Cross-platform Installation and Running

This project supports multiple platforms including Mac (both Intel and Apple Silicon) and NVIDIA Jetson devices. The installation process has been simplified to automatically adapt to different platforms.

### 1. Quick Setup (Recommended for All Platforms)

```bash
# Clone the repository
git clone https://github.com/username/human_detection_tracking.git
cd human_detection_tracking

# Set up the environment and start the application
./run.sh --setup
```

The `run.sh` script will:
- Automatically detect your platform (Mac, Jetson, Linux, Windows)
- Set up a virtual environment with the correct dependencies
- Download necessary models (if not present)
- Configure platform-specific optimizations
- Start the application

### 2. Manual Setup

If you prefer to perform manual setup, follow these steps:

#### For Mac (Apple Silicon/Intel)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_mac.txt

# Start the application
streamlit run src/app.py
```

#### For NVIDIA Jetson

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the Jetson-specific PyTorch version
# For JetPack 5.1.1 (L4T R35.3.1) - adjust URLs based on your JetPack version
wget https://nvidia.box.com/shared/static/i8pukei49fgbf1qp2i8nsi113dtj2xj0.whl -O torch-2.1.0-cp310-cp310-linux_aarch64.whl
pip install torch-2.1.0-cp310-cp310-linux_aarch64.whl

# Install compatible torchvision
git clone --branch v0.16.0 https://github.com/pytorch/vision torchvision
cd torchvision
python setup.py install
cd ..

# Install Jetson-specific utilities
pip install jetson-stats

# Start the application with optimizations
sudo ./run_jetson.sh  # Using sudo for maximum performance
```

#### For Standard Linux/Windows

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the application
streamlit run src/app.py
```

### 3. Download YOLO Models

Make sure all YOLO model files are placed in the `data/models/` directory.

Recommended models include:
- YOLOv10-nano: Lightweight model suitable for devices with limited computing power
- YOLOv10-medium: Medium model balancing speed and accuracy
- YOLOv10/YOLOv12-large/xlarge: High accuracy models requiring more computing power

## Platform-specific Optimizations

### Mac (Apple Silicon)
- Automatically uses Metal Performance Shaders (MPS) for GPU acceleration
- Optimized memory usage for Apple Silicon
- Power mode selection for performance/battery balance

### NVIDIA Jetson
- CUDA acceleration with TensorRT optimization
- FP16 half precision for improved performance
- Resolution scaling to balance performance and accuracy
- Batch size adjustment for optimal throughput

### Diagnostic Tools

If you encounter performance issues, the project includes diagnostic tools:

```bash
# Check platform configuration
python -m src.core.platform_utils

# For Jetson-specific CUDA diagnostics
python scripts/check_cuda.py
```

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
- Platform-specific optimizations

## Troubleshooting

### Mac Issues
- If you encounter MPS-related errors, try disabling MPS acceleration in the advanced settings
- Ensure you have the latest version of PyTorch that supports MPS

### Jetson Issues
- Make sure you've installed the correct PyTorch version for your JetPack/L4T version
- If CUDA is not detected, run the diagnostic script and check the error messages
- Try using a smaller model (nano) and lower resolution for better performance

## Requirements

- Python 3.8+
- PyTorch (platform-specific version)
- OpenCV
- Streamlit
- NumPy
- Other dependencies listed in requirements*.txt files