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