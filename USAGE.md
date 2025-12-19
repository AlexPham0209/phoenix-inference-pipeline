# Quick Start Guide

## Installation

1. Extract the inference package to your desired location
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Examples

### Example 1: Video File
```bash
python inference.py --video_path /path/to/your/video.mp4
```

Output:
```
MORGEN SONNE
```

### Example 2: Image Folder
```bash
python inference.py --video_path /path/to/image/folder
```

### Example 3: Using Python API
```python
from inference import infer

result = infer("/path/to/video.mp4", device="cpu")
print(result['gloss_string'])
```

### Example 4: GPU Inference
```bash
python inference.py --video_path video.mp4 --device 0
```

## Package Structure

```
inference_package/
├── inference.py          # Main inference script
├── checkpoint.pt         # Model weights
├── slr_network.py        # Model architecture
├── requirements.txt      # Python dependencies
├── README.md             # Full documentation
├── modules/              # Model components
│   ├── resnet.py
│   ├── BiLSTM.py
│   ├── tconv.py
│   └── criterions.py
├── utils/                # Utility functions
│   ├── decode.py
│   ├── device.py
│   └── video_augmentation.py
└── data/                 # Data files
    └── gloss_dict.npy
```

## Notes

- Works on both CPU and GPU
- Supports video files (.mp4, .avi, .mov, .mkv) and image folders
- Model is trained on PHOENIX14-T (German Sign Language)
- Default device is CPU (use `--device 0` for GPU)

