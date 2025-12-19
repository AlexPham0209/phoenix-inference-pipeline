# Sign Language Recognition Inference Package

Standalone inference package for continuous sign language recognition. This package contains all necessary files to run inference on videos or image sequences.

## Contents

- `inference.py` - Main inference script
- `checkpoint.pt` - Pre-trained model weights
- `slr_network.py` - Model architecture
- `modules/` - Model components (ResNet, BiLSTM, TemporalConv, etc.)
- `utils/` - Utility functions (decoding, device management, preprocessing)
- `data/gloss_dict.npy` - Gloss dictionary for PHOENIX14-T dataset

## Requirements

Install the following Python packages:

```bash
pip install torch torchvision numpy opencv-python Pillow scipy
```

**Note:** 
- PyTorch version >= 1.8 is required
- For GPU inference, install PyTorch with CUDA support
- For CPU inference, the CPU-only version works fine

## Usage

### Command Line Interface

```bash
# Basic usage (video file)
python inference.py --video_path /path/to/video.mp4

# With options
python inference.py \
    --video_path /path/to/video.mp4 \
    --device cpu \
    --max_frames_num 360 \
    --output_format string

# Image folder
python inference.py --video_path /path/to/image/folder

# Custom model/gloss dict paths
python inference.py \
    --video_path /path/to/video.mp4 \
    --model_path ./checkpoint.pt \
    --gloss_dict_path ./data/gloss_dict.npy
```

### Python API

```python
from inference import infer

# Run inference
result = infer(
    video_path="/path/to/video.mp4",
    device="cpu",  # or "0" for GPU
    max_frames_num=360
)

# Access results
print(result['gloss_string'])  # Space-separated glosses
print(result['glosses'])        # List of glosses
print(result['raw_output'])    # Raw model output
```

## Arguments

- `--video_path` (required): Path to video file or folder containing images
- `--model_path` (optional): Path to model checkpoint (default: ./checkpoint.pt)
- `--gloss_dict_path` (optional): Path to gloss dictionary (default: ./data/gloss_dict.npy)
- `--device` (optional): Device to use - 'cpu' or GPU id like '0' (default: 'cpu')
- `--max_frames_num` (optional): Maximum frames to sample from video (default: 360)
- `--language` (optional): Language dataset - 'phoenix', 'phoenix-T', or 'csl' (default: 'phoenix-T')
- `--output_format` (optional): Output format - 'string', 'list', or 'json' (default: 'string')

## Supported Formats

**Video files:**
- `.mp4`
- `.avi`
- `.mov`
- `.mkv`

**Image formats:**
- `.jpg`, `.jpeg`
- `.png`
- `.gif`
- `.bmp`

## Output

The script returns recognized glosses (sign language words). Example output:

```
MORGEN SONNE
```

Or as a list:
```python
['MORGEN', 'SONNE']
```

## Notes

- The model is trained on PHOENIX14-T dataset (German Sign Language)
- Inference works on both CPU and GPU
- For best performance, use GPU if available
- The model processes videos by uniformly sampling frames (up to max_frames_num)

## Troubleshooting

1. **CUDA errors**: If you get CUDA errors, use `--device cpu`
2. **File not found**: Ensure all paths are correct and relative to the package directory
3. **Memory errors**: Reduce `--max_frames_num` if running out of memory
4. **Empty output**: Check that the video/images are valid and readable

## License

See the original repository for license information.

