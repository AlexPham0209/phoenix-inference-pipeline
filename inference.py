#!/usr/bin/env python
"""
Standalone Inference Script for Sign Language Recognition
Takes a video file or folder of images and returns recognized glosses.
"""

import numpy as np
import os
import cv2
import torch
from collections import OrderedDict
from slr_network import SLRModel
from translator import GlossTranslator
from utils import video_augmentation
import utils

VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]


def is_image_by_extension(file_path):
    """Check if file is an image by extension."""
    _, file_extension = os.path.splitext(file_path)
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    return file_extension.lower() in image_extensions


def load_video(video_path, max_frames_num=360):
    """
    Load video using OpenCV.

    Args:
        video_path: Path to video file
        max_frames_num: Maximum number of frames to sample

    Returns:
        List of frames in RGB format
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Get total frame count
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames uniformly
    if total_frame_num > max_frames_num:
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
    else:
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, total_frame_num, dtype=int
        )

    frame_idx = uniform_sampled_frames.tolist()

    # Read frames
    frames = []
    for idx in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB (OpenCV reads in BGR format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")

    return frames


def extract_glosses(prediction):
    """
    Extract gloss list from model prediction format.

    Args:
        prediction: Model output in format [[('GLOSS1', 0), ('GLOSS2', 1), ...]]

    Returns:
        List of gloss strings
    """
    if not prediction or len(prediction) == 0:
        return []
    return [item[0] for item in prediction[0]]


def infer(
    video_path,
    model_path=None,
    gloss_dict_path=None,
    device="cpu",
    max_frames_num=360,
    language="phoenix-T",
):
    """
    Run inference on a video or folder of images.

    Args:
        video_path: Path to video file or folder containing images
        model_path: Path to model checkpoint (default: ./checkpoint.pt)
        gloss_dict_path: Path to gloss dictionary (default: ./data/gloss_dict.npy)
        device: Device to use ('cpu' or '0' for GPU)
        max_frames_num: Maximum frames to sample from video
        language: Language dataset ('phoenix', 'phoenix-T', or 'csl')

    Returns:
        Dictionary with:
            - 'glosses': List of recognized gloss strings
            - 'raw_output': Raw model prediction format
            - 'gloss_string': Space-separated gloss string
            - 'text': Translated German sentence using GlossTranslator
    """
    # Set default paths
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "checkpoint.pt")
    if gloss_dict_path is None:
        gloss_dict_path = os.path.join(
            os.path.dirname(__file__), "data", "gloss_dict.npy"
        )
        
    # Validate paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(gloss_dict_path):
        raise FileNotFoundError(f"Gloss dictionary not found: {gloss_dict_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path not found: {video_path}")

    # Load gloss dictionary
    gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()

    # Load images/video
    if os.path.isdir(video_path):
        # Folder of images
        img_list = []
        for img_path in sorted(os.listdir(video_path)):
            cur_path = os.path.join(video_path, img_path)
            if is_image_by_extension(cur_path):
                img_list.append(cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB))
        if len(img_list) == 0:
            raise ValueError(f"No images found in folder: {video_path}")
    elif os.path.splitext(video_path)[-1] in VIDEO_FORMATS:
        # Video file
        img_list = load_video(video_path, max_frames_num)
    else:
        raise ValueError(
            f"Unsupported file format. Use video file ({VIDEO_FORMATS}) or folder of images."
        )

    # Preprocess
    transform = video_augmentation.Compose(
        [
            video_augmentation.CenterCrop(224),
            video_augmentation.Resize(1.0),
            video_augmentation.ToTensor(),
        ]
    )
    vid, _ = transform(img_list, None, None)
    vid = vid.float() / 127.5 - 1
    vid = vid.unsqueeze(0)

    # Calculate padding
    left_pad = 0
    last_stride = 1
    total_stride = 1
    kernel_sizes = ["K5", "P2", "K5", "P2"]
    for layer_idx, ks in enumerate(kernel_sizes):
        if ks[0] == "K":
            left_pad = left_pad * last_stride
            left_pad += int((int(ks[1]) - 1) / 2)
        elif ks[0] == "P":
            last_stride = int(ks[1])
            total_stride = total_stride * last_stride

    max_len = vid.size(1)
    video_length = torch.LongTensor(
        [int(np.ceil(vid.size(1) / total_stride) * total_stride + 2 * left_pad)]
    )
    right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
    max_len = max_len + left_pad + right_pad
    vid = torch.cat(
        (
            vid[0, 0][None].expand(left_pad, -1, -1, -1),
            vid[0],
            vid[0, -1][None].expand(max_len - vid.size(1) - left_pad, -1, -1, -1),
        ),
        dim=0,
    ).unsqueeze(0)

    # Setup device and model
    device_manager = utils.GpuDataParallel()
    device_manager.set_device(device)

    model = SLRModel(
        num_classes=len(gloss_dict) + 1,
        c2d_type="resnet18",
        conv_type=2,
        use_bn=1,
        gloss_dict=gloss_dict,
        loss_weights={"ConvCTC": 1.0, "SeqCTC": 1.0, "Dist": 25.0},
    )

    translator = GlossTranslator(
        model_path=os.path.join("checkpoints", "gloss_to_german.pt"),
        vocab_path=os.path.join("data", "vocab.json"),
        device=device,
    )

    state_dict = torch.load(
        model_path, map_location=device_manager.output_device, weights_only=False
    )["model_state_dict"]
    state_dict = OrderedDict(
        [(k.replace(".module", ""), v) for k, v in state_dict.items()]
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device_manager.output_device)

    if torch.cuda.is_available() and device_manager.output_device != "cpu":
        model.cuda()

    model.eval()

    # Run inference
    vid = device_manager.data_to_device(vid)
    vid_lgt = device_manager.data_to_device(video_length)

    with torch.no_grad():
        ret_dict = model(vid, vid_lgt, label=None, label_lgt=None)

    # Extract results
    raw_output = ret_dict["recognized_sents"]
    glosses = extract_glosses(raw_output)
    gloss_string = " ".join(glosses)

    return {
        "glosses": glosses,
        "raw_output": raw_output,
        "gloss_string": gloss_string,
        "text": translator.translate(gloss_string.upper()),
    }


def main():
    """Command-line interface for inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Sign Language Recognition Inference")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to video file or folder containing images",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint (default: ./checkpoint.pt)",
    )
    parser.add_argument(
        "--gloss_dict_path",
        type=str,
        default=None,
        help="Path to gloss dictionary (default: ./data/gloss_dict.npy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or GPU id like '0')",
    )
    parser.add_argument(
        "--max_frames_num",
        type=int,
        default=360,
        help="Maximum frames to sample from video",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="phoenix-T",
        choices=["phoenix", "phoenix-T", "csl"],
        help="Target sign language",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="string",
        choices=["string", "list", "json"],
        help="Output format: 'string' (space-separated), 'list', or 'json'",
    )

    args = parser.parse_args()

    try:
        result = infer(
            video_path=args.video_path,
            model_path=args.model_path,
            gloss_dict_path=args.gloss_dict_path,
            device=args.device,
            max_frames_num=args.max_frames_num,
            language=args.language,
        )

        if args.output_format == "string":
            print(result["gloss_string"])
        elif args.output_format == "list":
            print(result["glosses"])
        elif args.output_format == "json":
            import json

            print(
                json.dumps(
                    {
                        "glosses": result["glosses"],
                        "gloss_string": result["gloss_string"],
                    },
                    indent=2,
                )
            )

    except Exception as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())


# gloss_dict_path = os.path.join(os.path.dirname(__file__), 'data', 'gloss_dict.npy')
# gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()

# print(gloss_dict)

# translator = GlossTranslator(
#     model_path=os.path.join("checkpoints", "gloss_to_german.pt"),
#     vocab_path=os.path.join("data", "vocab.json"),
# )
# print(translator.translate("DAZWISCHEN FREUNDLICH"))
