from pathlib import Path
import sys
import json

import yaml

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
# RTSP
RTSP_ADDRESS = 'rtsp://admin:1q2w3e4r@10.0.0.114/Streaming/Channels/102'




# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_PATH = VIDEO_DIR / 'video_1.mp4'


# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'


def save_config(yaml_dict):
    with open('config.yaml', 'w', encoding='utf-8', ) as f:
        yaml.dump(yaml_dict, f)
        
def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = f.read()
        return yaml.load(cfg,Loader=yaml.FullLoader) 
    

# 存储数据到json文件
def save_to_json(data, file_path='config.json'):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 从json文件读取数据
def load_from_json(file_path='config.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data