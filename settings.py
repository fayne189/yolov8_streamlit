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


        
def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = f.read()
        return yaml.load(cfg,Loader=yaml.FullLoader) 
    

def save_config(yaml_dict, file_path='config.yaml', overwrite=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        old_yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    if overwrite:
        old_yaml_dict = yaml_dict
    else:
        old_yaml_dict.update(yaml_dict)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(old_yaml_dict, f)

# 存储数据到json文件
def save_to_json(data, file_path='config.json', overwrite=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    if overwrite:
        old_data = data
    else:
        old_data.update(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(old_data, f, ensure_ascii=False, indent=4)

# 从json文件读取数据
def load_from_json(file_path='config.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data