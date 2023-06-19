from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO(yaml_path)  # build a new model from YAML
# model = YOLO(yaml_path).load(weights_path)  # build from YAML and transfer weights
# dataset_yaml = '/workspaces/yolov8-streamlit-detection-tracking/datasets/FS/yamls/dataset.yaml'
# model.train(data=dataset_yaml, epochs=100, imgsz=640, workers=0, batch=6)


# yaml_path = '/workspaces/yolov8-streamlit-detection-tracking/runs/detect/train9/args.yaml'
# weights_path = '/workspaces/yolov8-streamlit-detection-tracking/runs/detect/train9/weights/last.pt'
# model = YOLO(weights_path)

# Train the model
dataset_yaml = '/workspaces/yolov8-streamlit-detection-tracking/datasets/PS/yamls/dataset.yaml'
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model.train(data=dataset_yaml, epochs=100, imgsz=640, workers=0, resume=True, batch=6)


### 训练分类模型
# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
# model = YOLO('/workspaces/yolov8-streamlit-detection-tracking/runs/classify/train3/weights/last.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='/workspaces/yolov8-streamlit-detection-tracking/datasets/mask-nomask-unknow/images', epochs=1000, imgsz=224)