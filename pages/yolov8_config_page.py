# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving

import time
import os
import sys
import queue
import traceback
import cv2
from ultralytics import YOLO
import streamlit as st
import numpy as np
import threading
sys.path.append('..')
import settings
from src.video_converter import video_converte
from logger import get_logger
from src.camera import VideoCapture


logger = get_logger(__name__)

class Component(object):
    def __init__(self, name) -> None:
        self.name = name
        self.in_queue = queue.Queue()
        self.out_queue = None
        self.streaming = False
        self.timeout=5
        self.thread_event = threading.Event()
        self.last_result = None
        self.source_component = None
        
    def link(self, component)->None:
        # link the output queue to next component
        # find root source and set source name 
        self.out_queue = component.in_queue
        if self.source_component is None:
            self.source_component = self.name
        component.source_component = self.source_component
    
    def start_thread(self):
        if self.streaming:
            print('thread already started')
        else:
            self.streaming = True
            self.thread = threading.Thread(target=self._run_worker_thread, name=self.name, daemon=True)
            self.thread.start()
        
    def stop_thread(self):
        self.streaming = False
        self.thread.join()
        self.thread = None
         
    def _worker_thread(self):
        try:
            while True:
                if self.streaming == False:
                    break
                item = self.in_queue.get()
                while not self.in_queue.empty():
                    item = self.in_queue.get_nowait()
                item = self.handle(item)
                self.last_result = item
                self.thread_event.set()
                time.sleep(0)
                if self.out_queue is not None and item is not None:
                    self.out_queue.put(item)
        except Exception as e:
            raise e

    def _run_worker_thread(self):
        try:
            self._worker_thread()
        except Exception:
            logger.error(f"Error occurred in the {self.name} thread:")

            exc_type, exc_value, exc_traceback = sys.exc_info()
            for tb in traceback.format_exception(exc_type, exc_value, exc_traceback):
                for tbline in tb.rstrip().splitlines():
                    logger.error(tbline.rstrip())
    
    def get_result(self):
        self.thread_event.wait()
        self.thread_event.clear()
        return self.last_result
        
    def result_gen(self):
        while True:
            yield self.get_result()
            
    def handle(self, item):
        item_dict = {self.source_component: item}
        return item_dict
         
class Yolo(Component):
    def __init__(self, name, model_path, is_tracker) -> None:
        super().__init__(name)
        self.model = self.load_model(model_path)
        self.is_tracker = is_tracker
        self.confidence = 0.5
        
    def load_model(self, model_path):
        """
        Loads a YOLO object detection model from the specified model_path.

        Parameters:
            model_path (str): The path to the YOLO model file.

        Returns:
            A YOLO object detection model.
        """
        model = YOLO(model_path)
        return model
        
    def handle(self, item_dict):
        # images = item_dict[self.source_component]
        images = list(item_dict.values())
        sources = list(item_dict.keys())
        if self.is_tracker:
            # res = model.track(image, conf=conf, tracker='botsort.yaml')
            res = self.model.track(images, conf=self.confidence, tracker='bytetrack.yaml')
        else:
            res = self.model.predict(images, conf=self.confidence)
        # return super().handle(res)
        return dict(zip(sources, res))
    
    def set_confidence(self, confidence: int):
        self.confidence = confidence
        
    def set_tracker(self, on_off: bool):
        self.is_tracker = on_off
    
class OSD(Component):
    def __init__(self, name) -> None:
        super().__init__(name)

    def handle(self, item_dict):
        res = item_dict[self.source_component]
        plotted_res = [re.plot() for re in res]
        return super().handle(plotted_res)

class Source(Component):
    def __init__(self, name, video_path) -> None:
        super().__init__(name)
        self.video_path = video_path
        self.set_video_path(video_path)
        self.cap = None
        
    def handle(self, video_path):
        if self.cap and video_path is None: # 停止
            self.cap.release()
            return
                    
        if self.cap:
            if self.video_path != video_path:  # 路径有改变的时候
                self.cap.release()
                self.cap = None
                self.video_path = video_path
        
            if self.cap.isOpened():
                frame = self.cap.read()
                self.set_video_path(video_path)
                return super().handle(frame)
        else:
            self.cap = VideoCapture(video_path)    # 重连
        self.set_video_path(video_path)
        return 
    
    def set_video_path(self, video_path):
        self.in_queue.put(video_path)

class StreamMuxer(Component): # 合流
    def __init__(self, name) -> None:
        self.linked = {}
        super().__init__(name)
        
    
    def handle(self, item):
        
        source_name, image = item.items()
        
        return super().handle(item)

class Logic(Component):
    def __init__(self, name, cfg) -> None:
        super().__init__(name)
        self.subjects = []  # points, lines, areas
        self.objects = []  # yolo detections
        self.cfg = {}
        
    class Object(object):
        def __init__(self, name, xywh, id=None, *args) -> None:
            self.name = name
            self.xywh = xywh
            self.id = id 
            self.args = args
            
        def to_certerbox(self):
            center_x = self.xywh[0] + self.xywh[2]/2
            center_y = self.xywh[1] + self.xywh[3]/2
            return center_x, center_y
         
         
         
    def handle(self, item_dict):
        item = item_dict[self.source_component]
        print(item)

        return super().handle(item)
    
    def define_objects(self, item):
        # 将yolov8的结果变成主对象
        for i in range(item.boxes.shape[0]):
            cls = item.boxes.cls[i]
            # 根据yolo的cls来定义object对象
            object_dict = self.cfg['objects'].get(cls)
            class_name = object_dict['class_name']
            xywh = item.boxes.xywh[i]
            if item.boxes.id is not None:
                id = item.boxes.id[i]
            obj = object(class_name, xywh, id=None)
            self.objects.append(obj)
        pass
    
    def define_subjects(self):
        # 根据cfg来定义 点，线面
        pass
    
    def relations(self):
        # 计算objects和subjects的关系(距离，方位，是否交集)
        pass
    
    def object_status(self):
        # 目标的状态，（颜色，属性）
        pass
    
    
    
class FakeSink(Component):
    def __init__(self, name) -> None:
        super().__init__(name)

    def handle(self, item):
        print(item)
        return 
    
class ProcessPipeline(object):
    def __init__(self):
        self.pipeline = []
        self.component_dict = {}

    def add_component(self, comp):
        self.pipeline.append(comp)
        self.component_dict.update({comp.name:comp})
        
    def get_component(self, comp_name):
        return self.component_dict.get(comp_name)
        
            
    def start_pipeline(self):
        for component in self.pipeline:
            component.start_thread()


# # 启动子进程，获取画面数据
# t = threading.Thread(target=get_frame, args=(q,))
# t.start()

# 在 Streamlit 中显示画面数据
# stframe = st.empty()
# async def update_frame():
#     while True:
#         if not q.empty():
#             frame = q.get()
#             stframe.image(frame, channels="BGR")
#         await asyncio.sleep(0.01)

@st.cache_resource
def init_pipeline():
    model_path = settings.DETECTION_MODEL
    video_path = 'temp/demo_.mp4'
    pipeline = ProcessPipeline()
    
    source = Source('source', video_path)
    source2 = Source('source2', video_path)
    
    yolo = Yolo('yolo', model_path, is_tracker=False)
    logic = Logic('osd')
    osd = OSD('osd')
    fakesink = FakeSink('fakesink')
    
    # source.link(fakesink.in_queue)
    source.link(yolo)
    yolo.link(logic)
    # logic.link(osd.in_queue)
    
    
    # osd.link(fakesink.in_queue)
    # source.link(sink)
    pipeline.add_component(source)
    pipeline.add_component(yolo)
    pipeline.add_component(logic)
    # pipeline.add_component(fakesink)
    # pipeline.add_component(sink)
    return pipeline

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Render the readme as markdown using st.markdown.
    # readme_text = st.markdown('Readme.md')
    # if st.button('init pipeline'):
    pipeline = init_pipeline()
    pipeline.start_pipeline()
    stframe = st.empty()
    if st.button('Play'):
        osd = pipeline.get_component('osd')
        for res in osd.result_gen():
            stframe.image(res[0])
            time.sleep(0.05)
        # cols = st.columns(len(plotted_res))
        # for i, col in enumerate(cols):
        #     with col:
        #         st.image(plotted_res[i],
        #                 caption='Detected Frame',
        #                 channels="BGR",
        #                 use_column_width=True
        #                 )
        #         time.sleep(0.05)
    # if st.button('off video capture'):
    #     source = pipeline.get_component('source')
    #     source.set_video_path(None)
    #     source.terminate_thread()
    # if st.button('on video capture'):
    #     source = pipeline.get_component('source')
    #     source.start_thread()
        # source.set_video_path('temp/demo_.mp4')
    # run_the_app()



def save_upload(uploadedfile):
    save_temp_path = os.path.join("temp", uploadedfile.name)
    with open(save_temp_path,"wb") as f:
        f.write(uploadedfile.getbuffer())
    return save_temp_path
         
@st.cache_resource
def get_video_frames(file_name):
    frames = []
    vid_cap = cv2.VideoCapture(file_name)
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            frames.append(image)
        else:
            vid_cap.release()
            break
    return frames


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def draw_image_with_bbox_yolov8(image, model, conf, tracker=False):
    # Predict the objects in the image using YOLOv8 model
    if tracker:
        # res = model.track(image, conf=conf, tracker='botsort.yaml')
        res = model.track(image, conf=conf, tracker='bytetrack.yaml')
    else:
        res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    # st_frame.image(res_plotted,
    #                caption='Detected Frame',
    #                channels="BGR",
    #                use_column_width=True
    #                )
    # await asyncio.sleep(0.05)
    return res_plotted


def run_the_app():
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.file_uploader(
        label="Choose a video...",
        type=['mp4']
    )
    converted_video_path = 'temp/demo_.mp4'
    
    if source_video:
        video_path = save_upload(source_video)
        converted_video_path = video_path.replace('.mp4', '_.mp4')
    # 加载 yolov5模型
    model = load_model(settings.DETECTION_MODEL)
    try:
        if not os.path.exists(converted_video_path):
            converted_video_path = video_converte(video_path, converted_video_path)

        frames = get_video_frames(converted_video_path)
        # config_sidebar = st.sidebar.empty()
        cfg = object_detector_ui()
        # Choose a frame out of the selected frames.
        selected_frames = frame_selector_ui(frames)
        selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)
        selected_frame = selected_frames[selected_frame_index]
        # todo preprocessing, ex roi, transform
        
        # show image with bbox
        st_frame = st.empty()
        res_plotted = draw_image_with_bbox_yolov8(selected_frame, model, conf=cfg['confidence'], tracker=cfg['tracker'])
        # todo postprocessing, ex logits
        roi_plotted = darw_roi(res_plotted, cfg['ROIS'])
        st_frame.image(roi_plotted,
                caption='Detected Frame',
                channels="BGR",
                use_column_width=True
                )
        
        if st.sidebar.button('Run'):
            for i in range(len(selected_frames)):
                draw_image_with_bbox_yolov8(selected_frames[i], model, conf=cfg['confidence'], tracker=cfg['tracker'])
                roi_plotted = darw_roi(res_plotted, cfg['ROIS'])
                st_frame.image(roi_plotted,
                    caption='Detected Frame',
                    channels="BGR",
                    use_column_width=True
                    )
                time.sleep(0.1)
        
        st.video(converted_video_path)

    except Exception as e:
        st.error(f"Error loading video: {e}")
              

# This sidebar UI is a little search engine to find certain object types.
def frame_selector_ui(frames):
    st.sidebar.markdown("# Frame")

    # The user can select a range for how many of the selected objecgt should be present.
    min_elts, max_elts = st.sidebar.slider("Select a frame range?" , 0, len(frames), [0, len(frames)])
    
    selected_frames = frames[min_elts:max_elts]
    if len(selected_frames) < 1:
        return None, None

    return selected_frames


def new_ROI():
    roi_widget = st.sidebar.container()
    roi_widget.write('Input 4 points')
    x1 = roi_widget.number_input('x1')
    y1 = roi_widget.number_input('y1')
    if roi_widget.button('finshed'):
        print(x1, y1)
        roi_widget.empty()
        return x1, y1        
    
def darw_roi(frame, rois):
    pts = np.array(rois, np.int32)
    # Reshape array to 2D
    pts = pts.reshape((-1,1,2))
    # Draw ROI polygon on original image
    return cv2.polylines(frame, [pts], True, (0,255,0), 2)

def select_roi(rois):
    roi_index = st.sidebar.selectbox('Select ROI', options=list(range(len(rois))))
    selected = rois[roi_index]
    if selected:
        for i in range(len(selected)):
            cols_x, cols_y = st.sidebar.columns(2)
            with cols_x:
                rois[roi_index][i][0] = int(st.sidebar.text_input(f'x_{i}', value=rois[roi_index][i][0]))
            with cols_y:
                rois[roi_index][i][1] = int(st.sidebar.text_input(f'y_{i}', value=rois[roi_index][i][1]))
    return rois

# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.empty()
    st.sidebar.markdown("# Model")
    # overlap_threshold = config_sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    def on_click_save():
        settings.save_config(cfg)
        
    cfg = settings.load_config()
    if st.button('New ROI'):
        new_ROI()
    
    
    cfg['ROIS'] = select_roi(cfg['ROIS'])
    
    
    cfg['confidence'] = st.sidebar.slider("Confidence threshold", 0.0, 1.0, cfg.get('confidence', 0.5), 0.01)

    cfg['source'] = st.sidebar.text_input('source', value=cfg['source'])
    st.sidebar.write(cfg['source'])

    cfg['tracker'] = st.sidebar.radio('tracker', options=[True, False])

    cfg['tolerance'] = st.sidebar.slider('tolerance',0, 100, cfg.get('tolerance', 0)) 
    
    st.sidebar.button('save', key='save_config', on_click=on_click_save)
    # on_click_save()
    return cfg




if __name__ == "__main__":
    main()
