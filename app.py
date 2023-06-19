from pathlib import Path
import time
import streamlit as st
import subprocess
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import settings

from src.worker import Worker
from src.utils import DemoMonitor, PSMonitor
import src.utils as ps
def rtsp_record_app():
    
    class VideoRecorder(Worker):
        out_mp4 = 'temp/recorded.mp4'
        out_web_mp4 = 'temp/recorded_h264.mp4'
        def __init__(self, fps, width, height, stop_timeout=None):
            super().__init__(stop_timeout)
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.out_mp4, self.fourcc, fps, (width, height))

        def transform(self, frame):
            self.out.write(frame)
            return frame

        def to_web_mp4(self, input_file_path, output_file_path):
            '''
            ffmpeg -i test_result.avi -vcodec h264 test_result.mp4
            @param: [in] input_file_path 带avi或mp4的非H264编码的视频的全路径
            @return: [output] output_file_path 生成的H264编码视频的全路径
            '''
            cmd = 'ffmpeg -y -i {} -vcodec h264 {}'.format(input_file_path, output_file_path)
            subprocess.call(cmd, shell=True)
            return output_file_path
        
        def stop(self):
            self.out.release()
            self.to_web_mp4(self.out_mp4, self.out_web_mp4)
            super().stop()


    st.title("RTSP Video Stream Recorder")
    st.write("Enter the RTSP URL below and click 'Start Recording' to begin recording the video stream.")

    rtsp_url = st.text_input("RTSP URL")

    if rtsp_url:
        # frame_queue = st.session_state.get('frame_queue', queue.Queue())
        cap = st.session_state.get('cap', None)
        is_recording = st.session_state.get('is_recording', False)
        video_recorder = st.session_state.get('video_recorder', None)
        stframe = st.empty()
        
        if st.button('Start Recording') and not is_recording:
            st.write("Recording started. Click 'Stop Recording' to stop.")
            st.session_state['is_recording'] = True
            cap = cv2.VideoCapture(rtsp_url)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_recorder = VideoRecorder(fps, width, height)
            st.session_state['cap'] = cap
            st.session_state['video_recorder'] = video_recorder 

        if st.button('Stop Recording') and is_recording:
            st.session_state['is_recording'] = False
            cap.release()
            video_recorder.stop()
            st.write(f'video saved to:{video_recorder.out_web_mp4}')
            st.video(video_recorder.out_web_mp4)
            
                
        is_recording = st.session_state.get('is_recording', False)
        while is_recording:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Error reading video stream.")
            video_recorder.recv(frame)
            stframe.image(frame)

def infer_video_app():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # @st.cache_resource
    class YoloV8Worker(Worker):
        def __init__(self, model_path, use_tracker, 
                     conf: float = 0.5, 
                     stop_timeout: float | None = None):
            super().__init__(stop_timeout)
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.use_tracker = use_tracker
            self.conf = conf
        
        def set_model_path(self, model_path):
            if self.model_path != model_path:   # model weight replace
                self.model = YOLO(model_path)
            
        def set_use_tracker(self, use_traker):
            self.use_tracker = use_traker
            
        def set_conf(self, conf):
            self.conf = conf
            
        def transform(self, item):
            if self.use_tracker:
                # res = model.track(image, conf=conf, tracker='botsort.yaml')
                res = self.model.track(item, conf=self.conf, tracker='bytetrack.yaml')
            else:
                res = self.model.predict(item, conf=self.conf, show_conf=True)
            return super().transform(res)
        
    
    
    class Anotator(object):
        def __init__(self, img) -> None:
            self.img = img
        
        def draw_lines(self, points, line_color=(0,255,0)):
            # # Convert selected points to numpy array
            pts = np.array(points, np.int32)
            # Reshape array to 2D
            pts = pts.reshape((-1,1,2))
            # Draw ROI polygon on original image
            return cv2.polylines(self.img, [pts], True, line_color, 2)

        def draw_rect(self, x1, y1, x2, y2, color=(0,0,255)):
            return cv2.rectangle(self.img, (x1,y1), (x2,y2), color, 2)
        
        def get_result_img(self):
            return self.img
        
        def draw_all(self, cfg):
            rois = cfg.get('ROIS')
            patch_area = cfg.get('patch_area')
            alarmarea = cfg.get('alarmarea')
            if rois:
                for _index in range(len(rois)):
                    self.draw_lines(rois[_index])
            if patch_area:
                for _index in range(len(patch_area)):
                    y1, y2, x1, x2 = patch_area[_index]
                    self.draw_rect(x1, y1, x2, y2)
            if alarmarea:
                for _index in range(len(alarmarea)):
                    self.draw_lines(alarmarea[_index])
        
    def on_click_save(cfg):
        settings.save_config(cfg)
        settings.save_to_json(cfg)
        
        
    def select_roi(name, rois, roi_ui):
        # Create a ROI selector
        _index = roi_ui.selectbox(f'Select {name}', options=list(range(len(rois))))
        selected = rois[_index]
        if selected:
            for i in range(len(selected)):
                cols_x, cols_y = roi_ui.columns(2)
                with cols_x:
                    rois[_index][i][0] = cols_x.number_input(f'x_{i+1}_{name}', value=rois[_index][i][0], format='%d')
                with cols_y:
                    rois[_index][i][1] = cols_y.number_input(f'y_{i+1}_{name}', value=rois[_index][i][1], format='%d')
        return rois, _index

    def select_patch_area(patch_area, patch_area_ui):
        _index = patch_area_ui.selectbox('Select patcharea', options=list(range(len(patch_area))))
        if patch_area[_index]:
            cols_y1, cols_y2, cols_x1, cols_x2 = patch_area_ui.columns(4)
            with cols_y1:
                patch_area[_index][0] = cols_y1.number_input('y1', value=patch_area[_index][0], format='%d')
            with cols_y2:
                patch_area[_index][1] = cols_y2.number_input('y2', value=patch_area[_index][1], format='%d')
            with cols_x1:
                patch_area[_index][2] = cols_x1.number_input('x1', value=patch_area[_index][2], format='%d')
            with cols_x2:
                patch_area[_index][3] = cols_x2.number_input('x2', value=patch_area[_index][3], format='%d')                
        return patch_area, _index

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

    def yolov8_config_ui(cfg):
        st.sidebar.empty()
        st.sidebar.markdown("# YOLOV8 Model")
        
        model_path = st.sidebar.text_input('model_path', value=cfg.get('weights', 'weights/yolov8n.pt'), placeholder='input yolov8 weight path')
        last_model_path = st.session_state.get('model_path')
        # yolov8_worker = st.session_state.get('yolov8_worker',None)
        if model_path:
            use_tracker = st.sidebar.radio('use_tracker', [True, False], [True, False].index(cfg.get('use_tracker', False)) )
            conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, cfg.get('conf', 0.5), 0.01)
            # if yolov8_worker is None:
                # yolov8_worker = YoloV8Worker(model_path, use_tracker=use_tracker)
            yolov8_worker = YOLO(model_path, use_tracker)
            yolov8_worker.to(device)
            names = yolov8_worker.names
            cfg.update({'names': names})
            # st.session_state['yolov8_worker'] = yolov8_worker
            cfg.update({'use_tracker': use_tracker, 'conf': conf})
        return yolov8_worker
        
    # This sidebar UI is a little search engine to find certain object types.
    def frame_selector_ui(frames):
        st.sidebar.markdown("# Frame")
        # The user can select a range for how many of the selected objecgt should be present.
        min_elts, max_elts = st.sidebar.slider("Select a frame range?" , 0, len(frames), [0, len(frames)])
        selected_frames = frames[min_elts:max_elts]
        if len(selected_frames) < 1:
            return None, None
        return selected_frames
    
    cfg = st.session_state.get('cfg')
    if cfg is None:
        cfg = settings.load_config()  # yaml
    # cfg = settings.load_from_json()    # json
    yolov8_worker = yolov8_config_ui(cfg['yolov8'])
    st.session_state['cfg'] = cfg
    converted_video_path = st.text_input('video_path', value='temp/recorded_h264.mp4', placeholder='input video path')
    
    if converted_video_path:
        frames = get_video_frames(converted_video_path)
        # Choose a frame out of the selected frames.
        selected_frames = frame_selector_ui(frames)
        selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)
        selected_frame = selected_frames[selected_frame_index]
        if st.sidebar.button('save frame'):
            save_path = f'temp/saved/{Path(converted_video_path).stem}_{selected_frame_index}.jpg'
            cv2.imwrite(save_path, selected_frame)
            st.sidebar.info(f'Image saved to {save_path}')
        if st.sidebar.button('save all frames'):
            for i in selected_frames:
                save_path = f'temp/saved/{Path(converted_video_path).stem}_{i}.jpg'
                cv2.imwrite(save_path, selected_frame)
            st.sidebar.info(f'Images saved to temp/saved/')
                
        
        # st.sidebar.markdown("# Define")
        subject_config_ui, final_result_ui = st.tabs(['subject_config_ui', 'final_result'])
        monitor_cfg = cfg['monitor']
        
        with subject_config_ui:
            st_frame = st.empty()
            st_frame.image(selected_frame, channels='BGR')
            monitor_cfg = cfg['monitor']
            subject_list = monitor_cfg.get('subjects')
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)    # 把radio横着放
            action_type = st.radio('action', ['add', 'update', 'delete'])
            if action_type == 'add':    # 添加新的subject
                sub_cfg = {}
                sub_typ = st.selectbox('subject type', options=['Point', 'Line', 'Area', 'Rect'])
                sub_name = st.text_input('subject name', placeholder='Input Subject Name')
                if sub_typ == 'Point':
                    st.write(f"New Subject {sub_typ}")
                    x = st.number_input('x', step=1)
                    y = st.number_input('y', step=1)
                    sub_cfg.update({
                        "name": sub_name,
                        "type": sub_typ,
                        "x": x,
                        "y": y
                    })
                if sub_typ == 'Line':
                    st.write(f"New Subject {sub_typ}")
                    p1_input, p2_input = st.columns(2)
                    with p1_input:
                        x1 = st.number_input('x1', step=1)
                        x2 = st.number_input('x2', step=1)
                    with p2_input:
                        y1 = st.number_input('y1', step=1)
                        y2 = st.number_input('y2', step=1)
                    p1 = [x1,y1]
                    p2 = [x2,y2]   
                    sub_cfg.update({
                        "name": sub_name,
                        "type": sub_typ,
                        "p1": p1,
                        "p2": p2
                    })
                if sub_typ == 'Rect':
                    st.write(f"New Subject {sub_typ}")
                    x = st.number_input('x', step=1)
                    y = st.number_input('y', step=1)
                    w = st.number_input('w', step=1)
                    h = st.number_input('h', step=1)
                    sub_cfg.update({
                        'name': sub_name,
                        'type': sub_typ,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h
                    })
                if sub_typ == 'Area':
                    st.write(f"New Subject {sub_typ}")
                    pts_str = st.text_input('pts_add', placeholder='Format: [[x1,y1], [x2,y2], ... [xn,yn]]')
                    if pts_str:
                        pts = eval(pts_str)
                        sub_cfg.update({
                            'name': sub_name,
                            'type': sub_typ,
                            'pts': pts,
                        })   
                is_add = st.button("ADD")
                if is_add:
                    subject_list.append(sub_cfg)
                    on_click_save(cfg)
                    st.experimental_rerun()
                    
                if sub_cfg:
                    sub = ps.Subject.new_subject(sub_cfg)
                    img = sub.plot(selected_frame.copy(), show_name=True)
                    st_frame.image(img, channels='BGR')
                    
            if action_type == 'update':    
                # 展示现有的subject并可更改
                options = [[None, None]]
                options.extend([[i, sub['name']] for i, sub in enumerate(subject_list)])
                idx, selected_sub_name = st.selectbox('subjects', options, index=0)
                if selected_sub_name:
                    sub_cfg = subject_list[idx]
                    st.write(f"Subject Type: {sub_cfg['type']}")
                    sub_cfg['name'] = st.text_input('subject name', value=sub_cfg['name'])
                    if sub_cfg['type'] == 'Point':  
                        sub_cfg['x'] = st.number_input('x', value=sub_cfg['x'], step=1)
                        sub_cfg['y'] = st.number_input('y', value=sub_cfg['y'], step=1)
                    if sub_cfg['type'] == 'Line':
                        p1_input, p2_input = st.columns(2)
                        with p1_input:
                            sub_cfg['p1'][0] = st.number_input('x1', value=sub_cfg['p1'][0], step=1)
                            sub_cfg['p2'][0] = st.number_input('x2', value=sub_cfg['p2'][0], step=1)
                        with p2_input:
                            sub_cfg['p1'][1] = st.number_input('y1', value=sub_cfg['p1'][1], step=1)
                            sub_cfg['p2'][1] = st.number_input('y2', value=sub_cfg['p1'][1], step=1)
                    if sub_cfg['type'] == 'Rect':
                        sub_cfg['x'] = st.number_input('x', value=sub_cfg['x'], step=1)
                        sub_cfg['y'] = st.number_input('y', value=sub_cfg['y'], step=1)
                        sub_cfg['w'] = st.number_input('w', value=sub_cfg['w'], step=1)
                        sub_cfg['h'] = st.number_input('h', value=sub_cfg['h'], step=1)
                    if sub_cfg['type'] == 'Area':
                        pts_str = st.text_input('pts_update', value=sub_cfg['pts'])
                        if pts_str:
                            sub_cfg['pts'] = eval(pts_str)

                    is_update = st.button("UPDATE")
                    if is_update:
                        on_click_save(cfg)
                        st.experimental_rerun()
                        
                    sub = ps.Subject.new_subject(sub_cfg)
                    img = sub.plot(selected_frame.copy(), show_name=True)
                    st_frame.image(img, channels='BGR')
                
            if action_type == 'delete':    
                # 展示现有的subject并,不可编辑，可删除
                options = [[None, None]]
                options.extend([[i, sub['name']] for i, sub in enumerate(subject_list)])
                idx, selected_sub_name = st.selectbox('subjects', options, index=0)
                if selected_sub_name:
                    sub_cfg = subject_list[idx]
                    st.write(f"Subject Type: {sub_cfg['type']}")
                    sub_cfg['name'] = st.text_input('subject name', value=sub_cfg['name'], disabled=True)
                    if sub_cfg['type'] == 'Point':  
                        sub_cfg['x'] = st.number_input('x', value=sub_cfg['x'], step=1, disabled=True)
                        sub_cfg['y'] = st.number_input('y', value=sub_cfg['y'], step=1, disabled=True)
                    if sub_cfg['type'] == 'Line':
                        p1_input, p2_input = st.columns(2)
                        with p1_input:
                            sub_cfg['p1'][0] = st.number_input('x1', value=sub_cfg['p1'][0], step=1, disabled=True)
                            sub_cfg['p2'][0] = st.number_input('x2', value=sub_cfg['p2'][0], step=1, disabled=True)
                        with p2_input:
                            sub_cfg['p1'][1] = st.number_input('y1', value=sub_cfg['p1'][1], step=1, disabled=True)
                            sub_cfg['p2'][1] = st.number_input('y2', value=sub_cfg['p1'][1], step=1, disabled=True)
                    if sub_cfg['type'] == 'Rect':
                        sub_cfg['x'] = st.number_input('x', value=sub_cfg['x'], step=1, disabled=True)
                        sub_cfg['y'] = st.number_input('y', value=sub_cfg['y'], step=1, disabled=True)
                        sub_cfg['w'] = st.number_input('w', value=sub_cfg['w'], step=1, disabled=True)
                        sub_cfg['h'] = st.number_input('h', value=sub_cfg['h'], step=1, disabled=True)
                    if sub_cfg['type'] == 'Area':
                        sub_cfg['pts'] = eval(st.text_input('pts_delete', value=sub_cfg['pts'], disabled=True))

                    is_delete = st.button("DELETE")
                    if is_delete:
                        subject_list.pop(idx)
                        on_click_save(cfg)
                        st.experimental_rerun()
                    sub = ps.Subject.new_subject(sub_cfg)
                    img = sub.plot(selected_frame.copy(), show_name=True)
                    st_frame.image(img, channels='BGR')

           
        with final_result_ui:
            st_frame = st.empty()
            st_frame.image(selected_frame, channels='BGR')
            st_write = st.empty()
            if st.button('predict'):
                confidence = cfg['yolov8']['confidence']
                res = yolov8_worker(selected_frame, conf=confidence)
                # logic
                monitor = PSMonitor(monitor_cfg, res[0], yolov8_worker.names)
                # monitor = DemoMonitor(monitor_cfg, res[0], yolov8_worker.names)
                alarm_result = monitor.is_alarm()
                # plot
                img = monitor.display(alarm_result)
                st_frame.image(img, channels='BGR')
                # present result
                write_result = []
                # write_result.extend(monitor.distance)
                write_result.append(alarm_result)
                st_write.write(write_result)
                
                
            if st.button('predict video'):
                # source = selected_frames
                source = converted_video_path
                # source = cfg['source']
                confidence = cfg['yolov8']['confidence']
                names = yolov8_worker.names
                # results = yolov8_worker(source, stream=True, conf=confidence)
                # # todo config cls model
                cls_model_path = 'runs/classify/hat_nohat/weights/best.pt'
                classify_model = YOLO(cls_model_path)
                classify_model.to(device)
                

                DemoMonitor.set_classifer(classify_model)
                if cfg['yolov8']['use_tracker']:
                    results = yolov8_worker.track(source, stream=True, conf=confidence)
                else:
                    results = yolov8_worker.predict(source, stream=True, conf=confidence)
                for res in results:
                    # logic
                    # monitor = PSMonitor(monitor_cfg, res, names)
                    monitor = DemoMonitor.init_frame(monitor_cfg, res, names)
                    monitor.counting()
                    # alarm_result = monitor.is_alarm()
                    # plot
                    # resframe = monitor.display(alarm_result)
                    person_num = monitor.get_how_many_person()
                    classfiy_result_dict = monitor.person_classfiy()
                    person_in_area_list = monitor.person_in_area()
                    resframe = monitor.display()
                    st_frame.image(resframe, channels='BGR')
                    time.sleep(0.05)
                    # present result
                    count_dict = monitor.get_count_dict()
                    result_dict = {}
                    result_dict['person_num'] = person_num
                    result_dict['count_dict'] = count_dict
                    result_dict['person_in_area'] = person_in_area_list
                    result_dict['classfiy_result'] = classfiy_result_dict
                    st_write.write(result_dict)
                torch.cuda.empty_cache()
        
        st.sidebar.button('save', key='save_config', on_click=on_click_save, args=(cfg,))
        

 



def main():
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",["Video Record", "Infer Video", "Yolo Infer Video"])
    if app_mode == "Video Record":
        rtsp_record_app()
    elif app_mode == "Infer Video":
        infer_video_app()
        
if __name__ == '__main__':
    main()




