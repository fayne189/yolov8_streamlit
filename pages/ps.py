import sys
import os 
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
sys.path.append('..')
import settings
import collections
# from streamlit_drawable_canvas import st_canvas

def main():
    # infer_uploaded_image()
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )
    if source_img is None:
        uploaded_image = cv2.imread('temp/10.55.215.61.jpg')
    else:
        uploaded_image = cv2.imdecode(np.frombuffer(source_img.read(), np.uint8), 1)
    cfg = config_ui(uploaded_image)
    monitor = Monitor()
    st.write(monitor.check_fixed_update(uploaded_image, cfg['patch_area']))


def config_ui(uploaded_image):
    # sb = st.sidebar.empty()
    # st.sidebar.image(uploaded_image, channels='BGR', use_column_width=True)
    st.sidebar.markdown("# Model")
    # overlap_threshold = config_sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    def on_click_save():
        settings.save_config(cfg)
    
    cfg = settings.load_config()
    st.sidebar.write(cfg)
    roi_ui, alarmarea_ui, patch_area_ui = st.tabs(['ROI', 'alarmarea', 'patch_area'])
    with roi_ui:
        cfg['ROIS'] = select_roi('ROI',cfg['ROIS'], roi_ui, uploaded_image)
    
    with alarmarea_ui:
        cfg['alarmarea'] = select_roi('alarmarea', cfg['alarmarea'], alarmarea_ui, uploaded_image, (0,0,255))
        
    with patch_area_ui:
        cfg['patch_area'] = select_patch_area(cfg['patch_area'], patch_area_ui, uploaded_image)
    st.sidebar.button('save', key='save_config', on_click=on_click_save)
    return cfg

def select_roi(name, rois, roi_ui, img, line_color=(0,255,0)):
    st_frame = st.empty()
    # Create a ROI selector
    roi_index = roi_ui.selectbox(f'Select {name}', options=list(range(len(rois))))
    selected = rois[roi_index]
    if selected:
        for i in range(len(selected)):
            cols_x, cols_y = roi_ui.columns(2)
            with cols_x:
                rois[roi_index][i][0] = cols_x.number_input(f'x_{i}_{name}', value=rois[roi_index][i][0], format='%d')
            with cols_y:
                rois[roi_index][i][1] = cols_y.number_input(f'y_{i}_{name}', value=rois[roi_index][i][1], format='%d')
    
    # # Convert selected points to numpy array
    pts = np.array(rois, np.int32)

    # Reshape array to 2D
    pts = pts.reshape((-1,1,2))
    # Draw ROI polygon on original image
    cv2.polylines(img, [pts], True, line_color, 2)
    # Display image with ROI
    st_frame.image(img, channels='BGR', use_column_width=True)
    return rois

def select_patch_area(patch_area, patch_area_ui, img):
    st_frame = st.empty()
    _index = patch_area_ui.selectbox('Select patcharea', options=list(range(len(patch_area))))
    if patch_area[_index]:
        cols_y1, cols_y2, cols_x1, cols_x2 = patch_area_ui.columns(4)
        with cols_y1:
            patch_area[_index][0] = cols_y1.number_input(f'y1', value=patch_area[_index][0], format='%d')
        with cols_y2:
            patch_area[_index][1] = cols_y2.number_input(f'y2', value=patch_area[_index][1], format='%d')
        with cols_x1:
            patch_area[_index][2] = cols_x1.number_input(f'x1', value=patch_area[_index][2], format='%d')
        with cols_x2:
            patch_area[_index][3] = cols_x2.number_input(f'x2', value=patch_area[_index][3], format='%d')                
    y1, y2, x1, x2 = patch_area[_index]
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
    st_frame.image(img, channels='BGR', use_column_width=True)
    return patch_area
    

class Monitor(object):
    def __init__(self) -> None:
        self.color_dict = self.get_color_dict()
        self.thresh_list = [-100,-100]
    
    def get_color_dict(self):
        color_dict = collections.defaultdict(list)
        
        color_lst = []
        color_lst.append(np.array([156,0,46]))
        color_lst.append(np.array([180,255,255]))
        color_dict['red'] = color_lst

        
        color_lst = []
        color_lst.append(np.array([26,0,46]))
        color_lst.append(np.array([34,255,255]))
        color_dict['yellow'] = color_lst

        color_lst = []
        color_lst.append(np.array([35,0,46]))
        color_lst.append(np.array([77,255,255]))
        color_dict['green'] = color_lst

        return color_dict
    
    def check_patch_color(self, area):
        '''
        find thr patch color
        '''
        hsv = cv2.cvtColor(area,cv2.COLOR_BGR2HSV)
        maxsum = -100
        for d in self.color_dict:
            mask = cv2.inRange(hsv,self.color_dict[d][0], self.color_dict[d][1])
            binary = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)[1]
            binary = cv2.dilate(binary,None,iterations = 2)
            cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            _sum = 0
            for c in cnts:
                _sum+=cv2.contourArea(c)
            if _sum > maxsum:
                maxsum = _sum
                color = d
        return color

    def check_fixed_update(self, frame, patch_area_list):
        '''
        the light color
        '''
        for i, patch_area in enumerate(patch_area_list):
            img_region = frame[patch_area[0]:patch_area[1],patch_area[2]:patch_area[3],:]
            thresh = self.thresh_list[i]
            if i == 1:
                color = self.check_patch_color(img_region)
                print(color)
                if not color == 'green':
                    return False

            _img_region = img_region[:,:,0] * 0.2989 + img_region[:,:,1] * 0.5870 + img_region[:,:,2] * 0.1140  # 放大绿色的特征
            _img_region = np.array(_img_region).mean()  # 算平均
            if thresh < 0:
                _img_region = _img_region * -1
            if _img_region < thresh:
                continue
            else:
                return False
        return True

if __name__ == "__main__":
    main()