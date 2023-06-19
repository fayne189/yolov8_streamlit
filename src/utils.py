import time
import cv2
import numpy as np
import collections
import matplotlib.path as mplPath
import threading


   
class Subject(object):
    def __init__(self, name) -> None:
        self.name = name
        
    @classmethod    
    def new_subject(cls, sub):
        if sub['type'] == 'Point':
            name = sub['name']
            x = sub['x']
            y = sub['y']
            new_sub = Point(name, x, y)
        if sub['type'] == 'Line':
            name = sub['name']
            p1 = sub['p1']
            p2 = sub['p2']
            point1 = Point(f'{name}_p1', p1[0], p1[1])
            point2 = Point(f'{name}_p2', p2[0], p2[1])
            new_sub = Line(name, point1, point2)
        if sub['type'] == 'Area':
            name = sub['name']
            pts = sub['pts']
            new_sub = Area(name, pts)
            for i, pt in enumerate(pts):
                x = pt[0]
                y = pt[1]
                p = Point(f'{name}_{i}', x, y)
                new_sub.add_point(p)
        if sub['type'] == 'Rect':
            name = sub['name']
            x = sub['x']
            y = sub['y']
            w = sub['w']
            h = sub['h']
            new_sub = Rect(name, x, y, w, h)
        return new_sub
        
    def plot(self, img):
        pass

class Point(Subject):
    def __init__(self, name, x, y) -> None:
        super().__init__(name)
        self.x = x
        self.y = y
        
    def xy(self):
        return int(self.x), int(self.y)
    
    def plot(self, img, color=(0,0,255), show_name=True):
        x,y = self.xy()
        if show_name:
            cv2.putText(img, self.name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return cv2.circle(img, self.xy(), 3, color, thickness=-1)
           
class Line(Subject):
    def __init__(self, name, p1, p2) -> None:
        super().__init__(name)
        self.p1 = p1
        self.p2 = p2
        
    def plot(self, img, color=(0,0,255), show_name=False):
        if show_name:
            x,y = self.p1.xy()
            cv2.putText(img, self.name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return cv2.line(img, self.p1.xy(), self.p2.xy(), color=color, thickness=2)
    
class Area(Subject):
    def __init__(self, name, pts) -> None:
        super().__init__(name)
        self.pts = pts
        self.points = []
        
    def add_point(self, point):
        self.points.append(point)

    def plot(self, img, color=(0,0,255), show_name=True):
        if show_name:
            cv2.putText(img, self.name, self.points[0].xy(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        pts = np.array(self.pts, np.int32)
        pts = pts.reshape((-1,1,2))
        return cv2.polylines(img, [pts], True, color, thickness=2)
    
        
             
class Rect(Subject):
    def __init__(self, name, x, y, w, h) -> None:
        super().__init__(name)
        self.x = x
        self.y = y
        self.w = w
        self.h = h 
        self.center = self.to_center()
        self.p1 , self.p2 = self.to_p1p2()
        
    def to_center(self):
        return Point(f'{self.name}_center', self.x, self.y)
    
    def to_p1p2(self):
        return Point(f'{self.name}_lt', self.x - self.w/2, self.y - self.h/2), Point(f'{self.name}_rb', self.x + self.w/2, self.y + self.h/2)
    
    def to_pts(self):
        return [[self.x - self.w/2, self.y - self.h/2],[self.x - self.w/2, self.y + self.h/2],[self.x + self.w/2, self.y + self.h/2],[self.x + self.w/2, self.y - self.h/2]]
        
    def plot(self, img, color=(0,0,255), show_name=True):
        if show_name:
            cv2.putText(img, self.name, (int(self.p1.x), int(self.p1.y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        x1,y1 = self.p1.xy()
        x2,y2 = self.p2.xy()
        return cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    def crop(self, im, gain=1.02, pad=10, BGR=True):
        """Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop."""

        w = self.w * gain + pad
        h = self.h * gain + pad
        x1 = np.clip(self.x - w/2, 0, im.shape[1])
        x2 = np.clip(self.x + w/2, 0, im.shape[1])
        y1 = np.clip(self.y - h/2, 0, im.shape[0])
        y2 = np.clip(self.y + h/2, 0, im.shape[0])
        xyxy = [x1, y1, x2, y2]
        crop = im[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), ::(1 if BGR else -1)]
        return crop
         
class Object(object):
    count_dict = {}
    def __init__(self, name, xywh, id=None) -> None:
        self.name = name 
        self.xywh = xywh
        self.id = id
        self.rect = Rect(f'{self.name}_rect', *xywh)    
        self.classify_dict = {}
    
    def get_name(self):
        return self.name if self.id is None else f'{self.name}{self.id}'
            
    def classifiy(self, orig_img, model):
        img = self.rect.crop(orig_img)
        res = model(img)[0]
        result = model.names[res.probs.top1]
        self.classify_dict.update({model.ckpt_path: result})
        return result
    

class Anotator(object):
        def __init__(self, img) -> None:
            self.img = img
        
        def draw_rect(self, rect, color=(0,0,255)):
            x1,y1 = rect.p1.xy()
            x2,y2 = rect.p2.xy()
            return cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
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
        
        def draw_line(self, line, color=(0,0,255)):
            cv2.line(self.img, line.p1.xy(), line.p2.xy(), color=color, thickness=2)
            
        def draw_point(self, point, color=(0,255,0)):
            cv2.circle(self.img, point.xy(), 2, color, -1)
            
        def draw_area(self, area, color=(0,0,255)):
            # # Convert selected points to numpy array
            pts = np.array(area.pts, np.int32)
            # Reshape array to 2D
            pts = pts.reshape((-1,1,2))
            # Draw ROI polygon on original image
            return cv2.polylines(self.img, [pts], True, color, 2)
            
        def draw_subject(self, subject, color=(0,0,255)):
            if isinstance(subject, Point):
                self.draw_point(subject, color)
            if isinstance(subject, Line):
                self.draw_line(subject, color)
            if isinstance(subject, Area):
                self.draw_area(subject, color)
            if isinstance(subject, Rect):
                self.draw_rect(subject, color)   
                                                  

class BaseMonitor(object):
    on_screen = False
    frame = None
    lock = threading.Lock()
    def __init__(self, cfg, yolo_result, names) -> None:
        '''
        cfg: Monitor的config
        yolo_result: yolov8的结果
        names: yolo模型的类别
        '''
        self.objects = []
        self.subjects = []
        self.yolo_result = yolo_result
        self.cfg = cfg
        self.names = names
        self.init_objects_from_yolo_result()
        self.init_subjects_from_cfg()
        self.orig_img = yolo_result.orig_img
        self.display_frame = self.orig_img.copy()
    
    def init_objects_from_yolo_result(self):
        # pharse object
        for i in range(self.yolo_result.boxes.shape[0]):
            c = int(self.yolo_result.boxes.cls[i])
            name = self.names[c]
            xywh = list(self.yolo_result.boxes.xywh[i].cpu().numpy())
            id = None
            if self.yolo_result.boxes.id is not None:
                id = int(self.yolo_result.boxes.id[i])
            self.objects.append(Object(name, xywh, id))

    def init_subjects_from_cfg(self):
        ## pharse subject
        subjects =  self.cfg['subjects']
        for sub in subjects:
            new_sub = Subject.new_subject(sub)
            self.subjects.append(new_sub)
        
    def display(self, objects=True, subjects=True, sub_color=(0,128,0), show_sub_name=True):
        img = self.display_frame
        if objects:
            img = self.yolo_result.plot(img=self.display_frame)
        if subjects:
            if isinstance(subjects, list):
                for i in subjects:
                    self.subjects[i].plot(img, sub_color)
            else:
                for sub in self.subjects:
                    sub.plot(img, color=sub_color, show_name=show_sub_name)
        if BaseMonitor.on_screen:
            with BaseMonitor.lock:
                BaseMonitor.frame = img
        return img
       
    def get_subject_by_name(self, sub_name):
        for sub in self.subjects:
            if sub.name == sub_name:
                return sub
    
 
    @classmethod
    def init_frame(cls, cfg, yolo_result, names):
        return cls(cfg, yolo_result, names)

    @classmethod
    def set_on_screen(cls):
        cls.on_screen = True
    
    @classmethod
    def get_frame(cls):
        with BaseMonitor.lock:
            return cls.frame
    
class PatchMonitor(object):
    '''
    ps: patch_area的定义 [y1,y2,x1,x2]
    '''
    def __init__(self, patch_area_list, thresh_list) -> None:
        self.color_dict = self.get_color_dict()
        self.patch_area_list = patch_area_list
        self.thresh_list = thresh_list
    
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

    def check_fixed_update(self, frame):
        '''
        the light color
        '''
        for i, patch_area in enumerate(self.patch_area_list):
            x,y,w,h = patch_area.x, patch_area.y, patch_area.w, patch_area.h
            y1, y2, x1, x2 = int(y-h/2), int(y+h/2), int(x-w/2), int(x+w/2) 
            img_region = frame[y1:y2,x1:x2,:]
            if not img_region.any():
                return False
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
    
class PSMonitor(BaseMonitor):
    
    ''' 生产安全监控器
    ---
    判断逻辑如下：
    可分为有无PatchMonitor的两种情况,
    ## 有PatchMonitor的情况:
    1. 当PatchMonitor监控区域为绿色, 或者像素均值发生变换时, moving_flag返回为True
    2. 当moving_flag为True时, 计算手的bbox与线的距离, 小于阈值则报警
    ## 无PatchMonitor的情况:
    1. 计算手的bbox与线的距离, 小于阈值则报警
    '''
    def __init__(self, cfg, yolo_result, names) -> None:
        super().__init__(cfg, yolo_result, names)
        self.threahold = 30
        self.distance = []
        self.pm = None
        self.objects_names = ['person']
        self.alarm_area_names = ['alarm_area']
        self.alarm_line_points_index = [[0,3]]
        
        # self.alarm_line = cfg.get('alarm_line')
        if cfg.get('patch_monitor'):
            subject_name_list = cfg['patch_monitor']['subject_name']
            patch_area_list = [sub for sub in self.subjects if sub.name in subject_name_list]
            thresh_list = cfg['patch_monitor']['thresh_list']
            self.pm = PatchMonitor(patch_area_list, thresh_list)
        
    def caculate_distance(self):
        for obj in self.objects:
            if obj.name not in self.objects_names:
                continue
            center_point = obj.rect.p1
            for sub in self.subjects:
                try:
                    area_index = self.alarm_area_names.index(sub.name)  # 警戒区域
                    point_index = self.alarm_line_points_index[area_index]
                    alarm_line_p1 = sub.points[point_index[0]]
                    alarm_line_p2 = sub.points[point_index[1]]
                    self.distance.append(self.distance_point_to_line(center_point, alarm_line_p1, alarm_line_p2))
                except:
                    pass
    
    def get_alarmed_area_and_bbox(self):
        dis_arr = np.array(self.distance)
        min_index = np.argmin(dis_arr)
        obj_id, sub_id =  min_index.reshape(len(self.subjects), -1)
        alarmed_area = self.subjects[sub_id]
        alarmed_bbox = self.objects[obj_id].rect
        return alarmed_area, alarmed_bbox

    # def distance_point_to_line(self, point, line_point1, line_point2):
    #     #计算直线的三个参数
    #     A = line_point2.x - line_point1.y
    #     B = line_point1.x - line_point2.x
    #     C = (line_point1.y - line_point2.y) * line_point1.x + \
    #         (line_point2.x - line_point1.x) * line_point1.y
    #     #根据点到直线的距离公式计算距离
    #     distance = np.abs(A * point.x + B * point.y + C) / (np.sqrt(A**2 + B**2)+1e-6)
    #     return distance
    
    def distance_point_to_line(self, point, line_p1, line_p2):
        x0, y0 = point.xy()
        x1, y1 = line_p1.xy()
        x2, y2 = line_p2.xy()
        return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / ((y2-y1)**2 + (x2-x1)**2)**0.5
    
    def is_alarm(self):
        # 编写报警的逻辑
        moving_flag = True
        if self.pm:
            moving_flag = self.pm.check_fixed_update(self.orig_img)
        if moving_flag:
            self.caculate_distance()
            if len(self.distance):
                return min(self.distance) < self.threahold
        return False
    
    
    def display(self, is_alarm, objects=True, subjects=True, sub_color=(0, 128, 0), show_sub_name=True):
        sub_color = (0, 0, 255) if is_alarm else (0, 255, 0)
        for obj in self.objects:
            if obj.name not in self.objects_names:
                continue
            obj.rect.p1.plot(self.display_frame, show_name=False)
        for sub in self.subjects:
            try:
                area_index = self.alarm_area_names.index(sub.name)  # 警戒区域
                point_index = self.alarm_line_points_index[area_index]
                alarm_line_p1 = sub.points[point_index[0]]
                alarm_line_p2 = sub.points[point_index[1]]
                alarm_line_p1.plot(self.display_frame, show_name=False)
                alarm_line_p2.plot(self.display_frame, show_name=False)
            except:
                pass
        return super().display(objects, subjects, sub_color, show_sub_name)


class CountMonitor(object):
    count_dict = {}  # 计数字典
    time_dict = {}  # 时间字典
    
    def __init__(self, name, threshold) -> None:
        self.name = name  # 名称
        self.threshold = threshold  # 阈值
        self.is_over_threshold = False  # 是否超过阈值
        
    def update_count(self, count):
        CountMonitor.time_dict[self.name] = time.time()  # 更新时间
        if count > self.threshold:
            self.is_over_threshold = True  # 超过阈值
        else:
            self.is_over_threshold = False  # 未超过阈值

    @classmethod
    def clear_count(cls, name):
        if name in cls.count_dict:
            cls.count_dict[name] = 0  # 清零计数
            cls.time_dict[name] = time.time()  # 更新时间

    @classmethod
    def check_timeout(cls, timeout):
        to_delete = []
        for name in cls.time_dict:
            if time.time() - cls.time_dict[name] > timeout:
                cls.clear_count(name)  # 超时清零
                to_delete.append(name)
        for name in to_delete:
            del cls.time_dict[name]
            del cls.count_dict[name]

    @classmethod
    def monitor_count(cls, name, count, threshold):
        if name not in cls.count_dict:
            cls.count_dict[name] = 0  # 初始化计数
            cls.time_dict[name] = time.time()  # 初始化时间
        cls.count_dict[name] += count  # 更新计数
        counter = cls(name, threshold)  # 创建监控器
        counter.update_count(cls.count_dict[name])  # 更新监控器状态
        return counter.is_over_threshold  # 返回是否超过阈值
    
        
class DemoMonitor(BaseMonitor):
    CountMonitor
    def __init__(self, cfg, yolo_result, names) -> None:
        super().__init__(cfg, yolo_result, names)
        self.threshold = cfg.get('threshold', 10)  # 阈值
        self.timeout = 3    # 3s
        self.results = []
        self.object_names = ['person']
        self.to_count_objects = []
        self.get_to_count_objects()
        
    
    def get_to_count_objects(self):
        # 条件1 在区域内
        for sub in self.subjects:
            if sub.name.startswith('alarm_area'):
                for obj in self.objects:
                    x, y = obj.rect.center.xy()
                    if self.point_in_area(x,y,sub.pts) and obj.name.startswith('person'):
                        self.to_count_objects.append(obj)
            
    def point_in_area(self, x, y, pts):
        """
        判断点是否在多边形内
        """
        poly_path = mplPath.Path(np.array(pts))
        return poly_path.contains_point((x,y))
    
    def is_alarm(self):
        """
        判断是否超过阈值
        """
        alarmed_idx = [i for i, r in enumerate(self.results) if r]  # 找到true的index
        alarmed_names = [self.to_count_objects[i].get_name() for i in alarmed_idx]
        return alarmed_names

    def counting(self):
        CountMonitor.check_timeout(self.timeout)
        for obj in self.to_count_objects:
            name = obj.get_name()
            self.results.append(CountMonitor.monitor_count(name, 1, self.threshold))
            
    def get_count_dict(self):
        return CountMonitor.count_dict
    
    def get_time_dict(self):
        return CountMonitor.time_dict
    
    def get_how_many_person(self):
        person_count = 0
        for name in CountMonitor.count_dict:
            if name.startswith('person'):
                person_count += 1
        return person_count
    
    
    @classmethod
    def set_classifer(cls, model):
        cls.model = model
    
    def person_classfiy(self):
        model = DemoMonitor.model
        person_classfiy = {}
        for obj in self.to_count_objects:
            result = obj.classifiy(self.orig_img, model)
            person_classfiy.update({obj.get_name(): result})
        return person_classfiy
    
    def person_in_area(self):
        person_in_area = []
        for sub in self.subjects:
            if sub.name.startswith('alarm_area'):
                for obj in self.objects:
                    x, y = obj.rect.center.xy()
                    if self.point_in_area(x,y,sub.pts) and obj.name.startswith('person'):
                        person_in_area.append(obj.get_name())
        return person_in_area