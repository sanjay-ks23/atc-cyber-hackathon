'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
'''

import math
import os
from typing import List, Tuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *


 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True



class Vehicle():
    def __init__(self,id : int, vehicle_class : str, bbox : tuple,time_stamp : float = 0) -> None:
        """Class that contains data and methods associated with a tracked/detected vehicle.

        Args:
            id (int): vehicle id assigned from
            vehicle_class (str): class of vehicle based on COCO class name
            bbox (tuple): bounding box defined by 2 (x,y) coordinates to form a rectangle
            time_stamp (float, optional): time at which it was created. Defaults to 0.
        """
        self.id = id
        self.vehicle_class = vehicle_class
        self.bbox = bbox
        self.positions : List[Tuple[int, int]] = []
        self.positions.append(self.get_centre_of(bbox))
        self.time_stamps = []
        self.time_stamps.append(time_stamp)
        self.velocity = 0
        pass
    
    def get_centre_of(self,bbox : tuple) -> tuple:
        """Gets the centre of a bounding box

        Args:
            bbox (tuple): Bounding box defined as two corners of (x,y) coordinates

        Returns:
            tuple: (x,y) coordinate representing the centre
        """
        x_avg = (bbox[0] + bbox[2])/2
        y_avg = (bbox[1] + bbox[3])/2
        return (x_avg,y_avg)
    
    def update_pos(self, bbox : tuple, time_stamp : float) -> None:
        """Updates the position of the car and accordingly adds time stamp of position change

        Args:
            bbox (tuple): _description_
            time_stamp (float): _description_
        """
        self.positions.append(self.get_centre_of(bbox))
        self.bbox = bbox
        self.line_segment = (self.positions[-1],self.positions[-2])
        self.time_stamps.append(time_stamp)
        self.update_velocity()
    
    def intersects(self, other_line_segment : tuple) -> bool:
        """DO NOT USE
        Checks if last move by a vehicle intersects a line segment

        Args:
            other_line_segment (tuple): line segment defined using two (x,y) points

        Returns:
            bool: True or False based on if intersection occured
        """
        other_a = other_line_segment[0]
        other_b = other_line_segment[1]
        other_a_area = self.tri_area(self.line_segment[0],self.line_segment[1],other_a)
        other_b_area = self.tri_area(self.line_segment[0],self.line_segment[1],other_b)
        
        if(other_b_area >= 0 and other_a_area <= 0):
            return True
        elif(other_a_area <= 0 and other_b_area >= 0):
            return True
        else:
            return False
    
    def tri_area(a : tuple, b : tuple, c : tuple) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) -(c[0] - a[0]) * (b[1] - a[1])
    
    def update_velocity(self) -> None:
        """Updates velocity of vehicle in pixels/sec
        """
        x1,y1 = self.positions[-1]
        x2,y2 = self.positions[-2]
        dx = (x2 - x1)**2
        dy = (y2 - y1)**2
        ds = math.sqrt(dx + dy)
        dt = self.time_stamps[-1] - self.time_stamps[-2]
        self.velocity = math.ceil((ds) / dt)
            
    def __repr__(self) -> str:
        return f"{self.id}, Class: {self.vehicle_class}, Velocity:{self.velocity}"
    
    def draw_visualisation(self, img : cv2.Mat, draw_complex : bool = True) -> None:
        """Draws visualisation of car on given frame.

        Args:
            img (cv2.Mat): frame to draw on
            draw_complex (bool, optional): Whether to add extra details to visualisation. Defaults to True.
            
            Leave draw_complex as true to get extra details like id, velocity, vehicle class.
            Otherwise, a simple box is drawn.
        """
        cmap = plt.get_cmap('tab20b') #initialize color map
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        class_name = self.vehicle_class
        bbox = self.bbox
        track_id = self.id
        color = colors[int(track_id) % len(colors)]  # draw bbox on screen
        color = [i * 255 for i in color]
        if draw_complex:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-15)),  (int(bbox[0])+(len(self.__repr__()))*5 ,int(bbox[1])), color, -1)
            cv2.putText(img,self.__repr__(),(int(bbox[0]), int(bbox[1]-6)),0, 0.3, (255,255,255),1, lineType=cv2.LINE_AA) 
        else:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)

class TrackerWrapped:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="./io_data/input/classes/coco.names",  ):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.detector = Detector(classes=[2,3,7])
        self.detector.load_model('./weights/yolov7x.pt',)
        self.reID_model_path = './deep_sort/model_weights/mars-small128.pb'
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()

        # initialize deep sort
        self.encoder = create_box_encoder(self.reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric)


    def track_video(self,video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0) -> list:
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
        '''
        try: # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        out = None
        if output: # get video ready to save locally if flag is set
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        frame_num = 0
        while True: # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num +=1

            if skip_frames and not frame_num % skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
            if verbose >= 1:start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            yolo_dets = self.detector.detect(frame.copy(), plot_bb = False)  # Get the detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0
            
            else:
                bboxes = yolo_dets[:,:4]
                bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
                bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

                scores = yolo_dets[:,4]
                classes = yolo_dets[:,-1]
                num_objects = bboxes.shape[0]
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            
            names = []
            for i in range(num_objects): # loop through objects and use class index to get class name
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)

            names = np.array(names)
            count = len(names)

            if count_objects:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            self.tracker.predict()  # Call the tracker
            self.tracker.update(detections) #  updtate using Kalman Gain

            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
        
                color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    

                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count}")
            
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if output: out.write(result) # save output video

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cv2.destroyAllWindows()

    def track_frame(self,img : cv2.Mat, count_objects:bool=False, verbose:int = 0) -> dict:
        """Generates tracking data for given frame

        Args:
            img (cv2.Mat): input frame
            output (str): _description_
            count_objects (bool, optional): Number of objects. Defaults to False.
            verbose (int, optional): Verbosity level of logging. Defaults to 0.

        Returns:
            list: [class name, track id, bounding box] for all tracks detected
        """
        if verbose >= 1:start_time = time.time()

        # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
        yolo_dets = self.detector.detect(img.copy(), plot_bb = False)  # Get the detections
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if yolo_dets is None:
            bboxes = []
            scores = []
            classes = []
            num_objects = 0
        
        else:
            bboxes = yolo_dets[:,:4]
            bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

            scores = yolo_dets[:,4]
            classes = yolo_dets[:,-1]
            num_objects = bboxes.shape[0]
        # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
        
        names = []
        for i in range(num_objects): # loop through objects and use class index to get class name
            class_indx = int(classes[i])
            class_name = self.class_names[class_indx]
            names.append(class_name)

        names = np.array(names)
        count = len(names)

        # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
        features = self.encoder(img, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

        cmap = plt.get_cmap('tab20b') #initialize color map
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        self.tracker.predict()  # Call the tracker
        self.tracker.update(detections) #  updtate using Kalman Gain
        results = {}
        for track in self.tracker.tracks:  # update new findings AKA tracks
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            # print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            results[track.track_id] = Vehicle(track.track_id,
                                              class_name,
                                              bbox)
        return results