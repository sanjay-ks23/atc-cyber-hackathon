from typing import Dict, List
from Lane import Lane
from model_wrapper import Vehicle,TrackerWrapped
import cv2
import json



class Controller():
    def __init__(self,feed_source, *args,**kwargs):
        self.source = cv2.VideoCapture(feed_source)
        self.lanes : List[Lane] = [] # Add code to load in lanes
        self.tracked : Dict[int,Vehicle]= {}
        self.detected : Dict[int,Vehicle] = {}
        self.time = 0
        self.frame_count = 0
        self.Tracker = TrackerWrapped()
        self.last_deleted = 0
        self.frame_rate = self.source.get(cv2.CAP_PROP_FPS)
        self.lane_avg_densities : List[float] = []
        
    def load_lanes_from_json(self,fp : str):
        with open(fp) as f:
            data : dict = json.load(f)
            f.close()
        for key in data["lanes"].keys():
            pos = data["lanes"][key]["pos"]
            timing = data["lanes"][key]["timing"]
            self.lanes.append(Lane(pos,key,timing))
        self.lane_avg_densities = [0.0 for i in range(len(self.lanes))]
        self.intersection = Lane(data["intersection"]["pos"],"intersection",[100,100])
    
    def update_tracked_cars(self, frame : cv2.Mat) -> None:
        self.detected = self.Tracker.track_frame(frame)
        tracked_ids = list(self.tracked.keys())
        result_ids = list(self.detected.keys())
        for key in result_ids:
            car = self.detected[key]
            if key in tracked_ids:
                self.tracked[key].update_pos(car.bbox,self.time)
            else:
                self.tracked[key] = Vehicle(car.id,car.vehicle_class,car.bbox,self.time)
        if(self.frame_count - self.last_deleted >= self.frame_rate*2):
            self.last_deleted = self.frame_count
            for key in tracked_ids:
                if(not(key in result_ids)):
                    self.tracked.pop(key)
    
    def update_lane_data(self):
        for lane in self.lanes : 
            lane.update_lane(self.tracked)
        
        self.intersection.update_lane(self.tracked)
    
    def visualise(self, frame : cv2.Mat) -> None:
        
        for lane in self.lanes : 
            lane.visualize(frame)
        self.intersection.visualize(frame)
        
        in_lane_ids = []
        for lane_ids in [lane.car_ids for lane in self.lanes]:
            in_lane_ids += lane_ids
        # print(in_lane_ids)
        for key in self.tracked:
            if(key in in_lane_ids):
                self.tracked[key].draw_visualisation(frame,draw_complex=True)
            else:
                self.tracked[key].draw_visualisation(frame,draw_complex=False)
        pass
    
    def get_signal_statuses(self):
        
        pass
    
    def main_loop(self) -> None:
        while True:
            return_value, frame = self.source.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            self.frame_count += 1
            self.time += 1/self.frame_rate
            
            if(self.frame_count%10 != 0):
                continue
            
            self.update_tracked_cars(frame)
            self.update_lane_data()
            self.visualise(frame)
            
            cv2.imshow("Video out",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            