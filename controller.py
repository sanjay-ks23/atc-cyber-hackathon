from Lane import Lane
from model_wrapper import Vehicle,TrackerWrapped
import cv2




class Controller():
    def __init__(self,feed_source, *args,**kwargs):
        self.source = cv2.VideoCapture(feed_source)
        self.lanes = None # Add code to load in lanes
        self.tracked = {}
        self.detected = {}
        self.time = 0
        self.frame_count = 0
        self.Tracker = TrackerWrapped()
        self.last_deleted = 0
        self.frame_rate = self.source.get(cv2.CAP_PROP_FPS)
        
    
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
        pass
    
    def visualise(self, frame : cv2.Mat) -> None:
        for key in self.detected.keys():
            self.tracked[key].draw_visualisation(frame,(0,255,0))
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
            self.visualise(frame)
            
            cv2.imshow("Video out",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            