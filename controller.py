from typing import Dict, List, Tuple
from Lane import Lane
from model_wrapper import Vehicle,TrackerWrapped
import cv2
import json
import numpy as np

# Skip every nth frame
FRAME_SKIP = 3

class Controller():
    def __init__(self,feed_source, *args,**kwargs):
        """Class containing all necessary methods to get input and display visualisation for the traffic controller

        Args:
            feed_source (file path or camera no.): source to initialise a cv2.VideoCapture object from
        """
        self.source = cv2.VideoCapture(feed_source)
        self.frame_dimensions= np.array((self.source.get(cv2.CAP_PROP_FRAME_WIDTH),
        self.source.get(cv2.CAP_PROP_FRAME_HEIGHT)),dtype=np.int32)
        self.lanes : List[Lane] = [] # Add code to load in lanes
        self.tracked : Dict[int,Vehicle]= {}
        self.detected : Dict[int,Vehicle] = {}
        self.time = 0
        self.frame_count = 0
        self.Tracker = TrackerWrapped()
        self.last_deleted = 0
        self.frame_rate = self.source.get(cv2.CAP_PROP_FPS)
        self.lane_avg_densities : List[float] = []
        self.codec = cv2.VideoWriter_fourcc(*"XVID")
        
    def load_lanes_from_json(self,fp : str):
        """Creates Lane objects defined in json format and attaches them to Controller instance

        Args:
            fp (str): path to json
        """
        with open(fp) as f:
            data : dict = json.load(f)
            f.close()
        for cnt, lane in enumerate(data["lanes"]):
            pos = lane["pos"]
            timing = lane["timing"]
            self.lanes.append(Lane(pos, cnt, timing))
        self.lane_avg_densities = [0.0 for i in range(len(self.lanes))]
        self.intersection = Lane(data["intersection"]["pos"],"intersection",[100,100])
    
    def update_tracked_cars(self, frame : cv2.Mat) -> None:
        """Updates car tracking info given frame from camera.
        Updates the internal self.tracked dictionary containing tracked cars.
        Cars are represented internally using the Vehicle class.
        Args:
            frame (cv2.Mat): Input frame to process and track cars from
        """
        self.detected = self.Tracker.track_frame(frame)
        tracked_ids = list(self.tracked.keys())
        result_ids = list(self.detected.keys())
        for key in result_ids:
            car = self.detected[key]
            if key in tracked_ids:
                self.tracked[key].update_pos(car.bbox,self.time)
            else:
                self.tracked[key] = Vehicle(car.id,car.vehicle_class,car.bbox,self.time)
        if(self.frame_count - self.last_deleted >= self.frame_rate):
            self.last_deleted = self.frame_count
            for key in tracked_ids:
                if(not(key in result_ids)):
                    self.tracked.pop(key)
    
    def update_lane_data(self):
        """Helper function to update lane data by calling the update_lane method for attached lanes.
        """
        for lane in self.lanes : 
            lane.update_lane(self.tracked)
        
        self.intersection.update_lane(self.tracked)
    
    def visualise(self, frame : cv2.Mat) -> None:
        """Draws visualisation of the data stored in the Controller class onto the given frame.
        This includes:
            - Lane data on top left corner, including wait times calculated
            - Car data on tracked cars
            - Lane data on lane outlines
        

        Args:
            frame (cv2.Mat): frame to draw visualisation onto
        """
        pt1 = np.array((self.frame_dimensions[0]-400, (22*(len(self.lanes)+1)+20)),dtype=np.int32)
        pt2 = np.array((self.frame_dimensions[0], 0),dtype=np.int32)
        cv2.rectangle(frame,pt1,pt2, (255, 255, 255), -1)

        for lane in self.lanes : 
            lane.visualize(frame)
            stat = f"Lane{lane.lane_id}: Density: {lane.car_density} Duration: {lane.average_uptime:.2f}"
            cv2.putText(frame, stat, np.array((self.frame_dimensions[0]-400, 22*(lane.lane_id+1)),dtype=np.int32), 0, 0.7, (0, 0, 0),1, lineType=cv2.LINE_AA)

        stat = f"Intersection: Density: {self.intersection.car_density} Speed: {self.intersection.average_velocity:.2f}"
        cv2.putText(frame, stat, np.array((self.frame_dimensions[0]-400, 22*(len(self.lanes)+1)), dtype=np.int32), 0, 0.7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
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
        # Only when working with hardware
        pass
    
    def main_loop(self, out:str=None) -> None:
        """Driver loop to fetch input and run update methods on.
        Also visualises and can write visualised frames to a file.

        Args:
            out (str, optional): filepath to write output of visualisations to.
            Leave default to not write anything.
            Defaults to None.
        """
        if out : writer = cv2.VideoWriter(out, self.codec, self.frame_rate/FRAME_SKIP, self.frame_dimensions)
        while True:
            return_value, frame = self.source.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            self.frame_count += 1
            self.time += 1/self.frame_rate
            
            if(self.frame_count%FRAME_SKIP != 0):
                continue
            
            self.update_tracked_cars(frame)
            self.update_lane_data()
            self.visualise(frame)
            
            frame = np.asarray(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Video out",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            if out: writer.write(frame)

            