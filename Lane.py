from typing import Dict, List, Set, Tuple
import cv2
import numpy as np
from matplotlib import path as mpPath

from model_wrapper import Vehicle

class Lane : 
    color = (15, 220, 10)
    def __init__(self, co_ordinates: Tuple[Tuple[int, int]], id:int, timing : Tuple[int, int]) -> None:
        """Lane class to hold data related to a lane in intersection.
        Also contains methods to update the data.

        Args:
            co_ordinates (Tuple[Tuple[int, int]]): List of (x,y) pixel coordintes to define the bounding box for each lane
            id (int): the id associated with the lane
            timing (Tuple[int, int]): A tuple of [min, max] timings for a lane.
            
            The timing is set based on the immediate density of traffic in the lane.
            the average_uptime variable holds the sliding average of this over the LAST_TIME_MAX updates
            
            NOTE: CURRENTLY ONLY WORKS WITH QUADRILATERALS AS LANES
        """
        self.co_ordinates = co_ordinates
        self.lane_id = id
        self.direction : int = 1 # 1 -> forward, -1-> backward
        self.average_velocity : float = 0 
        self.count_cars : int = 0
        self.area : float = self.get_area()
        self.car_ids : Set[int] = set()
        self.mpPoly = mpPath.Path(np.array(self.co_ordinates))
        self.car_density = 0
        self.timing = timing
        self.uptime = timing[1]
        self.average_uptime = timing[1]
        self.last_times = []
        self.LAST_TIMES_MAX = 5
    
    def __repr__(self) -> str:
        return f"{self.lane_id}: Cars: {self.count_cars},Vel:{self.average_velocity:.1f},Density:{self.car_density}"
    
    def visualize(self, frame:cv2.Mat) -> None :
        """Draws the bounding box as well as basic data for the lane

        Args:
            frame (cv2.Mat): Frame to draw onto
        """
        cv2.polylines(frame, [np.array(self.co_ordinates, np.int32)], True, self.color, 4)
        cv2.rectangle(frame, (int(self.co_ordinates[-1][0]), int(self.co_ordinates[-1][1]-20)), (int(self.co_ordinates[-1][0])+(len(self.__repr__()))*6, int(self.co_ordinates[-1][1])), self.color, -1)
        cv2.putText(frame, self.__repr__(),(int(self.co_ordinates[-1][0]), int(self.co_ordinates[-1][1]-7)),0, 0.37, (255,255,255),1, lineType=cv2.LINE_AA) 

    def fuzzy_equals(self,a : float, b: float, threshold : bool = 0.001) -> bool:
        return abs(a-b) <= threshold

    def update_lane(self, tracked_dict : Dict[int, Vehicle]) -> None:
        """Updates a lanes data given a dictionary of Vehicle objects to update with.

        Args:
            tracked_dict (Dict[int, Vehicle]): The dictionary containing tracker_id, Vehicle pairs to update from.
        """
        self.count_cars = 0
        sum_velocity = 0
        self.car_ids = set()
        for key in tracked_dict : 
            v = tracked_dict[key]
            if (self.isInLane(v)) : 
                self.count_cars += 1
                sum_velocity += v.velocity
                self.car_ids.add(v.id)
        self.average_velocity =  sum_velocity / max(self.count_cars,1)
        self.car_density = int((self.count_cars/self.area)*100000)
        time = min(self.timing[1] * (self.car_density / 30), self.timing[1])
        self.uptime = time if time > self.timing[0] else self.timing[0]
        self.last_times.append(self.uptime)
        if(len(self.last_times) > self.LAST_TIMES_MAX): self.last_times.pop(0)
        
        self.average_uptime = np.average(self.last_times)

    def isInLane (self, vehicle: Vehicle) -> bool :
        """Checks if the given vehicle object is in the current lane or not

        Args:
            vehicle (Vehicle): Vehicle to check

        Returns:
            bool: True if it is in the lane, False if not.
        """
        pt1 = self.co_ordinates[0]
        pt2 = self.co_ordinates[1]
        pt3 = self.co_ordinates[2]
        pt4 = self.co_ordinates[3]
        vehicle_point = vehicle.positions[-1]
        area1 = self.tri_area(pt1, pt2, vehicle_point)
        area2 = self.tri_area(pt2, pt3, vehicle_point)
        area3 = self.tri_area(pt3, pt4, vehicle_point)
        area4 = self.tri_area(pt1, pt4, vehicle_point)
        if not self.fuzzy_equals((area1 + area2 + area3 + area4), self.area) : 
            return False
        else : 
            return True
        
    def isInLane2(self,vehicle : Vehicle) -> bool:
        # Alternative implementation
        vehicle_point = vehicle.positions[-1]
        return self.mpPoly.contains_point(vehicle_point)
    
    def tri_area(self, a : tuple, b : tuple, c : tuple) -> float:
        return abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))/2)
    
    def get_area (self) -> float :
        pt1 = self.co_ordinates[0]
        pt2 = self.co_ordinates[1]
        pt3 = self.co_ordinates[2]
        pt4 = self.co_ordinates[3]

        t1 = pt1[0]*pt2[1] + pt2[0]*pt3[1] + pt3[0]*pt4[1] + pt4[0]*pt1[1]
        t2 = pt2[0]*pt1[1] + pt3[0]*pt2[1] + pt4[0]*pt3[1] + pt1[0]*pt4[1]
        
        return abs((t1 - t2)/2)


if __name__ == "__main__" :
    # test code, will not be run when imported
    POINTS = ((419, 474), (557, 483), (642, 229), (559, 224))
    
    l = Lane(POINTS)
    v = Vehicle(0, "Car", (942, 515, 998, 493))
    print(v.positions[-1])
    v2 = Vehicle(0, "Car", (1257, 506, 1318, 463))
    print(l.get_area())
    print(l.isInLane(v))
    print(l.isInLane(v2))
    print()

    PATH = r"C:\Users\Christo\Documents\Programming\AInML\traffic controller\atc-cyber-hackathon\data\rene_video.mov"
    v = cv2.VideoCapture(PATH)
    while True : 
        ret, frame  = v.read()
        l.visualize(frame)
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        

