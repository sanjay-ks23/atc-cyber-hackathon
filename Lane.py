from typing import Dict, Set, Tuple
import cv2
import numpy as np
from matplotlib import path as mpPath

from model_wrapper import Vehicle

class Lane : 
    color = (15, 220, 10)
    def __init__(self, co_ordinates) -> None:
        self.co_ordinates: Tuple[Tuple[int, int]] = co_ordinates
        self.direction : int = 1 # 1 -> forward, -1-> backward
        self.average_velocity : float = 0 
        self.count_cars : int = 0
        self.area : float = self.get_area()
        self.car_ids : Set[int] = set()
        self.mpPoly = mpPath.Path(np.array(self.co_ordinates))
    
    def __repr__(self) -> str:
        return f"Cars: {self.count_cars},Avg. Velocity: {self.average_velocity}"
    
    def visualize(self, frame:cv2.Mat) -> None :

        cv2.polylines(frame, [np.array(self.co_ordinates, np.int32)], True, self.color, 6)
        cv2.rectangle(frame, (int(self.co_ordinates[0][0]), int(self.co_ordinates[0][1]-30)), (int(self.co_ordinates[0][0])+(len(self.__repr__()))*10, int(self.co_ordinates[0][1])), self.color, -1)
        cv2.putText(frame, self.__repr__(),(int(self.co_ordinates[0][0]), int(self.co_ordinates[0][1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    

    def fuzzy_equals(self,a : float, b: float, threshold : bool = 0.001) -> bool:
        return abs(a-b) <= threshold

    def update_lane(self, tracked_dict : Dict[int, Vehicle]) -> None:
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

    def isInLane (self, vehicle: Vehicle) -> bool :
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
        

