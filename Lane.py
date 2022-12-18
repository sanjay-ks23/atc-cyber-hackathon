from ast import Tuple
import cv2
from model_wrapper import Vehicle

class Lane : 

    def __init__(self, co_ordinates) -> None:
        self.co_ordinates: Tuple[Tuple[int, int]] = co_ordinates
        self.direction : int = 1 # 1 -> forward, -1-> backward
        self.average_velocity = 0 
        self.count_cars = 0
    
    def visualize(self, frame:cv2.Mat) -> None :
        pass

    def isInLane (self, veh: Vehicle) -> bool :
        return False