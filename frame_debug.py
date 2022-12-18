import cv2

from model_wrapper import TrackerWrapped,Vehicle
from Lane import Lane
from controller import Controller
PATH = r".\data\rene_video.mov"
# POINTS = ((419, 474), (557, 483), (642, 229), (559, 224))

def main():
    con = Controller(PATH)
    con.load_lanes_from_json("lanes.json")
    con.main_loop()
            

if __name__ == '__main__':
    main()