import cv2

from model_wrapper import TrackerWrapped,Vehicle
from controller import Controller
PATH = r"C:\Users\zahra\Documents\Cyber Hackathon\traffic controller\atc-rework\data\rene_video.mov"

def main():
    con = Controller(PATH)
    con.main_loop()
            

if __name__ == '__main__':
    main()