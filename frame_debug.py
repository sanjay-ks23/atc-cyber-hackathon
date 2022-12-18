import cv2

from model_wrapper import TrackerWrapped,Vehicle
from copy import deepcopy
PATH = r"C:\Users\Christo\Documents\Programming\AInML\traffic controller\atc-cyber-hackathon\data\rene_video.mov"

def main():
    tw = TrackerWrapped()
    vid = cv2.VideoCapture(PATH)
    frame_num = 0
    time = 0
    frame_rate = 30
    tracked = {}
    while True: # while video is running
        return_value, frame = vid.read()
        if not return_value:
            print('Video has ended or failed!')
            break
        frame_num +=1
        time += 1/frame_rate
        if(frame_num % 10 != 0):
            continue
        
        
        results = tw.track_frame(frame)
        tracked_ids = list(tracked.keys())
        result_ids = list(results.keys())
        for key in result_ids:
            car = results[key]
            if key in tracked_ids:
                tracked[key].update_pos(car.bbox,time)
            else:
                tracked[key] = Vehicle(car.id,car.vehicle_class,car.bbox,time)
        if(frame_num%frame_rate*2 == 0):
            for key in tracked_ids:
                if(not(key in result_ids)):
                    tracked.pop(key)
        # print(len(tracked_ids))
        for key in results.keys():
            tracked[key].draw_visualisation(frame,(0,255,0))
        
        cv2.imshow("Video out",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            

if __name__ == '__main__':
    main()