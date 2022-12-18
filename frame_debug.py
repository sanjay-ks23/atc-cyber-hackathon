import cv2

from model_wrapper import TrackerWrapped


def main():
    tw = TrackerWrapped()
    vid = cv2.VideoCapture(r"C:\Users\zahra\Documents\Cyber Hackathon\traffic controller\atc-rework\data\rene_video.mov")
    frame_num = 0
    while True: # while video is running
        return_value, frame = vid.read()
        if not return_value:
            print('Video has ended or failed!')
            break
        frame_num +=1
        results = tw.track_frame(frame)
        for i, key in enumerate(results.keys()):
            if(i > 5):
                break
            print(results[key])

if __name__ == '__main__':
    main()