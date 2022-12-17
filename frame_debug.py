import cv2

from model_wrapper import TrackerWrapped


def main():
    tw = TrackerWrapped()
    m = cv2.VideoCapture()