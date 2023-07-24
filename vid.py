import cv2
import os
from ultralytics import YOLO
import pandas as pd
import time

model = YOLO("./bike.pt")

# Assuming you have a function called `perform_object_detection` for object detection.
input_video_path = "vid.mp4"
input_folder = "./images"
output_video_path = "output_video.mp4"
frame_width, frame_height = 640, 480
fps = 24

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

REC = False
detected_frames = []

def perform_object_detection(img):
    x = model.predict(source=img, show=True)
    a = x[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    for index, row in px.iterrows():
        cls = int(row[5])
        # car
        if cls == 2:
            return True
    return False

cap = cv2.VideoCapture(input_video_path)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (frame_width, frame_height))

    # Perform object detection to get the detected objects and their class labels.
    REC_FLAG = perform_object_detection(img)

    # Check the REC flag and handle saving the frame.
    if REC_FLAG:
        detected_frames.append(img)

    # If REC_FLAG is True and REC is False, start a new output video and set REC to True.
    if REC_FLAG and not REC:
        REC = True
        output_video_path = "output_video_" + str(time.time()) + ".mp4"
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # If REC_FLAG is False and REC is True, stop saving frames and release the video writer.
    if not REC_FLAG and REC:
        REC = False
        out.release()

    # Write the frame to the output video if REC is True.
    if REC:
        out.write(img)

# Release the video writer if not already released (in case REC is True till the end of the loop).
if REC:
    out.release()

cap.release()
