import cv2
import os
from ultralytics import YOLO
import pandas as pd

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


# Get the list of image files in the input folder
image_list = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

for image_path in image_list:
    img = cv2.imread(os.path.join(input_folder, image_path))
    img = cv2.resize(img, (frame_width, frame_height))

    # Perform object detection to get the detected objects and their class labels.
    REC_FLAG = perform_object_detection(img)
    REC = REC_FLAG

    # Check the REC flag and handle saving the frame.
    if REC:
        detected_frames.append(img)
        out.write(img)

    # If no cars detected, set REC to False and stop saving frames.
    if not REC:
        detected_frames = []
        out.release()

# Release the video writer if not already released (in case REC is True till the end of the loop).
out.release()
