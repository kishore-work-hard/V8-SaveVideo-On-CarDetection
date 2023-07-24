# V8-SaveVideo-On-CarDetection

This Python script is an implementation of object detection in videos using YOLO (You Only Look Once) model with the Ultralytics library and OpenCV. The purpose of this script is to detect a specific object class (in this case, cars) in the input video and save the frames in which cars are detected as separate output videos.

Let's break down the script step by step:

1. Importing required libraries:
   - `cv2`: OpenCV library for image and video processing.
   - `os`: Operating system library for file handling.
   - `ultralytics.YOLO`: A YOLO implementation from the Ultralytics library.
   - `pandas as pd`: Pandas library for data manipulation.
   - `time`: Time library for handling time-related operations.

2. Initializing the YOLO model:
   The script initializes a YOLO model by loading pre-trained weights from a file called "bike.pt" using Ultralytics. This model will be used for object detection.

3. Setting up video input and output paths:
   - `input_video_path`: The path to the input video ("vid.mp4").
   - `input_folder`: The path to the folder where detected frames will be stored.
   - `output_video_path`: The path to the output video ("output_video.mp4") where frames with detected cars will be saved.
   - `frame_width`, `frame_height`: The dimensions (width and height) of the resized frames.
   - `fps`: The frames per second for the output video.

4. Initializing video writer:
   The script sets up a video writer using OpenCV with the given output path, codec, frame rate, and frame dimensions.

5. Object detection and recording:
   The script performs object detection on each frame of the input video and records frames containing detected cars in a list called `detected_frames`.

6. Handling the REC flag:
   The script uses a flag named `REC` to indicate whether it should be recording frames to the output video. The flag `REC` is set to False initially.

7. `perform_object_detection` function:
   This function takes an image (`img`) as input and uses the YOLO model to perform object detection. It returns True if it detects a car in the image; otherwise, it returns False.

8. Main loop for processing the video:
   The script uses a loop to read each frame of the input video (`cap`), resizes it, and then calls the `perform_object_detection` function to check if a car is detected in the frame.

   - If `REC_FLAG` (the result of object detection) is True and `REC` is False, it means a car has been detected, and a new output video is started with a unique name (based on the current timestamp). The `REC` flag is set to True to indicate that frames should be recorded to the new output video.

   - If `REC_FLAG` is False and `REC` is True, it means the detection of cars has stopped, and the script stops recording frames to the output video by releasing the video writer (`out`).

   - If `REC_FLAG` is True and `REC` is also True, it means the script is still recording frames, so the current frame is added to the list of detected frames (`detected_frames`).

   - If `REC_FLAG` is False and `REC` is also False, it means no car is detected, and the script does not perform any recording.

9. Final steps:
   Once the video processing loop is completed, the script releases the video writer (`out`) if `REC` is True (i.e., the script was recording frames until the end). It also releases the input video capture (`cap`) to free up system resources.

In summary, this script reads a video, performs object detection to detect cars, and saves the frames with detected cars as separate output videos. The output videos are named based on the timestamp when the detection started for each instance of detecting cars in the input video.
