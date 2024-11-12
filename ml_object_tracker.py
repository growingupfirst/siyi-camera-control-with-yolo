import cv2
from ultralytics import YOLO
import numpy as np
import cv2
from time import sleep, time
import threading
from multiprocessing import Process, Queue
from siyi_control import SIYIControl
from XAnnotator import XAnnotator
import supervision as sv
import queue

class VideoCapture:

  def __init__(self, name):
    
    self.cap = cv2.VideoCapture(name, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      # frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)
  def read(self):
    # print("qsize", self.q.qsize())
    return self.q.get()
  
model = YOLO(r"train_200_09_10.engine.Orin.fp16.1.1.engine", task='detect')

#CUSTOM COLORS FOR CLASSES
# Define classes and their ids
# cls_list = {0:'plane',
#             1:'helicopter',
#             2:'uav',
#             3:'bird',
#             4:'drone',
#             5:'other'}
# # Define a color list for track visualization
# colors = {0: sv.Color(0, 0, 255),
#           1: sv.Color(255,255,255),
#           2: sv.Color(191, 0, 191),
#           3: sv.Color(255, 127, 0),
#           4: sv.Color(255, 0, 0),
#           5: sv.Color(0, 127, 0)}
# color_palette = sv.ColorPalette([color for color in colors.values()])

# Instantiate a tracker
tracker = sv.ByteTrack(minimum_consecutive_frames=2)

# Initialize annotators
label_annotator = sv.LabelAnnotator() # color=color_palette)
corner_annotator = sv.BoxCornerAnnotator() #color=color_palette, thickness=2)
smoother = sv.DetectionsSmoother()
x_annotator = XAnnotator()#color=color_palette)
#Get classes list
cls_list = model.names
# Set the callback
def callback(frame: np.ndarray) -> np.ndarray:
    results = model.predict(frame, conf=0.4, iou=0.5, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)
    # labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
    labels = [
        f"#{track_id} {cls_list[class_id]} {confidence:.2f}"
        for class_id, confidence, track_id
        in zip(detections.class_id, detections.confidence, detections.tracker_id)
      ]
    annotated_frame = corner_annotator.annotate(
        scene=frame, detections=detections)
    annotated_frame = x_annotator.annotate(
        scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

if __name__ == "__main__":
    RTSP_URL = "rtsp://192.168.144.25:8554/main.264"
    siyi_cap = VideoCapture(RTSP_URL)

    cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)
    siyi_control = SIYIControl()

    while True:
        frame = siyi_cap.read()
        # ML track and detection
        start_time = time()
        predicted_frame = callback(frame)
        end_time = time() - start_time
        cv2.putText(predicted_frame, f'{float(end_time):.3f}', (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,0),2)                 
        # Show window    
        cv2.imshow("output", predicted_frame)

        pressed = cv2.waitKey(1)
        if pressed in [27]:
            # Pressed Esc to cancel
            break
        
    cv2.destroyAllWindows()
    siyi_cap.cap.release()
