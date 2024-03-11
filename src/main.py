import os
import sys
import cv2
import time
import pyrootutils
import numpy as np
import pandas as pd
from typing import List, Dict
from numpy.linalg import norm

ROOT = pyrootutils.setup_root(
    search_from=os.path.dirname('__file__'),
    indicator=["requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

from src.scripts.engine.detection import Detector
from src.scripts.engine.track import Tracker

PIXELS_PER_METER = 100

def calculate_speed(prev_centroid, curr_centroid, fps):
    dx = curr_centroid[0] - prev_centroid[0]
    dy = curr_centroid[1] - prev_centroid[1]
    distance = (dx ** 2 + dy ** 2) ** 0.5
    speed = (distance * fps) / PIXELS_PER_METER
    return speed

def draw_box(img, detections, fps=None, show_attributes=True):
    box_color = (131, 107, 49)
    font_color = (191, 191, 213)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    for detection in detections:
        box = detection["bbox_xywh"]
        classId = detection["id"]
        confidence = detection["confidence"]
        attributes: dict = detection["attributes"]
        class_name = detection["class_name"]
        # print( f"Bbox {i} Class: {classId} Confidence: {confidence} Scaled coords: [ cx: {(box[0] + (box[2] / 2)) / img.shape[1]}, cy: {(box[1] + (box[3] / 2)) / img.shape[0]}, w: {box[2]/ img.shape[1]}, h: {box[3] / img.shape[0]} ]" )
        xmax = box[0] + box[2]
        ymax = box[1] + box[3]

        text = f"{class_name} - {str(round(confidence,2))}"
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(xmax), int(ymax)), box_color, 3)
        img = cv2.rectangle(img, (int(box[0]), int(box[1]) - text_h), (int(box[0]+text_w), int(box[1])), box_color, cv2.FILLED)
        img = cv2.putText(img, text, (int(box[0]), int(box[1])), font, font_scale, font_color, font_thickness)

        if show_attributes:
            for i, (key, val) in enumerate(attributes.items(),start=1):
                text = f"{key}: {val}"
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                img = cv2.rectangle(img, (int(box[0]), int(box[1]) - text_h + i*text_h), (int(box[0]+text_w), int(box[1]) + i*text_h), box_color, cv2.FILLED)
                img = cv2.putText(img, text, (int(box[0]), int(box[1]) + i*text_h), font, font_scale, font_color, font_thickness)


    if fps is not None:
        img = cv2.putText(img, f"FPS: {str(round(fps, 2))}", (20,40), font, .6, font_color)
    return img

if __name__=="__main__":
    weight_path = f"{ROOT}/weights/onnx/yolov8n"
    cap = cv2.VideoCapture(str(ROOT / "tmp/ambulance_2.mp4"))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    detector = Detector(model_path=f"{weight_path}/model.onnx", 
                        class_mapping_path=f"{weight_path}/metadata.yaml", 
                        original_size=(w, h),
                        conf_thresold=0.5)
    tracker = Tracker(max_lost=2, classes=detector.classes)

    prev_centroid_states = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t1 = time.perf_counter()

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        speed_output = []


        for track in tracks:
            track_id = track['id']
            bbox_xyxy = track["bbox_xyxy"]
            centroid_box = track["centroid_box"].astype(int)
            cx, cy = centroid_box

            if track_id in prev_centroid_states.keys():
                speed = calculate_speed(prev_centroid_states[track_id]["centroid"], 
                                        centroid_box, video_fps)
                print(speed)
                speed_history = prev_centroid_states[track_id]["speed_history"]

                if len(speed_history) > 10:
                    speed_history.pop(0)
                    speed_history.append(speed)
                else:
                    speed_history.append(speed)
                
                average = sum(speed_history) / len(speed_history)
            else:
                speed = 0
                speed_history = []
                average = 0
            
            prev_centroid_states[track_id] = {"centroid": centroid_box, "speed_history": speed_history}
            track["attributes"] = {"s": f"{round(average * 3.6, 2)}km/h"}
            speed_output.append(track)
        
        t2 = time.perf_counter()

        frame = draw_box(frame, detections=speed_output, fps=1 / (t2 - t1))

        cv2.imshow("Preview", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()