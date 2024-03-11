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

from src.scripts.engine.track.tracker import Tracker as CentroidTracker

class Tracker(CentroidTracker):
    def __init__(self, max_lost=2, classes: Dict = None):
        super().__init__(max_lost=max_lost, tracker_output_format='visdrone_challenge')

        self.classes = classes

    def xyxy2xywh(self, xyxy: np.ndarray) -> np.ndarray:
        """
        Convert bounding box coordinates from (xmin, ymin, xmax, ymax) format to (xmin, ymin, width, height).
    
        Args:
            xyxy (numpy.ndarray):
    
        Returns:
            numpy.ndarray: Bounding box coordinates (xmin, ymin, width, height).
    
        """
    
        if len(xyxy.shape) == 2:
            w, h = xyxy[:, 2] - xyxy[:, 0] + 1, xyxy[:, 3] - xyxy[:, 1] + 1
            xywh = np.concatenate((xyxy[:, 0:2], w[:, None], h[:, None]), axis=1)
            return xywh.astype("int")
        elif len(xyxy.shape) == 1:
            (left, top, right, bottom) = xyxy
            width = right - left + 1
            height = bottom - top + 1
            return np.array([left, top, width, height]).astype('int')
        else:
            raise ValueError("Input shape not compatible.")
            
    def xywh2xyxy(self, bbox):
        x,y,w,h = bbox
        return np.array([int(x),int(y), int(x+w), int(y+h)])

    def centroidxywh_xywh(self, bbox):
        x, y, w, h = bbox
        w2 = int(w/2)
        h2 = int(h/2)
        return np.array([int(x-w2), int(y-h2), int(w), int(h)])

    def centroid_xywh(self, bbox):
        x, y, w, h = bbox
        x = int(x + int(w/2))
        y = int(y + int(h/2))
        return np.array([x,y])

    def update(self, detections: List) -> List:
        tracks_list = []
        track_dict = {}
        if len(detections) == 0:
            return tracks_list
        xywhs = self.xyxy2xywh(np.array([x["box"] for x in detections]))
        confs = np.array([x["confidence"] for x in detections])
        clss = np.array([x["class_index"] for x in detections])
        tracks = super().update(xywhs, confs, clss)
        for (frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class_id, y, z) in tracks:
            bbox_xywh = np.array([bb_left, bb_top, bb_width, bb_height])
            bbox_xyxy = self.xywh2xyxy(bbox_xywh)
            centroid = self.centroid_xywh(bbox_xywh)
            track_dict = {"frame": frame, "id": track_id, "bbox_xywh": bbox_xywh, "bbox_xyxy": bbox_xyxy, "centroid_box":centroid,
                          "y":y, "z":z, "class_id": class_id, "confidence": conf}
            if self.classes is not None:
                track_dict["class_name"] = self.classes[class_id]
            tracks_list.append(track_dict)
        return tracks_list