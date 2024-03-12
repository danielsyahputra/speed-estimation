import os
import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=os.path.dirname('__file__'),
    indicator=["requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import hydra
import time
from pathlib import Path
from omegaconf import DictConfig

from src.scripts.engine.detection import Detector
from src.scripts.utils.logger import get_logger

log = get_logger()

def get_detector(cfg: DictConfig):
    weights_path = cfg.inference.engine.onnx_path
    classes_path = cfg.inference.engine.classes_path
    source_path = cfg.inference.source
    assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
    assert os.path.isfile(classes_path), f"There's no classes file with name {weights_path}"
    assert os.path.isfile(source_path), f"There's no source file with name {weights_path}"

    if cfg.inference.mode == "image":
        image = cv2.imread(source_path)
        h,w = image.shape[:2]
    else: 
        cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    detector = Detector(model_path=weights_path,
                      class_mapping_path=classes_path,
                      original_size=(w, h),
                      score_threshold=cfg.inference.engine.score_threshold,
                      conf_thresold=cfg.inference.engine.conf_threshold,
                      iou_threshold=cfg.inference.engine.iou_threshold,
                      device=cfg.inference.device)
    return detector

def inference_on_image(cfg: DictConfig):
    print("[INFO] Intialize Model")
    detector = get_detector(cfg)
    image = cv2.imread(cfg.inference.source)

    print("[INFO] Inference Image")
    detections = detector.detect(image)
    detector.draw_detections(image, detections=detections)

    output_path = f"output/{Path(cfg.inference.source).name}"
    print(f"[INFO] Saving result on {output_path}")
    cv2.imwrite(output_path, image)

    if cfg.inference.show:
        cv2.imshow("Result", image)
        cv2.waitKey(0)

def inference_on_video(cfg):
    print("[INFO] Intialize Model")
    detector = get_detector(cfg)

    cap = cv2.VideoCapture(cfg.inference.source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter('output/result.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_fps, (w, h))

    print("[INFO] Inference on Video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        detector.draw_detections(frame, detections=detections)
        writer.write(frame)
        cv2.imshow("Result", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    print("[INFO] Finish. Saving result to output/result.avi")

if __name__=="__main__":
    @hydra.main(config_path=f"{ROOT}/configs", config_name="main", version_base=None)
    def main(cfg: DictConfig) -> None:
        """Main function."""
        if cfg.inference.mode == "video":
            inference_on_video(cfg)
        elif cfg.inference.mode == "image":
            inference_on_image(cfg)   
        else:
            raise ValueError("You can't process the result because you have not define the source type (video or image) in the argument")
    main()