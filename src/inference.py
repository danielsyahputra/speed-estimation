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
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from src.scripts.engine.detection import Detector
from src.scripts.utils.logger import get_logger

log = get_logger()



class Runner:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def run(self):
        weights_path = self.cfg.inference.engine.onnx_path
        classes_path = self.cfg.inference.engine.classes_path
        source_path = self.cfg.inference.source
        assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
        assert os.path.isfile(classes_path), f"There's no classes file with name {weights_path}"
        assert os.path.isfile(source_path), f"There's no source file with name {weights_path}"

        mode = self.cfg.inference.mode 
        if mode == "image":
            cap = cv2.imread(source_path)
            h,w = cap.shape[:2]
            log.info("Successfully load the image")
        elif mode == "video":
            cap = cv2.VideoCapture(source_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info("Successfully load the video")
        else:
            raise ValueError("You can't process the result because you have not define the source type (video or image) in the argument")

        
        detector = Detector(model_path=weights_path,
            class_mapping_path=classes_path,
            original_size=(w, h),
            score_threshold=self.cfg.inference.engine.score_threshold,
            conf_thresold=self.cfg.inference.engine.conf_threshold,
            iou_threshold=self.cfg.inference.engine.iou_threshold,
            device=self.cfg.inference.device
        )

        if mode == "image":
            self.inference_on_image(detector=detector, image=cap)
        else:
            self.inference_on_video(detector=detector, cap=cap)

    def inference_on_image(self, detector: Detector, image: np.ndarray):
        log.info("Start inference on image")
        detections = detector.detect(image)
        detector.draw_detections(image, detections=detections)

        output_path = f"output/{Path(self.cfg.inference.source).name}"
        log.info(f"Saving result on {output_path}")
        cv2.imwrite(output_path, image)

        if self.cfg.inference.show:
            cv2.imshow("Result", image)
            cv2.waitKey(0)

    def inference_on_video(self, detector: Detector, cap: cv2.VideoCapture):
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        save_path = self.cfg.inference.save_path
        
        writer = cv2.VideoWriter(f'{save_path}/result.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_fps, (w, h))

        log.info("Start inference on video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t1 = time.perf_counter()
            detections = detector.detect(frame)
            processing_time = time.perf_counter() - t1
            fps = 1 / processing_time
            detector.draw_detections(frame, detections=detections)
            log.info(f"succesfully process frame with fps = {round(fps, 3)}")

            writer.write(frame)

            cv2.imshow("Result", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        log.info(f"Finish. Saving result to {save_path}/result.avi")

if __name__=="__main__":
    @hydra.main(config_path=f"{ROOT}/configs", config_name="main", version_base=None)
    def main(cfg: DictConfig) -> None:
        """Main function."""
        runner = Runner(cfg)
        runner.run()
    main()