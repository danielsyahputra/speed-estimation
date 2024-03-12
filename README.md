# Ambulance Detection and Tracking Using YOLOv8 with ONNX

Performing Object Detection for YOLOv8 with ONNX and ONNXRuntime

![! ONNX Ambulance Detection](https://github.com/danielsyahputra/yolov8-onnx/blob/master/output/ambulance.jpeg)

## Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

## Installation

```shell
git clone https://github.com/danielsyahputra/yolov9-onnx.git
cd yolov9-onnx
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

## ONNX model and Class metadata

You can download the onnx model and class metadata file on the link below

```
https://drive.google.com/drive/folders/10SswFdlZzkXrrIUa_xluAlsPxC6hp7bv?usp=sharing
```

## Examples

### Arguments
List  the arguments available in `configs/inference/main.yaml` file.

```
engine:
  onnx_path: weights/onnx/yolov8n/model.onnx
  classes_path: weights/onnx/yolov8n/metadata.yaml
  score_threshold: 0.1
  conf_threshold: 0.4
  iou_threshold: 0.4

source: tmp/ambulance.jpeg
save_path: output
mode: image
show: true
device: cpu
```

- `source`: Path to image or video file
- `onnx_path`: Path to yolov8 onnx file (ex: weights/onnx/yolov8n/model.onnx)
- `classes_path`: Path to yaml file that contains the list of class from model (ex: weights/onnx/yolov8n/metadata.yaml)
- `score_threshold`: Score threshold for inference, range from 0 - 1
- `conf_threshold`: Confidence threshold for inference, range from 0 - 1
- `iou_threshold`: IOU threshold for inference, range from 0 - 1
- `mode`: inference mode. Options: ['image', 'video']
- `save_path`: Path to save your prediction
- `show`: Show result on pop-up window
- `device`: Device use for inference, default = cpu.

Note: If you want to use `cuda` for inference, please make sure you are already install `onnxruntime-gpu` before running the script.

If you have your own custom model, don't forget to provide a yaml file that consists the list of class that your model want to predict. This is example of yaml content for defining your own classes:

```
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  .
  .
  .
  .
  n: object
```

### Running via Terminal

This project is built for ease of use. For example, 

```
python src/inference.py
```

But, if you want to change the mode or something without modifying the config file, you can simply define the args when running the script. For example:

```
python src/inference.py inference.mode=video inference.source=tmp/ambulance_2.mp4
```

The `inference` keyword mean that you want to modify the inference config in your configs folder. Keyword `mode`, `show` is one of the parameters which are defined as I mention before.

### Running via Docker

This way is recommended if you want to do on-premise deployment. Simply, just up the docker by running this command:

```
docker-compose up --build
```

Note: Please make sure your docker compose version. Some version use `docker compose` instead of `docker-compose` command.

If you want to run the engine again, modify the config and then run 

```
docker-compose up
```

![Running Inference on Image via Docker](https://github.com/danielsyahputra/yolov8-onnx/blob/master/assets/docker.png)
![Running Inference on Video via Docker](https://github.com/danielsyahputra/yolov8-onnx/blob/master/assets/docker_video.png)