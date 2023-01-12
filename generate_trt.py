from yolov5_trt import yolov5
from constants import Scenario

args = {"scenario": Scenario.SingleStream, "batch_size": 1, "engine_dir": "./",
        'calibration_dir': "G:/dataset/coco_data/coco_ori/coco/val2017"}
yolov5_instance = yolov5(args)
yolov5_instance.initialize()
yolov5_instance.build_engines()
