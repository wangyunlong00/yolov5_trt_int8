
import numpy as np
import cv2
from general import img_process,non_max_suppression_cpu,scale_coords
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from runner import EngineRunner

CLASSNAME = ["person"        , "bicycle"   , "car"          , "motorcycle"   , "airplane", \

             "bus"           , "train"     , "truck"        , "boat"         , "traffic light", \

             "fire hydrant"  , "stop sign" , "parking meter", "bench"        , "bird", \

             "cat"           , "dog"       , "horse"        , "sheep"        , "cow", \

             "elephant"      , "bear"      , "zebra"        , "giraffe"      , "backpack", \

             "umbrella"      , "handbag"   , "tie"          , "suitcase"     , "frisbee", \

             "skis"          , "snowboard" , "sports ball"  , "kite"         , "baseball bat", \

             "baseball glove", "skateboard", "surfboard"    , "tennis racket", "bottle", \

             "wine glass"    , "cup"       , "fork"         , "knife"        , "spoon", \

             "bowl"          , "banana"    , "apple"        , "sandwich"     , "orange", \

             "broccoli"      , "carrot"    , "hot dog"      , "pizza"        , "donut", \

             "cake"          , "chair"     , "couch"        , "potted plant" , "bed", \

             "dining table"  , "toilet"    , "tv"           , "laptop"       , "mouse",

             "remote"        , "keyboard"  , "cell phone"   , "microwave"    , "oven", \

             "toaster"       , "sink"      , "refrigerator" , "book"         , "clock", \

             "vase"          , "scissors"  , "teddy bear"   , "hair drier"   , "toothbrush"]




def run_yolov5(engine_file, batch_size, num_images, verbose=False, output_file="build/out/SSDResNet34/dump.json"):
    runner = EngineRunner(engine_file, verbose=verbose)
    print(" load success ")
    image1_path = "E:/project/python/work/amba_project/image/000000000025.jpg"
    image1 = cv2.imread(image1_path)

    image_height = 640
    image_width = 640

    image11 = img_process(image1, (image_height, image_width))

    image11 = np.array(image11)

    outputs = runner([image11], batch_size=1)
    trt_detections = outputs[0]
    trt_detections = np.reshape(trt_detections,(1,25200,85))

    yolov5_pred_result = non_max_suppression_cpu(torch.from_numpy(trt_detections), conf_thres=0.3, iou_thres=0.5)
    for index, det in enumerate(yolov5_pred_result):  # detections per imag
        # print(" index ",det)
        if len(det):

            det[:, :4] = scale_coords((image_height, image_width), det[:, :4], image1.shape[:2]).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = CLASSNAME[c]
                cv2.rectangle(image1, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 100, 0),
                              thickness=2)
                cv2.putText(image1, str(label), ((int(xyxy[0]), int(xyxy[1]))), cv2.FONT_HERSHEY_SIMPLEX,
                            0.85,
                            (100, 0, 50), thickness=2)

    cv2.imshow("xxxxxxxxxx ", image1)
    cv2.waitKey(0)


    # print(" trt_detections ",trt_detections)
def main():


    threshold = 0.20
    # model_path = "/wyl/download/project/amba_project/tensorrt_int8/ssd-resnet34-gpu-b8-int8.plan"
    # model_path = "/wyl/download/project/amba_project/tensorrt_int8/model_file/ssd-resnet34-gpu-b8-int8.plan"
    # model_path = "D:/project/python/study/yolov5/0105/yolov5_trt_int8_file/yolov5-gpu-b1-int8_calibration_int8.plan"
    # model_path = "D:/project/python/study/yolov5/0105/yolov5_trt_int8_file/yolov5n_new-gpu-b1-fp16_calibration_int8.plan"
    # model_path = "yolov5s_int8_2.trt"
    # model_path = "yolov5s_int8.plan"
    model_path = "yolov5s_fp16.plan"
    map_score = run_yolov5(model_path, 8, 500,
                                         verbose=True)


if __name__ == "__main__":
    main()
