
import ctypes
import os
import sys


import argparse
import json
import time
sys.path.insert(0, os.getcwd())
from runner import EngineRunner, get_input_format
import numpy as np
import torch
import tensorrt as trt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def run_SSDResNet34_accuracy(engine_file, batch_size, num_images, verbose=False, output_file="build/out/SSDResNet34/dump.json"):
    runner = EngineRunner(engine_file, verbose=verbose)
    input_dtype, input_format = get_input_format(runner.engine)
    print(" input_dtype ",input_dtype)
    if input_dtype == trt.DataType.FLOAT:
        format_string = "fp32"
    elif input_dtype == trt.DataType.INT8:
        if input_format == trt.TensorFormat.LINEAR:
            format_string = "int8_linear"
        elif input_format == trt.TensorFormat.CHW4:
            format_string = "int8_chw4"
    image_dir = os.path.join("G:/dataset/coco_data/preprocessed_data/coco/val2017/YOLOV5", format_string)

    val_annotate = os.path.join("G:/dataset/coco_data/preprocessed_data/coco/annotations/instances_val2017.json")
    calib_data_map = "G:/dataset/coco_data/data_maps/coco/val_map.txt"
    coco = COCO(annotation_file=val_annotate)

    image_ids = []
    with open(calib_data_map) as f:
        for line in f:
            image_ids.append(int(line[:12].strip()))

    # img_list = os.listdir(image_dir)
    # image_ids=[]
    # for img_name in img_list:
    #     image_ids.append(int(img_name[:12]))

    # image_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    # Class 0 is background
    cat_ids.insert(0, 0)
    num_images = min(num_images, len(image_ids))
    coco_detections = []

    batch_idx = 0
    for image_idx in range(0, num_images, batch_size):
        end_idx = min(image_idx + batch_size, num_images)
        img = []
        img_sizes = []
        for idx in range(image_idx, end_idx):
            image_id = image_ids[idx]
            img.append(np.load(os.path.join(image_dir, coco.imgs[image_id]["file_name"] + ".npy")))
            img_sizes.append([coco.imgs[image_id]["height"], coco.imgs[image_id]["width"]])

        img = np.stack(img)

        start_time = time.time()
        outputs = runner([img], batch_size=batch_size)
        trt_detections = outputs[0]
        if verbose:
            print("Batch {:d} >> Inference time:  {:f}".format(batch_idx, time.time() - start_time))

        for idx in range(0, end_idx - image_idx):
            keep_count = trt_detections[idx * (200 * 7 + 1) + 200 * 7].view('int32')
            trt_detections_batch = trt_detections[idx * (200 * 7 + 1):idx * (200 * 7 + 1) + keep_count * 7].reshape(keep_count, 7)
            image_height = img_sizes[idx][0]
            image_width = img_sizes[idx][1]
            for prediction_idx in range(0, keep_count):
                loc = trt_detections_batch[prediction_idx, [2, 1, 4, 3]]
                label = trt_detections_batch[prediction_idx, 6]
                score = float(trt_detections_batch[prediction_idx, 5])

                bbox_coco_fmt = [
                    loc[0] * image_width,
                    loc[1] * image_height,
                    (loc[2] - loc[0]) * image_width,
                    (loc[3] - loc[1]) * image_height,
                ]

                coco_detection = {
                    "image_id": image_ids[image_idx + idx],
                    "category_id": cat_ids[int(label)],
                    "bbox": bbox_coco_fmt,
                    "score": score,
                }
                coco_detections.append(coco_detection)

        batch_idx += 1

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w") as f:
        json.dump(coco_detections, f)

    cocoDt = coco.loadRes(output_file)
    eval = COCOeval(coco, cocoDt, 'bbox')
    eval.params.imgIds = image_ids[:num_images]
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    map_score = eval.stats[0]
    return map_score


def main():


    threshold = 0.20
    # model_path = "/wyl/download/project/amba_project/tensorrt_int8/ssd-resnet34-gpu-b8-int8.plan"
    # model_path = "/wyl/download/project/amba_project/tensorrt_int8/model_file/ssd-resnet34-gpu-b8-int8.plan"
    # model_path = "D:/project/python/study/yolov5/0105/yolov5_trt_int8_file/yolov5-gpu-b1-int8_calibration_int8.plan"
    # model_path = "D:/project/python/study/yolov5/0105/yolov5_trt_int8_file/yolov5n_new-gpu-b1-fp16_calibration_int8.plan"
    model_path = "yolov5s_int8_2.trt"
    map_score = run_SSDResNet34_accuracy(model_path, 8, 500,
                                         verbose=True)
    print("Get mAP score = {:f} Target = {:f}".format(map_score, threshold))


if __name__ == "__main__":
    main()
