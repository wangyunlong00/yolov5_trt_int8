

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
import numpy as np
import shutil

from image_preprocessor import ImagePreprocessor
import cv2
import math
from general import img_process

def preprocess_coco_for_yolov5(data_dir, preprocessed_data_dir, formats, overwrite=False, cal_only=False, val_only=False):
    def loader(file):
        image = cv2.imread(file)
        image = img_process(image, (640, 640))
        return image

    def quantizer(image):
        # Dynamic range of image is [-2.64064, 2.64064] based on calibration cache.
        max_abs = 2.64064
        image_int8 = image.clip(-max_abs, max_abs) / max_abs * 127.0
        return image_int8.astype(dtype=np.int8, order='C')
    preprocessor = ImagePreprocessor(loader, None)
    if not val_only:
        # Preprocess calibration set. FP32 only because calibrator always takes FP32 input.
        preprocessor.run(os.path.join(data_dir, "coco", "train2017"),
                         os.path.join(preprocessed_data_dir, "coco", "train2017", "SSDResNet34"),
                         "/wyl/download/project/NVIDIA/data_maps/coco/cal_map.txt", ["fp32"], overwrite)
    if not cal_only:
        # Preprocess validation set.
        # preprocessor.run(os.path.join(data_dir, "coco", "val2017"),
        #                  os.path.join(preprocessed_data_dir, "coco", "val2017", "SSDResNet34"),
        #                  "/wyl/download/project/NVIDIA/data_maps/coco/val_map.txt", formats, overwrite)



        preprocessor.run(os.path.join(data_dir, "coco", "val2017"),
                         os.path.join(preprocessed_data_dir, "coco", "val2017", "YOLOV5"),
                         "G:/dataset/coco_data/data_maps/coco/val_map.txt", formats, overwrite)


def copy_coco_annotations(data_dir, preprocessed_data_dir):
    src_dir = os.path.join(data_dir, "coco/annotations")
    dst_dir = os.path.join(preprocessed_data_dir, "coco/annotations")
    if not os.path.exists(dst_dir):
        shutil.copytree(src_dir, dst_dir)


def main():
    # Parse arguments to identify the data directory with the input images
    #   and the output directory for the preprocessed images.
    # The data dicretory is assumed to have the following structure:
    # <data_dir>
    #  └── coco
    #      ├── annotations
    #      ├── train2017
    #      └── val2017
    # And the output directory will have the following structure:
    # <preprocessed_data_dir>
    #  └── coco
    #      ├── annotations
    #      ├── train2017
    #      │   └── SSDResNet34
    #      │       └── fp32
    #      └── val2017
    #          └── SSDResNet34
    #              └── int8_linear
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-d",
        help="Specifies the directory containing the input images.",
        default="G:/dataset/coco_data/coco_ori/"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Specifies the output directory for the preprocessed data.",
        default="G:/dataset/coco_data/preprocessed_data/"
    )
    parser.add_argument(
        "--formats", "-t",
        help="Comma-separated list of formats. Choices: fp32, int8_linear, int8_chw4.",
        default="default"
    )
    parser.add_argument(
        "--overwrite", "-f",
        help="Overwrite existing files.",
        action="store_true"
    )
    parser.add_argument(
        "--cal_only",
        help="Only preprocess calibration set.",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--val_only",
        help="Only preprocess validation set.",
        default=True,
        action="store_true"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    preprocessed_data_dir = args.preprocessed_data_dir
    formats = args.formats.split(",")
    overwrite = args.overwrite
    cal_only = args.cal_only
    val_only = args.val_only
    # default_formats = ["int8_linear"]
    default_formats = ["fp32"]

    # Now, actually preprocess the input images
    if args.formats == "default":
        formats = default_formats
    preprocess_coco_for_yolov5(data_dir, preprocessed_data_dir, formats, overwrite, cal_only, val_only)

    # Copy annotations from data_dir to preprocessed_data_dir.
    copy_coco_annotations(data_dir, preprocessed_data_dir)



if __name__ == '__main__':
    main()
