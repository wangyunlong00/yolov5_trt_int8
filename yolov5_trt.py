import os
import sys

import tensorrt as trt
sys.path.insert(0, os.getcwd())

# from calibrator import yolov5EntropyCalibrator,Dataloader
from calibrator2 import yolov5EntropyCalibrator
from builder import TensorRTEngineBuilder,dict_get
from constants import TRT_LOGGER
from trt_utils import get_dyn_ranges
import pycuda.driver as cuda
import pycuda.autoinit



class yolov5(TensorRTEngineBuilder):


    def __init__(self, args):
        workspace_size = (8 << 30)
        super().__init__(args, "yolov5", workspace_size=workspace_size)

        # Model path
        # self.model_path = dict_get(args, "model_path", default="build/models/SSDResNet34/resnet34-ssd640.pytorch")
        self.onnx_file_path = "yolov5s.onnx"

        if self.precision == "int8":
            # cache_file = dict_get(self.args, "cache_file", default="yolov5s_calibration.cache")
            cache_file = dict_get(self.args, "cache_file", default="yolov5s_new.cache")
            batch = dict_get(self.args,'batch',default=100)
            batch_size = dict_get(self.args,'batch_size',default=8)
            calibration_dir = dict_get(self.args,'calibration_dir')
            image_height = dict_get(self.args,'image_height',default=640)
            image_width = dict_get(self.args,'image_width',default=640)

            calib_data_map = "G:/dataset/coco_data/data_maps/coco/val_map.txt"
            calib_image_dir = "G:/dataset/coco_data/preprocessed_data/coco/val2017/YOLOV5/fp32"
            shape = (batch_size,3,image_height,image_width)
            self.calibrator = yolov5EntropyCalibrator(calib_image_dir, cache_file, batch_size,
                                                           batch, False, calib_data_map)

            self.builder_config.int8_calibrator = self.calibrator
            self.cache_file = cache_file

    def initialize(self):
        # Create network.
        network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_creation_flag)

        # Populate network.
        # self.populate_network(self.network)

        parser = trt.OnnxParser(self.network,TRT_LOGGER)
        with open(self.onnx_file_path,'rb') as model:
            parser.parse(model.read())

        # Set input dtype and format
        input_tensor = self.network.get_input(0)
        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
            dynamic_range_dict = dict()
            if os.path.exists(self.cache_file):
                dynamic_range_dict = get_dyn_ranges(self.cache_file)
                input_dr = dynamic_range_dict.get("input", 0)
                input_tensor.set_dynamic_range(-input_dr, input_dr)
            else:
                print("WARNING: Calibration cache file not found! Calibration is required")
        if self.input_format == "linear":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        elif self.input_format == "chw4":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

        self.initialized = True
