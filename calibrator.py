import glob
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os
import cv2
from general import img_process
import sys
sys.path.insert(0, os.getcwd())

class Dataloader:
    def __init__(self,batch,batch_size,calibration_dir,image_height,image_width):
        self.batch=batch
        self.batch_size = batch_size
        self.image_height =image_height
        self.image_width =image_width
        self.img_list = glob.glob(os.path.join(calibration_dir,"*.jpg"))
        assert len(self.img_list) > self.batch*self.batch_size," need more  images"
        self.calibration_data = np.zeros((self.batch_size,3,self.image_height,self.image_width),dtype=np.float32)


    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.batch:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i+self.index*self.batch_size]) ,' not found image'
                print("  self.img_list[i+self.index*self.batch_size] ",self.img_list[i+self.index*self.batch_size])
                img = cv2.imread(self.img_list[i+self.index*self.batch_size])
                img = img_process(img,(self.image_height,self.image_width))
                self.calibration_data[i] = img
            self.index += 1
            return  np.ascontiguousarray(self.calibration_data,dtype=np.float)
        else:
            return np.array([])

    def __len__(self):
        return self.batch

class yolov5EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataset,cache_file,shape,force_calibration=False):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.device_input = cuda.mem_alloc(trt.volume(shape) * 4)
        self.force_calibration = force_calibration

        self.batches = dataset

        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if not self.force_calibration and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.cache = f.read()
        else:
            self.cache = None

    def get_batch_size(self):
        return self.shape[0]

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        try:
            # Get a single batch.
            data = next(self.batches)
            # Copy to device, then return a list containing pointers to input device buffers.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        return self.cache

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def clear_cache(self):
        self.cache = None
