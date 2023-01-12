1. 本实现是参考

1. https://github.com/mlcommons/inference_results_v2.0/tree/master/closed/NVIDIA

2. https://github.com/mlcommons/inference_results_v2.0/tree/master/closed/NVIDIA/code/ssd-resnet34/tensorrt

   目前 yolov5s可以转换成tensorrt_int8  但是检测没有成功，并且速度还没有float16 快，初步怀疑是好多的layers 不支持

3. 使用本仓库转换成的模型，如果采用其他方式加载，则推理阶段速度会变慢，如果采用本仓库的方法，则速度并不会降低

4. 采用其他仓库转换的模型，采用其他仓库和本仓库方法推理，速度相同