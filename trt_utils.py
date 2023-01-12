import os
import numpy as np
from typing import Dict
import re

def get_dyn_ranges(cache_file: str) -> Dict[str, np.uint32]:
    """
    Get dynamic ranges from calibration file for network tensors.

    Args:
        cache_file (str):
            Path to INT8 calibration cache file.

    Returns:
        Dict[str, np.uint32]: Dictionary of tensor name -> dynamic range of tensor
    """
    dyn_ranges = {}
    if not os.path.exists(cache_file):
        raise FileNotFoundError("{} calibration file is not found.".format(cache_file))

    with open(cache_file, "rb") as f:
        lines = f.read().decode('ascii').splitlines()
    for line in lines:
        regex = r"(.+): (\w+)"
        results = re.findall(regex, line)
        # Omit unmatched lines
        if len(results) == 0 or len(results[0]) != 2:
            continue
        results = results[0]
        tensor_name = results[0]
        # Map dynamic range from [0.0 - 1.0] to [0.0 - 127.0]
        dynamic_range = np.uint32(int(results[1], base=16)).view(np.dtype('float32')).item() * 127.0
        dyn_ranges[tensor_name] = dynamic_range
    return dyn_ranges
