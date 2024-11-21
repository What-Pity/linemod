import numpy as np
from ..utils import lookup
from .global_config import global_config
from ..utils import utils
import logging

__all__ = ["color_config","depth_config"]


color_config = {
    "sobel_threshold":100, # 边缘梯度的检测阈值
    "lookup_table":lookup.get_color_table(),
}

depth_config = {
    "cx": 325.2611,
    "cy": 242.04899,
    "depth_scale": 1.0,
    "fx": 572.4114,
    "fy": 573.57043,
    "depth_threshold":15,
    "base_normal":utils.depth_base_normal(),
    "lookup_table":lookup.get_depth_table(),
}

color_config.update(global_config)
depth_config.update(global_config)

if __name__ == "main":
    pass

