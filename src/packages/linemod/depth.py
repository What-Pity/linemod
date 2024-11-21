import cv2
import numpy as np
from . import config

cfg=config.depth_config
__all__=[]

def extract_feature_id(normals,invalid):
    """提取图像的法向量特征序号"""
    global cfg
    dot_prod_map=np.zeros((normals.shape[0],normals.shape[1],cfg["FeatureNumber"]),dtype=np.float16)
    for i,base_normal in enumerate(cfg["base_normal"]):
        dot_prod_map[:,:,i]=(normals*base_normal).sum(axis=2)
    dot_prod_map=np.abs(dot_prod_map)
    feature_id=np.argmax(dot_prod_map,axis=2)+1
    feature_id[invalid]=0
    return feature_id
