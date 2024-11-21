import cv2
import numpy as np
from . import config

cfg=config.color_config
__all__=[]

def _compute_zone_id(x,y):
    """根据角度计算所属分区的序号：1,...,feature_number
    注意：并不返回分区编码"""
    global cfg
    feature_number=cfg["FeatureNumber"]
    zone_boundary_tangent=np.tan(np.linspace(-np.pi/2,np.pi/2,feature_number+1)) # 区域的边界的正切值，包括0和pi
    with np.errstate(divide='ignore'):
        tangent=y/x # 计算正切值，x为0时，tan值为np.inf
    # 根据tangent量化梯度方向
    tangent_idx=np.zeros_like(tangent,dtype=bool)
    zone_id=np.zeros_like(x,dtype=int)
    for i in range(1,feature_number+1):
        # 逐个区间判断，如果tangent[m,n]属于区间i-1，则tangent_idx[m,n]=True，否则为False
        tangent_idx=(tangent>zone_boundary_tangent[i-1])&(tangent<=zone_boundary_tangent[i])
        zone_id[tangent_idx]=i # 区间序号赋值，编号为1,...,feature_number
    zone_id[tangent>zone_boundary_tangent[-1]]=feature_number # 添加上剩余的元素
    return zone_id
    
def extract_feature_id(image):
    """通用的特征提取函数，仅提取图像的原生梯度方向"""
    global cfg

    # 提取rgb图像梯度幅值最大通道的梯度
    sobx=cv2.Sobel(image,cv2.CV_64F,1,0,None,3) # x方向梯度
    soby=cv2.Sobel(image,cv2.CV_64F,0,1,None,3) # y方向梯度
    sob_magnitude=np.abs(sobx)+np.abs(soby) # 优化计算sobel阈值
    sob_idx=np.argmax(sob_magnitude,axis=2) # 梯度最大通道的索引
    rows, cols = np.indices((sob_idx.shape[0], sob_idx.shape[1]))
    sobelx = sobx[rows, cols, sob_idx]
    sobely=soby[rows,cols,sob_idx]

    # 计算梯度的方向，并量子化到feature_number个方向，返回梯度分区的序号
    roi=(np.abs(sobelx)+np.abs(sobely))>cfg["sobel_threshold"] # 优化计算
    orient_id=np.zeros_like(roi,dtype=np.uint8)
    orient_id[roi]=_compute_zone_id(sobelx[roi],sobely[roi])
    return orient_id,roi

