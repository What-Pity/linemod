import cv2
import numpy as np

zone_num=8 # 分区数量
zone_boundary=np.linspace(0,np.pi,zone_num+1) # 区域的边界，包括0和pi
unit_angle=np.pi/zone_num
magnitude_threshold=100 # 梯度幅值的阈值，大于该值认为是边缘
angle_buff=[]

def _compute_angle_buff():
    global zone_num, angle_buff
    for i in range(zone_num):
        _this_angle_buff=[]
        angle_self=2**i
        _this_angle_buff.append(angle_self)
        _right_distance,_left_distance=i,zone_num-i-1
        if _right_distance ==_left_distance:
            for move in range(1,_right_distance+1):
                _this_angle_buff.append(angle_self<<move|angle_self>>move)
        elif _right_distance>_left_distance:
            for move in range(1,_left_distance+1):
                _this_angle_buff.append(angle_self<<move|angle_self>>move)
            for move in range(_left_distance+1,_right_distance+1):
                _this_angle_buff.append(angle_self>>move)
        else:
            for move in range(1,_right_distance+1):
                _this_angle_buff.append(angle_self<<move|angle_self>>move)
            for move in range(_right_distance+1,_left_distance+1):
                _this_angle_buff.append(angle_self<<move)
        angle_buff.append(_this_angle_buff)

_compute_angle_buff()


def _angle_cosine(tempalte_angle,img_angle):
    """计算模板和图像的夹角余弦值"""
    global angle_buff,unit_angle
    tempalte_angle_position=int(np.log2(tempalte_angle))
    angle_position_buff=angle_buff[tempalte_angle_position]
    for i,angle_mask in enumerate(angle_position_buff):
        if (angle_mask&img_angle):
            return np.abs(np.cos(i*unit_angle))

# 预定义lookup table
lookupTable=np.zeros((2**zone_num,zone_num),dtype=float)
for _template_id in range(zone_num):
    _tempalte_angle=2**_template_id
    for _img_angle in range(1,2**zone_num):
        lookupTable[_img_angle,_template_id]=_angle_cosine(_tempalte_angle,_img_angle)

def _compute_zone_id(angle):
    """根据角度计算所属分区"""
    global zone_boundary
    for i in range(zone_num):
        if angle>=zone_boundary[i] and angle<zone_boundary[i+1]:
            return _encode_id(i)
    return _encode_id(i)

def _encode_id(id:int):
    """将id编码为二进制字符串"""
    return 2**id

def extrac_image_feature(image):
    """计算待测图像的梯度方向"""
    global magnitude_threshold
    h,w=image.shape[:2]

    # 提取rgb图像梯度幅值最大通道的梯度
    sobx=cv2.Sobel(image,cv2.CV_64F,1,0,None,3) # x方向梯度
    soby=cv2.Sobel(image,cv2.CV_64F,0,1,None,3) # y方向梯度
    sob_magnitude=np.hypot(sobx,soby) # 梯度大小
    sob_idx=np.argmax(sob_magnitude,axis=2) # 梯度最大通道的索引
    rows, cols = np.indices((sob_idx.shape[0], sob_idx.shape[1]))
    sobelx = sobx[rows, cols, sob_idx]
    sobely=soby[rows,cols,sob_idx]

    # 计算梯度的方向，并量子化到zone_num个方向
    roi=np.hypot(sobelx,sobely)>magnitude_threshold
    ori=np.zeros_like(roi,dtype=np.uint32)
    for _row in range(h):
        for _col in range(w):
            if roi[_row,_col]:
                _x,_y=sobelx[_row,_col],sobely[_row,_col]
                if _y<0: _x,_y=-_x,-_y
                _angle=np.arctan2(_y,_x)
                ori[_row-1:_row+2,_col-1:_col+2]|=_compute_zone_id(_angle)
    return ori

def extrac_template_feature(image):
    """计算模板图像的梯度方向，并返回相应掩码"""
    global magnitude_threshold
    h,w=image.shape[:2]

    # 提取rgb图像梯度幅值最大通道的梯度
    sobx=cv2.Sobel(image,cv2.CV_64F,1,0,None,3) # x方向梯度
    soby=cv2.Sobel(image,cv2.CV_64F,0,1,None,3) # y方向梯度
    sob_magnitude=np.hypot(sobx,soby) # 梯度大小
    sob_idx=np.argmax(sob_magnitude,axis=2) # 梯度最大通道的索引
    rows, cols = np.indices((sob_idx.shape[0], sob_idx.shape[1]))
    sobelx = sobx[rows, cols, sob_idx]
    sobely=soby[rows,cols,sob_idx]

    # 计算梯度的方向，并量子化到zone_num个方向
    roi=np.hypot(sobelx,sobely)>magnitude_threshold
    ori=np.zeros_like(roi,dtype=np.uint32)
    for _row in range(h):
        for _col in range(w):
            if roi[_row,_col]:
                _x,_y=sobelx[_row,_col],sobely[_row,_col]
                if _y<0: _x,_y=-_x,-_y
                _angle=np.arctan2(_y,_x)
                ori[_row,_col]|=_compute_zone_id(_angle)
    return roi,ori

def compute_feature_similarity(template_feature,image_feature,mask):
    """针对模板图像和待测图像计算相似度，要求两图像相同大小，且经过特征提取，返回平均相似度"""
    feature_t=template_feature[mask]
    feature_i=image_feature[mask]
    feature_num=len(feature_t)
    similarity=0
    for ft,fi in zip(feature_t,feature_i):
        looup_col=int(np.log2(ft))
        similarity+=lookupTable[fi,looup_col]
    return similarity/feature_num

