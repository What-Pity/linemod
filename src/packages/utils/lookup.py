import numpy as np
from ..linemod.global_config import global_config

__all__=[]

feature_number=global_config["FeatureNumber"]


# 计算预定义的角度位置掩码列表angle_position_mask，以二进制存储，列表中的每个元素为，对应编码的量化角度位置掩码及其左右移后的位置掩码组成的列表，用于方便地查找量化特征和传播特征的距离，以便于快速计算lookup table
# 例如：
# 对于3号量化特征001000（实际以int类型存储），其位置掩码及左右移位置掩码为
# 0：001000（左右移0位）
# 1：010100（左右移1位）
# 2：100010（左右移2位）
# 3：000001（左右移3位）
# 那么angle_position_mask[3]=[001000,010100,100010,000001]
# 注意，angle_position_mask的大小为feature_number，其中每个元素的大小不同，由位置掩码本身决定
angle_position_masks_list=[] # 掩码列表的列表，大小为feature_number
for i in range(feature_number):
    ith_angle_mask=[] # 第i个量化特征的位置掩码列表
    angle_self=2**i
    ith_angle_mask.append(angle_self) # 第一个元素为量化特征位置掩码本身
    _right_distance,_left_distance=i,feature_number-i-1 # 计算量化特征位置左右移的最大距离，例如对于位置特征001000，左右移最大距离为2，3
    if _right_distance ==_left_distance:
        for move in range(1,_right_distance+1):
            ith_angle_mask.append(angle_self<<move|angle_self>>move)
    elif _right_distance>_left_distance:
        for move in range(1,_left_distance+1):
            ith_angle_mask.append(angle_self<<move|angle_self>>move)
        for move in range(_left_distance+1,_right_distance+1):
            ith_angle_mask.append(angle_self>>move)
    else:
        for move in range(1,_right_distance+1):
            ith_angle_mask.append(angle_self<<move|angle_self>>move)
        for move in range(_right_distance+1,_left_distance+1):
            ith_angle_mask.append(angle_self<<move)
    angle_position_masks_list.append(ith_angle_mask)

def compute_feature_distance(quantize_feature_masks,spread_feature):
    """计算量化特征到传播特征的最小距离"""
    # 遍历量化特征掩码列表，找出量化特征到传播特征之间的最短距离，并计算相似度
    for i,angle_mask in enumerate(quantize_feature_masks):
        if (angle_mask & spread_feature): # 如果量化特征位置掩码和传播特征有重叠，则两者的与运算为True，可知道该量化特征到传播特征的距离为i
            return i

def color_table_method(distance):
    """根据特征距离计算rgb图像相似度的方法"""
    global feature_number
    unit_angle=np.pi/feature_number # 每一相邻分区的夹角
    return np.abs(np.cos(distance*unit_angle))

def get_lookup_table(cosine_method):
    """ 计算rgb图像的lookup table，大小为 (2**feature_number) x feature_number
    列序号表示某一个量化特征，行序号表示传播后的量化特征，其中的每个元素表示量化特侦和传播特征的最大相似度
    例如：
    lookupTable[m,n]表示第m个传播特征和第n个量化特征之间的最大相似度，即 m 与 2**n 编码的余弦最大值
    若m = 0110 (6), n = 3，2**n = 1000 (8)，则 lookupTable[m,n] = max(cos(1*unit_angle),cos(2*unit_angle)))=cos(1*unit_angle)，unit_angle的系数1和2是1000到0110的距离1、2
    """
    global feature_number, angle_position_masks_list

    lookupTable=np.zeros((2**feature_number,feature_number),dtype=float)
    for quantize_feature_id in range(feature_number): # 量化特征的序号
        angle_position_masks=angle_position_masks_list[quantize_feature_id] # 取出对应的量化特征掩码列表
        for spread_feature_id in range(1,2**feature_number): # 传播的特征序号，也是传播特征本身
            distance=compute_feature_distance(angle_position_masks,spread_feature_id) # 计算量化特征到传播特征的最短距离
            lookupTable[spread_feature_id,quantize_feature_id] = cosine_method(distance) # 计算量化特征到传播特征的最大相似度
    return lookupTable

def get_color_table():
    return get_lookup_table(color_table_method)

def depth_table_method(distance):
    """计算法向量相似度的方法"""
    global feature_number
    dist_angle=np.pi*2/feature_number*distance # xy平面的投影上法向量的角度差
    _SQRT2DIV2=np.sqrt(2)/2 # 二分之根号二，避免重复计算
    _SQRT3DIV2=np.sqrt(3)/2
    _1DIV2=0.5

    normal1=np.array([_1DIV2,0,_SQRT3DIV2])
    normal2=np.array([\
        _1DIV2*np.cos(dist_angle),\
        _1DIV2*np.cos(dist_angle),\
        _SQRT3DIV2])
    return (normal1*normal2).sum()

def get_depth_table():
    """计算法向量的lookup table"""
    return get_lookup_table(depth_table_method)