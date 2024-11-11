import cv2
import numpy as np
from scipy import stats

zone_num:int=8 # 分区数量，不能大于8
zone_boundary_tangent=np.tan(np.linspace(-np.pi/2,np.pi/2,zone_num+1)) # 区域的边界的正切值，包括0和pi
unit_angle=np.pi/zone_num
magnitude_threshold:int=100 # 梯度幅值的阈值，大于该值认为是边缘
angle_buff=[]
IMAGE_WIDTH:int=640
IMAGE_HEIGHT:int=480
spread_neibour:int=8

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

def _compute_zone_id(x,y,binarize):
    """根据角度计算所属分区"""
    global zone_boundary_tangent, zone_num
    zero_idx=x==0
    with np.errstate(divide='ignore'):
        tangent=np.abs(y/x)
    tangent_idx=np.zeros_like(tangent,dtype=bool)
    zone_id=np.zeros_like(x,dtype=int)
    zone_id[zero_idx]=0
    for i in range(1,zone_num+1):
        tangent_idx=tangent_idx^(zone_boundary_tangent[i]<=tangent)
        zone_id[tangent_idx]=i-1
    if binarize:
        return 2**zone_id
    else:
        return zone_id

def _spread(gradient,neibour):
    """传播梯度方向到邻域，邻域大小为neibour"""
    boundary=neibour//2
    neibour=boundary*2
    h,w=gradient.shape[:2]
    gradient_expand=cv2.copyMakeBorder(
        gradient,boundary,boundary,boundary,boundary,cv2.BORDER_CONSTANT,value=0)
    for i in range(neibour):
        for j in range(neibour):
            gradient=gradient|gradient_expand[i:i+h,j:j+w]
    return gradient

def extract_image_feature(image):
    """计算待测图像的梯度方向"""
    global spread_neibour,IMAGE_HEIGHT,IMAGE_WIDTH
    h,w=image.shape[:2]
    if h!=IMAGE_HEIGHT or w!=IMAGE_WIDTH: # 如果图像大小与预设大小不同，则报错
        raise ValueError("image size is not match")
    gradient=_extract_general_feature(image,binarize=True)
    return _spread(gradient,spread_neibour)

def make_one_template(template_feature,maintain_shape=False):
    """根据模板的特征，空间采样，生成稀疏的特征，以及特征对应位置（适合linemod匹配）"""
    h,w=template_feature.shape[:2]
    sample_rate=5 # sample_rate x sample_rate作为一个选区，求解众数，并赋值到中点，其他位置置零
    result=np.zeros((h,w),dtype=int)
    for i in range(h//sample_rate):
        for j in range(w//sample_rate):
            region=template_feature[i*sample_rate:(i+1)*sample_rate,j*sample_rate:(j+1)*sample_rate]
            none_zero=(region!=0)
            if none_zero.any(): # 如果该选区有值
                mode=stats.mode(region[none_zero])[0] # 众数
                result[i*sample_rate+sample_rate//2,j*sample_rate+sample_rate//2]=mode
    if maintain_shape:
        return result
    else: # 如果不想保持图片形状，则抽象为特征（方向的二进制表示）、坐标（偏移值，用于线性化内存）和索引
        global IMAGE_WIDTH,spread_neibour
        mask=result!=0
        X,Y=np.meshgrid(range(w),range(h))
        offset=(Y[mask]/spread_neibour).astype(int)*int(IMAGE_WIDTH/spread_neibour)+(X[mask]/spread_neibour).astype(int)
        index=Y[mask]%spread_neibour*spread_neibour+X[mask]%spread_neibour
        return result[mask],offset,index

def extract_template_feature(template):
    """计算模板图像的梯度方向，并返回相应掩码"""
    return _extract_general_feature(template,binarize=False)
def _extract_general_feature(image, binarize):
    """通用的特征提取函数，仅提取图像的原生梯度方向"""
    global magnitude_threshold
    h,w=image.shape[:2]

    # 提取rgb图像梯度幅值最大通道的梯度
    sobx=cv2.Sobel(image,cv2.CV_64F,1,0,None,3) # x方向梯度
    soby=cv2.Sobel(image,cv2.CV_64F,0,1,None,3) # y方向梯度
    # sob_magnitude=np.hypot(sobx,soby) # 梯度大小
    sob_magnitude=np.abs(sobx)+np.abs(soby) # 优化计算
    sob_idx=np.argmax(sob_magnitude,axis=2) # 梯度最大通道的索引
    rows, cols = np.indices((sob_idx.shape[0], sob_idx.shape[1]))
    sobelx = sobx[rows, cols, sob_idx]
    sobely=soby[rows,cols,sob_idx]

    # 计算梯度的方向，并量子化到zone_num个方向
    # roi=np.hypot(sobelx,sobely)>magnitude_threshold
    roi=(np.abs(sobelx)+np.abs(sobely))>magnitude_threshold # 优化计算
    ori=np.zeros_like(roi,dtype=np.uint8)
    ori[roi]=_compute_zone_id(sobelx[roi],sobely[roi],binarize)
    return ori

def compute_response_map(image_feature):
    """计算相似度响应图（优化存储后的格式），其shape=(spread_neibour**2,-1,zone_num)"""
    global lookupTable, IMAGE_WIDTH, IMAGE_HEIGHT,spread_neibour,zone_num
    memory_length=int(IMAGE_HEIGHT*IMAGE_WIDTH/spread_neibour/spread_neibour)
    response_map=np.zeros((spread_neibour*spread_neibour,memory_length,zone_num),dtype=float)
    for i in range(zone_num):
        _response=lookupTable[image_feature,i]
        for j in range(spread_neibour**2):
            response_map[j,:,i]=(_response[\
                j//spread_neibour::spread_neibour,\
                j%spread_neibour::spread_neibour]).flatten()
    return response_map

def compute_feature_similarity(response_maps,template_features,offsets,map_indexs):
    """针对模板图像和待测图像计算相似度，返回平均相似度"""
    feature_num=len(offsets)
    feature_length=response_maps.shape[1]
    similarity=np.zeros((feature_num,feature_length),dtype=float)
    for i,(template_feature,offset,index) in enumerate(zip(template_features,offsets,map_indexs)):
        similarity[i,:feature_length-offset]=response_maps[index,offset:,template_feature]
    return np.mean(similarity,axis=0)

def decode_match_location(similaritys,threshold=0.8):
    """根据计算得到的相似度，返回相似度超过阈值的位置（左上角坐标）"""
    global IMAGE_WIDTH,spread_neibour
    _neibour_per_row=IMAGE_WIDTH//spread_neibour
    if similaritys.ndim==1:
        _idx=np.where(similaritys>threshold)
        return _idx[0]%_neibour_per_row*spread_neibour,_idx[0]//_neibour_per_row*spread_neibour
    elif similaritys.ndim==2:
        _,_idx=np.where(similaritys>threshold)
        return _idx[0]%_neibour_per_row*spread_neibour,_idx[0]//_neibour_per_row*spread_neibour
    else: raise ValueError("similaritys.ndim must be 1 or 2")

if __name__ == '__main__':
    print(angle_buff)

