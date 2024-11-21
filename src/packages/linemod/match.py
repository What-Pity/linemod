import numpy as np
import cv2
from scipy import stats
from . import global_config

__all__=["make_one_template","compute_response_map","compute_feature_similarity","decode_match_location"]

def make_one_template(template_feature_id,cfg):
    """根据模板的特征，空间采样，生成稀疏的特征，以及特征对应位置（适合linemod匹配）"""
    image_width=cfg["ImageWidth"]
    spread_neibour=cfg["SpreadNeibour"]
    # 每sample_rate x sample_rate的像素块作为一个采样分区，求解分区的众数，并赋值到中点，其他位置置零，求解的result即为空间采样后的template_feature_id
    h,w=template_feature_id.shape[:2]
    sample_rate=cfg["TemplateSampleRate"]
    result=np.zeros((h,w),dtype=int)
    for i in range(h//sample_rate):
        for j in range(w//sample_rate):
            region=template_feature_id[i*sample_rate:(i+1)*sample_rate,j*sample_rate:(j+1)*sample_rate]
            none_zero=(region!=0)
            if none_zero.any(): # 如果该分区有值
                mode=stats.mode(region[none_zero])[0] # 众数
                result[i*sample_rate+sample_rate//2,j*sample_rate+sample_rate//2]=mode
    mask=result!=0
    X,Y=np.meshgrid(range(w),range(h))
    offset=(Y[mask]/spread_neibour).astype(int)*int(image_width/spread_neibour)+(X[mask]/spread_neibour).astype(int) # 计算偏移量
    index=Y[mask]%spread_neibour*spread_neibour+X[mask]%spread_neibour # 计算对应的索引
    return result[mask],offset,index

def compute_response_map(image_feature,cfg):
    """计算相似度响应图（优化存储后的格式），其shape=(spread_neibour**2,-1,zone_num)"""
    lookup_table,image_width,image_height,spread_neibour,feature_number = \
        cfg["lookup_table"], cfg["ImageWidth"],cfg["ImageHeight"],cfg["SpreadNeibour"],cfg["FeatureNumber"]
    memory_length=int(image_height*image_width/spread_neibour/spread_neibour)
    response_map=np.zeros((spread_neibour*spread_neibour,memory_length,feature_number),dtype=float)
    for i in range(feature_number):
        _response=lookup_table[image_feature,i]
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
        similarity[i,:feature_length-offset]=response_maps[index,offset:,template_feature-1]
    return np.mean(similarity,axis=0)

def decode_match_location(similaritys,threshold=0.8):
    """根据计算得到的相似度，返回相似度超过阈值的位置（左上角坐标）和相应的角度序列号"""
    global global_config
    image_width,spread_neibour=global_config.global_config["ImageWidth"],global_config.global_config["SpreadNeibour"]
    _neibour_per_row=image_width//spread_neibour
    if similaritys.ndim==1:
        _location_idx=np.where(similaritys>threshold)[0]
        if _location_idx.size==0: return None
        return _location_idx%_neibour_per_row*spread_neibour,_location_idx//_neibour_per_row*spread_neibour
    elif similaritys.ndim==2:
        _angle_idx,_location_idx=np.where(similaritys>threshold)
        if _location_idx.size==0: return None
        return (_location_idx%_neibour_per_row*spread_neibour,_location_idx//_neibour_per_row*spread_neibour),_angle_idx
    else: raise ValueError("similaritys.ndim must be 1 or 2")