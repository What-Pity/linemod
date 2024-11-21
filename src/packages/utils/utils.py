import numpy as np
import cv2
from ..linemod.global_config import global_config


__all__=["binary_search_position","spread","compute_pointcloud","compute_normals"]

def binary_search_position(n_list, n):
    left, right = 0, len(n_list) - 1
    while left <= right:
        mid = (left + right) // 2
        if n_list[mid] < n:
            left = mid + 1
        else:
            right = mid - 1
    return left

def spread(gradient,cfg):
    """传播梯度方向到邻域，邻域大小为neibour"""
    neibour=cfg["SpreadNeibour"]
    boundary=neibour//2
    neibour=boundary*2
    h,w=gradient.shape[:2]
    gradient_expand=cv2.copyMakeBorder(
        gradient,boundary,boundary,boundary,boundary,cv2.BORDER_CONSTANT,value=0)
    for i in range(neibour):
        for j in range(neibour):
            gradient=gradient|gradient_expand[i:i+h,j:j+w]
    return gradient

def depth_base_normal():
    """生成量化的feature_number个基法向量，与z轴夹角45度，指向z轴正方向，shape=(feature_number,3)"""
    global global_config
    feature_number=global_config["FeatureNumber"]

    _SQRT2DIV2=np.sqrt(2)/2 # 二分之根号二，避免重复计算
    _SQRT3DIV2=np.sqrt(3)/2
    _1DIV2=0.5
    z_coef=_1DIV2
    xy_coef=_SQRT3DIV2

    _base_normal_circle_angle=np.linspace(0,np.pi*2,feature_number+1)[:-1] # 基向量单位圆的夹角，从0到2*pi，包括0，不包括2*pi，划分为feature_number个区间
    base_normal=np.zeros((feature_number,3),dtype=np.float64)
    base_normal[:,2]=z_coef # z
    base_normal[:,0]=xy_coef*np.cos(_base_normal_circle_angle) # x
    base_normal[:,1]=xy_coef*np.sin(_base_normal_circle_angle) # y
    return base_normal

def compute_pointcloud(depth,cfg):
    """对给定深度图计算法向量"""
    # 根据内参计算点云位置
    u,v=np.meshgrid(range(depth.shape[1]),range(depth.shape[0]))
    pointcloud=np.empty((depth.shape[0],depth.shape[1],3),dtype=np.float64)
    pointcloud[:,:,0]=(u-cfg["cx"])*depth/cfg["fx"]
    pointcloud[:,:,1]=(v-cfg["cy"])*depth/cfg["fy"]
    pointcloud[:,:,2]=depth
    return pointcloud

def compute_normals(pointcloud,cfg):
    distance_thre=cfg["depth_threshold"] # 距离阈值，相邻像素距离差超过distance_thre的位置会被标记为无效位置，不参与最终的法向量计算
    # 计算相邻点云梯度
    pointcloud_extend=cv2.copyMakeBorder(pointcloud,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    dx=pointcloud_extend[1:-1,1:-1]-pointcloud_extend[2:,1:-1]
    dy=pointcloud_extend[1:-1,1:-1]-pointcloud_extend[1:-1,2:]
    # 计算法向量
    normals=np.cross(dx,dy) # u、v方向的梯度叉乘为法向量
    n_norm=np.linalg.norm(normals,axis=-1,keepdims=True) # 法向量的模
    invalid_position=(np.abs(dx).sum(axis=-1)>distance_thre)|(np.abs(dy).sum(axis=-1)>distance_thre)|(pointcloud[:,:,2]==0) # 法向量大于阈值或深度图无效（为零）的区域无效
    with np.errstate(invalid='ignore'):
        normals=normals/n_norm
    # 将无效区域的法向量置0
    normals[invalid_position,:]=0
    normals=np.where(np.isnan(normals),0,normals)
    return normals, invalid_position

