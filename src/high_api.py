import packages.linemod as lm
import packages.utils as utils
import numpy as np

__all__=[\
    "rgb_make_template",\
    "rgb_extract_image_feature",\
    "depth_make_template",\
    "depth_extract_image_feature",\
    "get_response_map",\
    "find_high_similar",
         ]

def rgb_make_template(template_img):
    feature_id,_=lm.color.extract_feature_id(template_img)
    return lm.match.make_one_template(feature_id,lm.color_config)

def rgb_extract_image_feature(image):
    image_feature_id,roi_mask=lm.color.extract_feature_id(image)
    image_feature=np.zeros_like(image_feature_id,dtype=np.uint8)
    image_feature[roi_mask]=2**(image_feature_id[roi_mask]-1)
    return image_feature

def depth_make_template(template_depth):
    pointcloud=utils.compute_pointcloud(template_depth,lm.depth_config)
    normals,invalid=utils.compute_normals(pointcloud,lm.depth_config)
    feature_id=lm.depth.extract_feature_id(normals,invalid)
    return lm.match.make_one_template(feature_id,lm.depth_config)

def depth_extract_image_feature(depth):
    pointcloud=utils.compute_pointcloud(depth,lm.depth_config)
    normals,invalid=utils.compute_normals(pointcloud,lm.depth_config)
    roi_mask=~invalid
    depth_feature_id=lm.depth.extract_feature_id(normals,invalid)
    depth_feature=np.zeros_like(depth_feature_id,dtype=np.uint8)
    depth_feature[roi_mask]=2**(depth_feature_id[roi_mask]-1)
    return depth_feature

def get_response_map(image_feature,cfg):
    image_feature_spread=utils.spread(image_feature,cfg)
    return lm.match.compute_response_map(image_feature_spread,cfg)

def _compute_similarity(templates,response):
    similarity=np.empty((len(templates),response.shape[1]))
    for i,(template,offset,response_index) in enumerate(templates):
        similarity[i,:]=lm.match.compute_feature_similarity(response,template,offset,response_index)
    return similarity

def find_high_similar(rgb_templates,rgb_response,depth_templates,depth_response,thre=0.9):
    # rgb
    rgb_similarity=_compute_similarity(rgb_templates,rgb_response)
    # depth
    depth_similarity=_compute_similarity(depth_templates,depth_response)

    similarity=rgb_similarity*depth_similarity
    (xs,ys),template_id=lm.match.decode_match_location(similarity,thre)
    return xs,ys,template_id