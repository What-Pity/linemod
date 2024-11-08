import cv2
import numpy as np

def best_matching_location(image,template,mask):
    """返回最佳匹配位置的xywh"""
    matching=cv2.matchTemplate(image,template,cv2.TM_SQDIFF,mask=mask)
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(matching)
    return min_loc[0],min_loc[1],template.shape[1],template.shape[0]

def best_matching_orientation(image,template,mask):
    """对于相同大小的image和template，返回最佳匹配角度"""
    h,w=template.shape[:2]
    templates_masks=[(cv2.warpAffine(template,cv2.getRotationMatrix2D((w//2,h//2),i,1.0),(w,h)),cv2.warpAffine(mask,cv2.getRotationMatrix2D((w//2,h//2),i,1.0),(w,h))) for i in range(360)]
    scores=[cv2.matchTemplate(image,t,cv2.TM_CCOEFF_NORMED,mask=m).max() for t,m in templates_masks]

    return np.argmax(scores),np.max(scores)