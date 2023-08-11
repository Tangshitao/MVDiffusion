import os
import sys
import cv2
import numpy as np
import lib.Perspec2Equirec as P2E
class Perspective:
    def __init__(self, img_array , F_T_P_array ):
        
        assert len(img_array)==len(F_T_P_array)
        
        self.img_array = img_array
        self.F_T_P_array = F_T_P_array
    
    def GetEquirec(self,height,width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        merge_image = np.zeros((height,width,3))
        merge_mask = np.zeros((height,width,3))
        for img_dir,[F,T,P] in zip (self.img_array,self.F_T_P_array):
            per = P2E.Perspective(img_dir,F,T,P)        # Load equirectangular image
            img , mask = per.GetEquirec(height,width)   # Specify parameters(FOV, theta, phi, height, width)
            mask = mask.astype(np.float32)
            img = img.astype(np.float32)
            weight_mask = np.zeros((img_dir.shape[0],img_dir.shape[1], 3))
            w = img_dir.shape[1]
            weight_mask[:,0:w//2,:] = np.linspace(0,1,w//2)[...,None]
            weight_mask[:,w//2:,:] = np.linspace(1,0,w//2)[...,None]
            weight_mask = P2E.Perspective(weight_mask,F,T,P)
            weight_mask, _ = weight_mask.GetEquirec(height,width)
            blur = cv2.blur(mask,(5,5))
            blur = blur * mask
            mask = (blur == 1) * blur + (blur != 1) * blur * 0.05
            merge_image += img * weight_mask
            merge_mask += weight_mask
        merge_image[merge_mask==0] = 255.
        merge_mask = np.where(merge_mask==0,1,merge_mask)
        merge_image = (np.divide(merge_image,merge_mask))
        
        return merge_image
        
        
