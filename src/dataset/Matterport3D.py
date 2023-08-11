
import torch
import os
import numpy as np
import cv2
import random
from .utils import get_K_R
import copy

def warp_img(fov, theta, phi, images, vx, vy):
    img_combine = np.zeros(images[0].shape).astype(np.uint8)

    min_theta = 10000
    for i, img in enumerate(images):
        _theta = vx[i]-theta
        _phi = vy[i]-phi

        if i == 2 and theta > 270:
            _theta = max(360-theta, _theta)
        if _phi == 0 and np.absolute(_theta) > 90:
            continue

        if i > 0 and i < 5 and np.absolute(_theta) < min_theta:
            min_theta = _theta
            min_idx = i

        im_h, im_w, _ = img.shape
        K, R = get_K_R(fov, _theta, _phi, im_h, im_w)
        homo_matrix = K@R@np.linalg.inv(K)
        img_warp1 = cv2.warpPerspective(img, homo_matrix, (im_w, im_h))
        if i == 0:
            img_warp1[im_h//2:] = 0
        elif i == 5:
            img_warp1[:im_h//2] = 0

        img_combine += img_warp1  # *255).astype(np.uint8)
    return img_combine


class MP3Ddataset(torch.utils.data.Dataset):
    def __init__(self, config, mode='train'):
        random.seed(config['seed'])
        self.mode = mode
        self.image_root_dir = config['image_root_dir']

        if mode=='train':
            self.data = np.load(os.path.join(self.image_root_dir, 'train.npy'))
        else:
            self.data = np.load(os.path.join(self.image_root_dir, 'test.npy'))

        self.vx = [-90, 270, 0, 90, 180, -90]
        self.vy = [90, 0, 0, 0, 0, -90]
        self.fov = config['fov']
        self.rot = config['rot']
        self.resolution = config['resolution']
        self.crop_size= config['crop_size']

    def __len__(self):
        return len(self.data)

    def load_prompt(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        return prompt

    def crop_img(self, img, K):
        margin = (self.resolution-self.crop_size)//2
        img_crop = img[margin:-margin, margin:-margin]
        K=copy.deepcopy(K)
        K[0, 2] -= margin
        K[1, 2] -= margin

        return img_crop, K

    def __getitem__(self, idx):
        images_raw = [cv2.imread(os.path.join(self.image_root_dir,path)) for path in self.data[idx]]
        images_raw = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                      for img in images_raw]

        imgs = []
        Rs = []
        num_views=8
        
        if self.mode == 'train':
            init_degree = random.randint(0, 359)
        else:
            init_degree = 0
        
        for i in range(num_views):
            _degree = (init_degree+self.rot*i) % 360
            img = warp_img(
                90, _degree, 0, images_raw, self.vx, self.vy)
            img = cv2.resize(img, (self.resolution, self.resolution))
            
            

            K, R = get_K_R(90, _degree, 0,
                           self.resolution, self.resolution)
            if self.crop_size!=self.resolution:
                img, K = self.crop_img(img, K)
            Rs.append(R)
            imgs.append(img)

        images = (np.stack(imgs).astype(np.float32)/127.5)-1

        K = np.stack([K]*len(Rs)).astype(np.float32)
        R = np.stack(Rs).astype(np.float32)
      
        prompt_dir = os.path.dirname(self.data[idx][0].replace(
            'matterport_skybox_images', 'blip3'))

        prompt = []
        image_name = self.data[idx][0].split('/')[-1].split('_')[0]
        for i in range(num_views):
            _degree = (init_degree+i*self.rot) % 360
            _degree = int(np.round(_degree/45)*45) % 360
            txt_path = os.path.join('{}_{}.txt'.format(
                image_name, _degree))

            prompt_path = os.path.join(self.image_root_dir, prompt_dir, txt_path)
            prompt.append('This is one view of a scene. '+
                          self.load_prompt(prompt_path))

        return {
            'image_paths': self.data[idx][0],
            'images': images,
            'prompt': prompt,
            'R': R,
            'K': K
        }
        