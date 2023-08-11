import torch
import os
import cv2
import random
import numpy as np
import copy


class Scannetdataset(torch.utils.data.Dataset):
    def __init__(self, config, mode='train'):
        self.mode=mode
        super().__init__()
        self.data_list = []
        self.prompt_paths = []
        self.scene_kfs = dict()
        self.valid_ids = dict()
        scannet_root_dir=config['image_dir']
        scenes=os.listdir(scannet_root_dir)
        if len(scenes)==0:
            raise Exception('No scenes in {}, please check ScanNet is in the correct path'.format(scannet_root_dir))
        else:
            print('Found {} scenes'.format(len(scenes)))
        for scene_id in sorted(scenes):
            scene_dir = os.path.join(scannet_root_dir, scene_id)
            if os.path.exists(os.path.join(scene_dir, 'valid_frames.npy')):
                valid_id=np.load(os.path.join(scene_dir, 'valid_frames.npy'))
            else:
                raise Exception('{} does not contain valid_frames.npy'.format(scene_id))
            
            if config['data_load_mode']=='fix_interval':
                max_id=max(valid_id)
                kf_ids=[i for i in range(0, max_id, config['test_interval']) if i in valid_id]
                if len(kf_ids)>150: # relex this number if using GPU with large memory
                    continue
            else:
                kf_path = os.path.join(scene_dir, 'key_frame_0.6.txt')
                if not os.path.exists(kf_path):
                    raise Exception("{} does not contain key_frame_0.6.txt".format(scene_id))
                with open(kf_path, 'r') as f:
                    kf_ids = f.readlines()
                kf_ids = [int(item.strip()) for item in kf_ids]
           
            self.valid_ids[scene_id] = valid_id

            scene_kfs = []
            if len(kf_ids) < 10:
                continue
            
            for i, kf_id in enumerate(kf_ids):
                kf_rgb_path = os.path.join(
                    scene_dir, 'color/{}.jpg'.format(kf_id))
                if not os.path.exists(kf_rgb_path):
                    continue
                prompt_path = kf_rgb_path.replace(
                    'color', 'prompt').replace('jpg', 'txt')
                if os.path.exists(prompt_path):
                    scene_kfs.append(kf_rgb_path)       
                    self.prompt_paths.append(prompt_path)
                    if mode=='train' or i==0 or (config['data_load_mode']=='fix_frame' and i%config['num_views']==0):
                        self.data_list.append((scene_id, kf_rgb_path))

            self.scene_kfs[scene_id] = scene_kfs
        
        # number of views
        self.num_views = config['num_views']
        self.resolution = (config['resolution_w'], config['resolution_h'])
        self.data_load_mode=config['data_load_mode']
        self.gen_data_ratio=config['gen_data_ratio']

    def __len__(self):
        return len(self.data_list)

    def _get_consecutive_kfs_inp(self, scene_id):
        img_key_paths=self.scene_kfs[scene_id]
        img_paths=[]
        masks=[]
        valid_ids=self.valid_ids[scene_id]
        for i in range(len(img_key_paths)-1):
            idx1=int(img_key_paths[i].split('/')[-1].split('.')[0])
            idx2=int(img_key_paths[i+1].split('/')[-1].split('.')[0])
            interval=20
            j=1
            img_paths.append(img_key_paths[i])
            masks.append(True)
            while j*interval+idx1<idx2:
                idx=idx1+j*interval
                if idx in valid_ids:
                    img_paths.append(os.path.join(
                            '/'.join(img_key_paths[0].split('/')[:-1]), str(idx)+'.jpg'))
                    masks.append(False)
                j+=1
        img_paths.append(img_key_paths[-1])
        masks.append(True)

        return img_paths, np.array(masks)

    def _get_consecutive_kfs_fix_frame(self, scene_id, img_path):
        scene_kfs = self.scene_kfs[scene_id]
        num_views=self.num_views
        num_kfs = len(scene_kfs)
        idx_base = scene_kfs.index(img_path)
        
        if idx_base + num_views < num_kfs:
            idx_list = np.arange(idx_base, idx_base + num_views)
            img_paths = [scene_kfs[item] for item in idx_list]
        else:
            num_back = num_kfs - idx_base - 1
            num_front = num_views - 1 - num_back
            idx_start = idx_base - num_front
            if idx_start < 0:
                idx_start = 0
                idx_list = np.arange(len(scene_kfs))
                
                img_paths=copy.deepcopy(scene_kfs)
                valid_ids=self.valid_ids[scene_id]

                min_id=0
                max_id=valid_ids.max()
                while len(img_paths)<num_views:
                    idx=random.randint(min_id, max_id)
                    if idx in valid_ids:
                        img_paths.append(os.path.join(
                            '/'.join(img_path.split('/')[:-1]), str(idx)+'.jpg'))   
            else:
                idx_list = np.arange(idx_start, idx_start + num_views)
                img_paths = [scene_kfs[item] for item in idx_list]

        return img_paths
    
    def _get_consecutive_kfs_inp(self, scene_id, img_path):
        valid_ids=self.valid_ids[scene_id]
        scene_kfs=self.scene_kfs[scene_id]
        num_kfs = len(scene_kfs)
        idx_base = scene_kfs.index(img_path)
        num_views=2
        if idx_base + num_views >= num_kfs:
            idx_base=num_kfs-num_views-1
        idx_list = [idx_base, idx_base + num_views]
        img_key1=scene_kfs[idx_list[0]]
        img_key2=scene_kfs[idx_list[1]]
        img_paths = [img_key1]

        idx1=int(img_key1.split('/')[-1].split('.')[0])
        idx2=int(img_key2.split('/')[-1].split('.')[0])

        _idx_list=list(range(idx1+1, idx2-1))
        random.shuffle(_idx_list)
        
        for idx in _idx_list:
            if idx in valid_ids:
                img_paths.append(os.path.join(
                    '/'.join(img_path.split('/')[:-1]), str(idx)+'.jpg'))
            if len(img_paths)==self.num_views-1:
                break
        if len(img_paths)!=self.num_views-1:
            img_paths=img_paths+[img_paths[-1]]*(self.num_views-len(img_paths)-1)
        img_paths.append(img_key2)
        mask=np.zeros(len(img_paths)).astype(np.bool)
        mask[0]=True
        mask[-1]=True

        return img_paths, mask

    def load_seq(self, image_paths, resolution):
        images_ori = [cv2.cvtColor(cv2.imread(
            path)[..., :3], cv2.COLOR_BGR2RGB) for path in image_paths]

        # load pose
        poses = []
        prompts = []
        for p in image_paths:
            pose_path = p.replace('color', 'pose').replace('jpg', 'txt')
            pose = np.loadtxt(pose_path)
            poses.append(pose)
            if np.isnan(np.linalg.inv(pose)).any():
                return self.__getitem__(torch.randint(0, len(self.data_list), (1, )).item())
            prompt_path = p.replace('color', 'prompt').replace('jpg', 'txt')
            with open(prompt_path, 'r') as f:
                prompt = f.readlines()[0].strip()

            prompts.append(prompt)

        poses = np.stack(poses, axis=0)  # [num_views, 4, 4]

        # load k
        k_path = os.path.join(
            '/'.join(image_paths[0].replace('color', 'intrinsic').split('/')[:-1]), 'intrinsic_depth.txt')
        k = np.loadtxt(k_path)

        images = np.stack([cv2.resize(x, resolution)
                           for x in images_ori])/127.5 - 1

        # load depth
        depth_path = [p.replace('color', 'depth').replace(
            'jpg', 'png') for p in image_paths]
        depths = [cv2.imread(p, cv2.IMREAD_ANYDEPTH)/1000 for p in depth_path]

        h_ori, w_ori = depths[0].shape
        scale = h_ori / resolution[1]

        k[:2] /= scale

        depths = [cv2.resize(x, resolution,
                             interpolation=cv2.INTER_NEAREST) for x in depths]
        depths = np.stack(depths, axis=0)

        depth_valid_mask = depths > 0
        depth_inv = 1. / (depths + 1e-6)
        depth_max = [depth_inv[i][depth_valid_mask[i]].max()
                     for i in range(depth_inv.shape[0])]
        depth_min = [depth_inv[i][depth_valid_mask[i]].min()
                     for i in range(depth_inv.shape[0])]
        depth_max = np.stack(depth_max, axis=0)[:, None, None]
        depth_min = np.stack(depth_min, axis=0)[
            :, None, None]  # [num_views, 1, 1]
        depth_inv_norm_full = (depth_inv - depth_min) / \
            (depth_max - depth_min + 1e-6) * 2 - 1  # [-1, 1]
        depth_inv_norm_full[~depth_valid_mask] = -2
        depth_inv_norm_full = depth_inv_norm_full.astype(np.float32)
        return images, depths, depth_inv_norm_full, poses, k, prompts

    def __getitem__(self, idx):
        scene_id, img_path = self.data_list[idx]
        
        if self.data_load_mode=='fix_interval':
            image_paths=self.scene_kfs[scene_id]
            mask=np.ones(len(image_paths)).astype(np.bool)
        elif self.data_load_mode=='fix_frame':
            p_rand=random.random()
            if self.mode=='train' and p_rand>self.gen_data_ratio:
                image_paths, mask=self._get_consecutive_kfs_inp(
                    scene_id, img_path)
            else:
                image_paths= self._get_consecutive_kfs_fix_frame(
                    scene_id, img_path)
                mask=np.ones(len(image_paths)).astype(np.bool)
        elif self.data_load_mode=='two_stage':
            image_paths, mask=self._get_consecutive_kfs_inp(
                scene_id, img_path)

        images, depths, depth_inv_norm, poses, K, prompts = self.load_seq(
            image_paths, self.resolution)
        
        
        depth_inv_norm_small= np.stack([cv2.resize(depth_inv_norm[i], (
            self.resolution[0]//8, self.resolution[1]//8), interpolation=cv2.INTER_NEAREST) for i in range(depth_inv_norm.shape[0])])

        images = images.astype(np.float32)
        depths = depths.astype(np.float32)
        poses = poses.astype(np.float32)
        K = K.astype(np.float32)
        
        return {
            'image_paths': image_paths,
            'mask': mask,
            'images': images,
            'depths': depths,
            'poses': poses,
            'K': K,
            'prompt': prompts,
            'depth_inv_norm': depth_inv_norm,
            'depth_inv_norm_small': depth_inv_norm_small,
            'data_load_mode': self.data_load_mode,
        }