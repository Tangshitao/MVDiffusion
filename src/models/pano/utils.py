import torch
import torch.nn.functional as F
from ..modules.utils import get_x_2d
from einops import rearrange


def get_correspondences(R, K, img_h, img_w):
    m=R.shape[1]
    
    correspondences=torch.zeros((R.shape[0], m, m, img_h, img_w, 2), device=R.device)
    for i in range(m):  
        for j in range(m):

            R_right = R[:, j:j+1]
            K_right = K[:, j:j+1]
            l = R_right.shape[1]

            R_left = R[:, i:i+1].repeat(1, l, 1, 1)
            K_left = K[:, i:i+1].repeat(1, l, 1, 1)

            R_left = R_left.reshape(-1, 3, 3)
            R_right = R_right.reshape(-1, 3, 3)
            K_left = K_left.reshape(-1, 3, 3)
            K_right = K_right.reshape(-1, 3, 3)

            homo_l = (K_right@torch.inverse(R_right) @
                    R_left@torch.inverse(K_left))
            

            xyz_l = torch.tensor(get_x_2d(img_h, img_w),
                                device=R.device)
            xyz_l = (
                xyz_l.reshape(-1, 3).T)[None].repeat(homo_l.shape[0], 1, 1)
            
            xyz_l = homo_l@xyz_l 
            

            xy_l = (xyz_l[:, :2]/xyz_l[:, 2:]).permute(0,
                                                    2, 1).reshape(-1, l, img_h, img_w, 2)
            
            correspondences[:,i,j]=xy_l[:,0]
    return correspondences


def get_key_value(key_value, xy_l, homo_r, ori_h, ori_w, ori_h_r, query_h):
    
    b, c, h, w = key_value.shape
    query_scale = ori_h//query_h
    key_scale = ori_h_r//h

    xy_l = xy_l[:, query_scale//2::query_scale,
                query_scale//2::query_scale]/key_scale-0.5

    key_values = []

    xy_proj = []
    kernal_size=3
    for i in range(0-kernal_size//2, 1+kernal_size//2):
        for j in range(0-kernal_size//2, 1+kernal_size//2):
            xy_l_norm = xy_l.clone()
            xy_l_norm[..., 0] = xy_l_norm[..., 0] + i
            xy_l_norm[..., 1] = xy_l_norm[..., 1] + j
            xy_l_rescale = (xy_l_norm+0.5)*key_scale

            xy_proj.append(xy_l_rescale)

            xy_l_norm[..., 0] = xy_l_norm[..., 0]/(w-1)*2-1
            xy_l_norm[..., 1] = xy_l_norm[..., 1]/(h-1)*2-1
            _key_value = F.grid_sample(
                key_value, xy_l_norm, align_corners=True)
            key_values.append(_key_value)

    xy_proj = torch.stack(xy_proj, dim=1)
    mask = (xy_proj[..., 0] > 0)*(xy_proj[..., 0] < ori_w) * \
        (xy_proj[..., 1] > 0)*(xy_proj[..., 1] < ori_h)

    xy_proj_back = torch.cat([xy_proj, torch.ones(
        *xy_proj.shape[:-1], 1, device=xy_proj.device)], dim=-1)
    xy_proj_back = rearrange(xy_proj_back, 'b n h w c -> b c (n h w)')
    xy_proj_back = homo_r@xy_proj_back
    
    xy_proj_back = rearrange(
        xy_proj_back, 'b c (n h w) -> b n h w c', h=h, w=w)
    xy_proj_back = xy_proj_back[..., :2]/xy_proj_back[..., 2:]

    xy = get_x_2d(ori_w, ori_h)[:, :, :2]
    xy = xy[query_scale//2::query_scale, query_scale//2::query_scale]
    xy = torch.tensor(xy, device=key_value.device).float()[
        None, None]

    xy_rel = (xy_proj_back-xy)/query_scale

    key_values = torch.stack(key_values, dim=1)

    return key_values, xy_rel, mask


def get_query_value(query, key_value, xy_l, homo_r, img_h_l, img_w_l, img_h_r=None, img_w_r=None):
    if img_h_r is None:
        img_h_r = img_h_l
        img_w_r = img_w_l

    b = query.shape[0]
    m = key_value.shape[1]

    key_values = []
    masks = []
    xys = []

    for i in range(m):
        _, _, q_h, q_w = query.shape
        _key_value, _xy, _mask = get_key_value(key_value[:, i], xy_l[:, i], homo_r[:, i],
                                               img_h_l, img_w_l, img_w_r, q_h)

        key_values.append(_key_value)
        xys.append(_xy)
        masks.append(_mask)

    key_value = torch.cat(key_values, dim=1)
    xy = torch.cat(xys, dim=1)
    mask = torch.cat(masks, dim=1)

    return query, key_value, xy, mask
