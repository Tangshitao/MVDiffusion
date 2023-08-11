
import torch
import torch.nn.functional as F
from einops import rearrange
from ..modules.utils import back_projection, get_x_2d



def get_correspondence(depth, pose, K, x_2d):
    b, h, w = depth.shape
    x3d = back_projection(depth, pose, K, x_2d)
    x3d = rearrange(x3d, 'b h w c -> b c (h w)')
    x3d = K[:, :3, :3]@x3d
    x3d = rearrange(x3d, 'b c (h w) -> b h w c', h=h, w=w)
    x2d = x3d[..., :2]/(x3d[..., 2:3]+1e-6)

    mask = depth == 0
    x2d[mask] = -1000000
    x3d[mask] = -1000000

    return x2d, x3d

def get_key_value(key_value, xy_l, xy_r, depth_query, depths, pose_rel, K, ori_h, ori_w, ori_h_r, ori_w_r, query_h, query_w):

    b, c, h, w = key_value.shape
    query_scale = ori_h//query_h
    key_scale = ori_h_r//h

    xy_l = xy_l[:, query_scale//2::query_scale,
                query_scale//2::query_scale]/key_scale-0.5

    key_values = []

    xy_proj = []
    depth_proj = []
    mask_proj = []
    kernal_size = 1
    depth_query = depth_query[:, query_scale//2::query_scale,query_scale//2::query_scale]
    for i in range(0-kernal_size//2, 1+kernal_size//2):
        for j in range(0-kernal_size//2, 1+kernal_size//2):
            xy_l_norm = xy_l.clone()
            # displacement
            xy_l_norm[..., 0] = xy_l_norm[..., 0] + i
            xy_l_norm[..., 1] = xy_l_norm[..., 1] + j
            xy_l_rescale = (xy_l_norm+0.5)*key_scale
            xy_l_round = xy_l_rescale.round().long()
            mask = (xy_l_round[..., 0] >= 0)*(xy_l_round[..., 0] < ori_w) * (
                xy_l_round[..., 1] >= 0)*(xy_l_round[..., 1] < ori_h)
            xy_l_round[..., 0] = torch.clamp(xy_l_round[..., 0], 0, ori_w-1)
            xy_l_round[..., 1] = torch.clamp(xy_l_round[..., 1], 0, ori_h-1)

            depth_i = torch.stack([depths[b_i, xy_l_round[b_i, ..., 1], xy_l_round[b_i, ..., 0]]
                                  for b_i in range(b)])
            mask = mask*(depth_i > 0)
            depth_i[~mask] = 1000000
            depth_proj.append(depth_i)
            
            
            mask_proj.append(mask*(depth_query>0))

            xy_proj.append(xy_l_rescale.clone())

            xy_l_norm[..., 0] = xy_l_norm[..., 0]/(w-1)*2-1
            xy_l_norm[..., 1] = xy_l_norm[..., 1]/(h-1)*2-1
            _key_value = F.grid_sample(
                key_value, xy_l_norm, align_corners=True)
            key_values.append(_key_value)

    xy_proj = torch.stack(xy_proj, dim=1)
    depth_proj = torch.stack(depth_proj, dim=1)
    mask_proj = torch.stack(mask_proj, dim=1)

    xy_proj = rearrange(xy_proj, 'b n h w c -> (b n) h w c')
    depth_proj = rearrange(depth_proj, 'b n h w -> (b n) h w')


    xy = get_x_2d(ori_w, ori_h)[:, :, :2]
    xy = xy[query_scale//2::query_scale, query_scale//2::query_scale]
    
    xy = torch.tensor(xy, device=key_value.device).float()[
        None].repeat(xy_proj.shape[0], 1, 1, 1)   
    
    xy_rel = (depth_query-depth_proj).abs()[...,None] # depth check

    xy_rel = rearrange(xy_rel, '(b n) h w c -> b n h w c', b=b)

    key_values = torch.stack(key_values, dim=1)
   
    return key_values, xy_rel, mask_proj


def get_query_value(query, key_value, xy_l, xy_r, depth_query, depths, pose_rel, K, img_h_l, img_w_l, img_h_r=None, img_w_r=None):
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
        _key_value, _xy, _mask = get_key_value(key_value[:, i], xy_l[:, i], xy_r[:, i], depth_query, depths[:, i], pose_rel[:, i], K,
                                               img_h_l, img_w_l, img_h_r, img_w_r, q_h, q_w)

        key_values.append(_key_value)
        xys.append(_xy)
        masks.append(_mask)

    key_value = torch.cat(key_values, dim=1)
    xy = torch.cat(xys, dim=1)
    mask = torch.cat(masks, dim=1)

    return query, key_value, xy, mask