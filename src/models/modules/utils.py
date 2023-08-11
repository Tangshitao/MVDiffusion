import torch
import numpy as np
from einops import rearrange

def pi_inv(K, x, d):
    
    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2,
                                       1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    X_x = d * (x[..., 0] - cx) / fx
    X_y = d * (x[..., 1] - cy) / fy
    X_z = d

    X = torch.stack([X_x, X_y, X_z], dim=-1)
    return X


def inv_pose(R, t):
    Rwc = R.T
    tw = -Rwc.dot(t)
    return Rwc, tw


def transpose(R, t, X):
    b, h, w, c = X.shape
    X = rearrange(X, 'b h w c -> b c (h w)')

    X_after_R = R@X + t[:, :, None]

    X_after_R = rearrange(X_after_R, 'b c (h w) -> b h w c', h=h)
    return X_after_R


def back_projection(depth, pose, K, x_2d=None):
    b, h, w = depth.shape
    if x_2d is None:
        x_2d = x_2d_coords(h, w, device=depth.device)[
            None, ...].repeat(b, 1, 1, 1)

    X_3d = pi_inv(K, x_2d, depth)

    Rwc, twc = pose[:, :3, :3], pose[:, :3, 3]
    X_world = transpose(Rwc, twc, X_3d)

    X_world = X_world.reshape((-1, h, w, 3))
    return X_world

def get_x_2d(width, height):
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate(
        [x[..., None], y[..., None], z[..., None]], axis=-1).astype(np.float32)
    return xyz


def x_2d_coords(h, w, device):
    x_2d = torch.zeros((h, w, 2), device=device)
    for y in range(0, h):
        x_2d[y, :, 1] = y
    for x in range(0, w):
        x_2d[:, x, 0] = x
    return x_2d
