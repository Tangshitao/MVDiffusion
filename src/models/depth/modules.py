import torch
import torch.nn as nn
from einops import rearrange
from ..modules.resnet import BasicResNetBlock
from ..modules.transformer import BasicTransformerBlock, PosEmbedding
from .utils import get_query_value


class ImageEncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.nonlinearity = nn.SiLU()
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.conv2.weight.data.fill_(0)
        self.conv2.bias.data.fill_(0)

    def forward(self, lr_x):
        hidden_states = lr_x
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)


        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        return hidden_states

class CPBlock(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.attn1 = CPAttn(dim, flag360=flag360)
        self.resnet = BasicResNetBlock(dim, dim, zero_init=True)

    def forward(self, x, reso, cp_package, m):
        x = self.attn1(x, reso, cp_package, m)
        x = self.resnet(x)
        return x


class CPAttn(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.flag360 = flag360
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = PosEmbedding(2, dim//2)

    def forward(self, x, reso, cp_package, m):
        _, c, h, w = x.shape
        x = rearrange(x, '(b m) c h w -> b m c h w', m=m)
        b=x.shape[0]
        img_h, img_w = reso
        outs = []

        poses = cp_package['poses']
        K = cp_package['K']
        depths = cp_package['depths']
        correspondence = cp_package['correspondence']
        overlap_mask=cp_package['overlap_mask']
       
        for b_i in range(b):
            _outs=[]
            for i in range(m):
                
                indexs = [j for j in range(m) if overlap_mask[b_i, i, j] and i!=j]
                if len(indexs)==0: # if the image does not have overlap with others, use the nearby images
                    if i==0:
                        indexs=[1]
                    elif i==m-1:
                        indexs=[m-2]
                    else:
                        indexs=[i-1, i+1]

                xy_l = []
                xy_r = []
                x_right = []
                
                xy_l = correspondence[b_i:b_i+1, i, indexs]
                xy_r = correspondence[b_i:b_i+1, indexs, i]
                    
                x_left = x[b_i:b_i+1, i]
                x_right = x[b_i:b_i+1, indexs]  # bs, l, h, w, c
                pose_l = poses[b_i:b_i+1, i]
                pose_r = poses[b_i:b_i+1, indexs]
                
                pose_rel = torch.inverse(pose_l)[:, None]@pose_r
                _depths=depths[b_i:b_i+1, indexs]
                depth_query=depths[b_i:b_i+1, i]
                _K=K[b_i:b_i+1]
                
                query, key_value, key_value_xy, mask = get_query_value(
                    x_left, x_right, xy_l, xy_r, depth_query, _depths, pose_rel, _K, img_h, img_w, img_h, img_w)
              

                key_value_xy = rearrange(key_value_xy, 'b l h w c->(b h w) l c')
                key_value_pe = self.pe(key_value_xy)
               

                query = rearrange(query, 'b c h w->(b h w) c')[:, None]

                key_value = rearrange(
                    key_value, 'b l c h w-> (b h w) l c')
                mask = rearrange(mask, 'b l h w -> (b h w) l')

                key_value = (key_value + key_value_pe)*mask[..., None]
                query_pe = self.pe(torch.zeros(
                    query.shape[0], 1, 1, device=query.device))
    
                out = self.transformer(query, key_value, query_pe)

                out = rearrange(out[:, 0], '(b h w) c -> b c h w', h=h, w=w)
                _outs.append(out)
            _outs=torch.cat(_outs)
            outs.append(_outs)
            
                
        out = torch.stack(outs)
        out=rearrange(out, 'b m c h w -> (b m) c h w')
      
        return out
