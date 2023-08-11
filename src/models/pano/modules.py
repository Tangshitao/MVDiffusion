import torch
import torch.nn as nn
from einops import rearrange
from ..modules.resnet import BasicResNetBlock
from ..modules.transformer import BasicTransformerBlock, PosEmbedding
from .utils import get_query_value

class CPBlock(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.attn1 = CPAttn(dim, flag360=flag360)
        self.attn2 = CPAttn(dim, flag360=flag360)
        self.resnet = BasicResNetBlock(dim, dim, zero_init=True)

    def forward(self, x, correspondences, img_h, img_w, R, K, m):
        x = self.attn1(x, correspondences, img_h, img_w, R, K, m)
        x = self.attn2(x, correspondences, img_h, img_w, R, K, m)
        x = self.resnet(x)
        return x


class CPAttn(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.flag360 = flag360
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = PosEmbedding(2, dim//4)

    def forward(self, x, correspondences, img_h, img_w, R, K, m):
        b, c, h, w = x.shape
        x = rearrange(x, '(b m) c h w -> b m c h w', m=m)
        outs = []

        for i in range(m):
            indexs = [(i-1+m) % m, (i+1) % m]

            xy_l=correspondences[:, i, indexs]
            xy_r=correspondences[:, indexs, i]
           
            x_left = x[:, i]
            x_right = x[:, indexs]
            
            R_right = R[:, indexs]
            K_right = K[:, indexs]

            l = R_right.shape[1]
            
            R_left = R[:, i:i+1].repeat(1, l, 1, 1)
            K_left = K[:, i:i+1].repeat(1, l, 1, 1)

            R_left = R_left.reshape(-1, 3, 3)
            R_right = R_right.reshape(-1, 3, 3)
            K_left = K_left.reshape(-1, 3, 3)
            K_right = K_right.reshape(-1, 3, 3)
            
            homo_r = (K_left@torch.inverse(R_left) @
                      R_right@torch.inverse(K_right))

            homo_r = rearrange(homo_r, '(b l) h w -> b l h w', b=xy_r.shape[0])
            query, key_value, key_value_xy, mask = get_query_value(
                x_left, x_right, xy_l, homo_r, img_h, img_h)

            key_value_xy = rearrange(key_value_xy, 'b l h w c->(b h w) l c')
            key_value_pe = self.pe(key_value_xy)

            key_value = rearrange(
                key_value, 'b l c h w-> (b h w) l c')
            mask = rearrange(mask, 'b l h w -> (b h w) l')

            key_value = (key_value + key_value_pe)*mask[..., None]

            query = rearrange(query, 'b c h w->(b h w) c')[:, None]
            query_pe = self.pe(torch.zeros(
                query.shape[0], 1, 2, device=query.device))

            out = self.transformer(query, key_value, query_pe=query_pe)

            out = rearrange(out[:, 0], '(b h w) c -> b c h w', h=h, w=w)
            outs.append(out)
        out = torch.stack(outs, dim=1)

        out = rearrange(out, 'b m c h w -> (b m) c h w')

        return out

