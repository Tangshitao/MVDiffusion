import torch
import argparse
import yaml
from src.lightning_pano_gen import PanoGenerator
from src.lightning_pano_outpaint import PanoOutpaintGenerator
import numpy as np
import cv2
import os
from generate_video_tool.pano_video_generation import generate_video
from PIL import Image
torch.manual_seed(0)

def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--text', type=str, default='This kitchen is a charming blend of rustic and modern, featuring a large reclaimed wood island with marble countertop, a sink surrounded by cabinets. To the left of the island, a stainless-steel refrigerator stands tall. To the right of the sink, built-in wooden cabinets painted in a muted.')
    parser.add_argument(
        '--image_path', type=str, default=None, help='image condition outpainting')
    parser.add_argument('--gen_video',
                    action='store_true', help='generate video')
    parser.add_argument('--text_path',
                    type=str, help='text path allow to specify 8 texts')

    return parser.parse_args()

def resize_and_center_crop(img, size):
    H, W, _ = img.shape
    if H==W:
        img = cv2.resize(img, (size, size))
    elif H > W:
        current_size = int(size*H/W)
        img = cv2.resize(img, (size, current_size))
        # center crop to square
        margin_l=(current_size-size)//2
        margin_r=current_size-margin_l-size
        img=img[margin_l:-margin_r, :]
    else:
        current_size=int(size*W/H)
        img = cv2.resize(img, (current_size, size))
        margin_l=(current_size-size)//2
        margin_r=current_size-margin_l-size
        img=img[:, margin_l:-margin_r]
    return img

args = parse_args()
if args.image_path is None:
    config_file = 'configs/pano_generation.yaml'
    config = yaml.load(open(config_file, 'rb'), Loader=yaml.SafeLoader)
    model = PanoGenerator(config)
    model.load_state_dict(torch.load('weights/pano.ckpt', map_location='cpu')[
            'state_dict'], strict=True)
    model=model.cuda()
    img=None
else:

    config_file = 'configs/pano_generation_outpaint.yaml'
    config = yaml.load(open(config_file, 'rb'), Loader=yaml.SafeLoader)
    model = PanoOutpaintGenerator(config)
    model.load_state_dict(torch.load('weights/pano_outpaint.ckpt', map_location='cpu')[
            'state_dict'], strict=True)
    model=model.cuda()

    img=cv2.imread(args.image_path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=resize_and_center_crop(img, config['dataset']['resolution'])
    img=img/127.5-1

    img=torch.tensor(img).cuda()
    
resolution=config['dataset']['resolution']
Rs=[]
Ks=[]
for i in range(8):
    degree = (45*i) % 360
    K, R = get_K_R(90, degree, 0,
                    resolution, resolution)

    Rs.append(R)
    Ks.append(K)

images=torch.zeros((1,8,resolution,resolution, 3)).cuda()
if img is not None:
    images[0,0]=img


if args.text_path is not None:
    prompt=[]
    with open(args.text_path, 'r') as f:
        for i, line in enumerate(f):
            prompt.append(line.strip())
    if len(prompt)<8:
        raise ValueError('text file should contain 8 lines')
    args.text=prompt[0]
else:
    prompt=[args.text]*8
K=torch.tensor(Ks).cuda()[None]
R=torch.tensor(Rs).cuda()[None]

batch= {
        'images': images,
        'prompt': prompt,
        'R': R,
        'K': K
    }
images_pred=model.inference(batch)
res_dir=args.text[:20]
print('save in fold: {}'.format(res_dir))
os.makedirs(res_dir, exist_ok=True)
with open(os.path.join(res_dir, 'prompt.txt'), 'w') as f:
    f.write(args.text)
image_paths=[]
for i in range(8):
    im = Image.fromarray(images_pred[0,i])
    image_path=os.path.join(res_dir, '{}.png'.format(i))
    image_paths.append(image_path)
    im.save(image_path)
generate_video(image_paths, res_dir, args.gen_video)
