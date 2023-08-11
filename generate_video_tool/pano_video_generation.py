import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import lib.Equirec2Perspec as E2P
import lib.multi_Perspec2Equirec as m_P2E
import uuid
from PIL import Image
from tqdm import tqdm



def generate_video(image_paths, out_dir, gen_video=True):
    pers = [cv2.imread(image_path) for image_path in image_paths]

    ee = m_P2E.Perspective(pers,
                            [[90, 0, 0], [90, 45, 0], [90, 90, 0], [90, 135, 0],
                             [90, 180, 0], [90, 225, 0], [90, 270, 0], [90, 315, 0]]
                            )

    new_pano = ee.GetEquirec(2048, 4096)
    cv2.imwrite(os.path.join(out_dir, 'pano.png'), new_pano.astype(np.uint8)[540:-540])
    if not gen_video:
        return
    equ = E2P.Equirectangular(new_pano)
    fov = 60
    video_size = (450, 600)
    
    img = equ.GetPerspective(fov, 0, 0, video_size[0], video_size[1])  # Specify parameters(FOV, theta, phi, height, width)

    h = img.shape[0]
    margin = 0
    if margin > 0:
        img = img[margin:-margin]
    size = (img.shape[1], img.shape[0])

    tmp_video_path = '/tmp/' + str(uuid.uuid4()) + '.mp4'
    save_video_path = os.path.join(out_dir, 'video.mp4')
    out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'MP4V'), 60, size)

    interval_deg = 0.5
    num_frames = int(360 / interval_deg)
    for i in range(num_frames):
        deg = i * interval_deg
        img = equ.GetPerspective(fov, deg, 0, video_size[0], video_size[1])  # Specify parameters(FOV, theta, phi, height, width)
        h = img.shape[0]
        if margin > 0:
            img = img[margin:-margin]
        img = np.clip(img, 0, 255).astype(np.uint8)
        out.write(img)
    out.release()
   # os.system(f"ffmpeg -y -i {tmp_video_path} -vcodec libx264 {save_video_path}")

if __name__ == '__main__':
    data_dir='logs/tb_logs/test_mp3d_outpaint=2/version_0/images'
    out_dir='out_paint_example'
    for scene in tqdm(os.listdir(data_dir)):
        data_path = os.path.join(data_dir, scene)
        image_paths = [os.path.join(data_path, f'{i}.png') for i in range(8)]
        _out_dir=os.path.join(out_dir, scene)
        os.makedirs(_out_dir, exist_ok=True)
        for i, image_path in enumerate(image_paths):
            img = cv2.imread(image_path)
            cv2.imwrite(os.path.join(_out_dir, '{}.png'.format(i)), img)
        generate_video(image_paths, _out_dir, gen_video=False)