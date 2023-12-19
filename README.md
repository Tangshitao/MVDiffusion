# MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion, (NeurIPS2023, spotlight)

<div align="center">
  <img width="800" src="assets/teaser.gif">
</div>

# [Project page](https://mvdiffusion.github.io/) |  [Paper](https://arxiv.org/abs/2307.01097) | [Demo](https://huggingface.co/spaces/tangshitao/MVDiffusion)

## Citation

If you use our work in your research, please cite it as follows:

```bibtex
@article{tang2023MVDiffusion,
  title={MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion},
  author={Tang, Shitao and Zhang, Fuayng and Chen, Jiacheng and Wang, Peng and Yasutaka, Furukawa},
  journal={arXiv preprint 2307.01097},
  year={2023}
}
```

### Updates: MVDiffusion is able to extrapolate a single perspective image into a 360-degree view panorama. The paper has been updated. 

## Installation

Install the necessary packages by running the following command:

```bash
pip install -r requirements.txt
```

## Model Zoo

We provide baseline results and models for the following:

- [Panorama model](https://www.dropbox.com/scl/fi/yx9e0lj4fwtm9xh2wlhhg/pano.ckpt?rlkey=kowqygw7vt64r3maijk8klfl0&dl=0)
- [Panorama outpainting model](https://www.dropbox.com/scl/fi/3mtj06qx6mxt4eme1oz2r/pano_outpaint.ckpt?rlkey=xat6cwt47lzfjawum05xa5ftq&dl=0)
- [Depth-conditioned generation model](https://www.dropbox.com/scl/fi/56hcmoj0tx7lza7s2m0jq/depth_gen.ckpt?rlkey=upcdbd4kxd9zwms78dssm3gh7&dl=0)
- [Depth pretrained model](https://www.dropbox.com/scl/fi/i1u8jzadcq1mx23aef7s6/depth_single_view.ckpt?rlkey=4in8g1g8vxrbx21o7do4hqy3c&dl=0)

Please put those files in 'MVDiffusion/weights'.

## Demo

Test the demo by running:
- Text conditioned generation
```bash
python demo.py --text "This kitchen is a charming blend of rustic and modern, featuring a large reclaimed wood island with marble countertop, a sink surrounded by cabinets. To the left of the island, a stainless-steel refrigerator stands tall. To the right of the sink, built-in wooden cabinets painted in a muted."
```
- Dual contioned generation
```bash
python demo.py --text_path assets/prompts.txt --image_path assets/outpaint_example.png
```

## Data

- Panorama generation, please download data from [matterport3D](https://niessner.github.io/Matterport/) skybox data and [labels](https://www.dropbox.com/scl/fi/recc3utsvmkbgc2vjqxur/mp3d_skybox.tar?rlkey=ywlz7zvyu25ovccacmc3iifwe&dl=0).
```
├── data
    ├── mp3d_skybox
      ├── train.npy
      ├── test.npy
      ├── 5q7pvUzZiYa
        ├──blip3
        ├──matterport_skybox_images
      ├── 1LXtFkjw3qL
      ├── ....
```
- Depth conditioned generation, please download data from [scannet](http://www.scan-net.org/), [training labels](https://www.dropbox.com/scl/fi/lwgcnrxfaiwic3kuqrwh4/scannet_train.tar?rlkey=dom83ygwvnjkyuog3y8wue30j&dl=0), and [testing labels](https://www.dropbox.com/scl/fi/lzh6vrj4ck37t7efymxar/scannet_test.tar?rlkey=cr1k0d06941qusgan6t6ks863&dl=0).
```
├── data
    ├── scannet
      ├── train
        ├── scene0435_01
          ├── color
          ├── depth
          ├── intrinsic
          ├── pose
          ├── prompt
          ├── key_frame_0.6.txt
          ├── valid_frames.npy
      ├── test
```

## Testing

Execute the following scripts for testing:

- ```sh test_pano.sh```: Generate 8 multi-view panoramic images in the Matterport3D testing dataset.
- ```sh test_pano_outpaint.sh```: Generate 8 multi-view images conditioned on a single view image (outpaint) in the Matterport3D testing dataset.
- ```sh test_depth_fix_frames.sh```: Generate 12 depth-conditioned images in the ScanNet testing dataset.
- ```sh test_depth_fix_interval.sh```: Generate a sequence of depth-conditioned images (every 20 frames) in the ScanNet testing dataset.
- ```sh test_depth_two_stage.sh```: Generate a sequence of depth-conditioned images (key frames), and interpolate the in-between images, in the ScanNet testing dataset.

After running either ```sh test_depth_fix_interval.sh``` or ```sh test_depth_two_stage.sh```, you can use [TSDF fusion](https://github.com/andyzeng/tsdf-fusion-python) to get textured mesh.

## Training

Execute the following scripts for training:

- ```sh train_pano.sh```: Train the panoramic image generation model.
- ```sh train_pano_outpaint.sh```: Train the panoramic image outpaint model.
- ```sh train_depth.sh```: Train the depth conditioned generation model.

# Custom data
Panorama generation: 

1. Convert the panorama into 6 skybox images using the provided tool, [Equirec2Perspec](https://github.com/fuenwang/Equirec2Perspec). You will get left, right, front, back, up, and down images. 
2. Convert the panorama to 8 perspective images. Each image will capture a 45-degree horizontal view. Four of these images will overlap with the skybox images, specifically the left, right, front, and back views. 
3. Once you have the perspective images, you can use [BLIP2](https://github.com/salesforce/LAVIS) to generate prompts from them.

Multi-view Depth-to-Image Generation: 

1. Using Scannet Format: For this, you would typically follow the structure and format of the Scannet dataset.
2. use [BLIP2](https://github.com/salesforce/LAVIS) to generate prompts from each perspective image.


## License

This project is licensed under the terms of the MIT license.

## Contact

For any questions, feel free to contact us at [shitaot@sfu.ca].
