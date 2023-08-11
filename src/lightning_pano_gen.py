import pytorch_lightning as pl
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
import torch
import os
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from .models.pano.MVGenModel import MultiViewBaseModel


class PanoGenerator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs'] if 'max_epochs' in config['train'] else 0
        self.diff_timestep = config['model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']

        self.tokenizer = CLIPTokenizer.from_pretrained(
            config['model']['model_id'], subfolder="tokenizer", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(
            config['model']['model_id'], subfolder="text_encoder", torch_dtype=torch.float16)

        self.vae, self.scheduler, unet = self.load_model(
            config['model']['model_id'])
        self.mv_base_model = MultiViewBaseModel(
            unet, config['model'])
        self.trainable_params = self.mv_base_model.trainable_parameters

        self.save_hyperparameters()
       
    def load_model(self, model_id):
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae")
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet")
        return vae, scheduler, unet

    @torch.no_grad()
    def encode_text(self, text, device):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask)

        return prompt_embeds[0].float(), prompt_embeds[1]

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]

        x_input = x_input.permute(0, 1, 4, 2, 3)  # (bs, 2, 3, 512, 512)
        x_input = x_input.reshape(-1,
                                  x_input.shape[-3], x_input.shape[-2], x_input.shape[-1])
        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()
        z = z.reshape(b, -1, z.shape[-3], z.shape[-2],
                      z.shape[-1])  # (bs, 2, 4, 64, 64)

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        z = z.float()
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)
        images = []
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
        image = (image * 255).round().astype('uint8')

        return image

    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            param_groups.append({"params": params, "lr": self.lr * lr_scale})
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
            'interval': 'epoch',  # update the learning rate after each epoch
            'name': 'cosine_annealing_lr',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):      
        meta = {
            'K': batch['K'],
            'R': batch['R']
        }

        device = batch['images'].device
        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(self.encode_text(
                prompt, device)[0])
        latents = self.encode_image(
            batch['images'], self.vae)
        t = torch.randint(0, self.scheduler.num_train_timesteps,
                        (latents.shape[0],), device=latents.device).long()
        prompt_embds = torch.stack(prompt_embds, dim=1)

        noise = torch.randn_like(latents)
        noise_z = self.scheduler.add_noise(latents, noise, t)
        t = t[:, None].repeat(1, latents.shape[1])
        denoise = self.mv_base_model(
            noise_z, t, prompt_embds, meta)
        target = noise       

        # eps mode
        loss = torch.nn.functional.mse_loss(denoise, target)
        self.log('train_loss', loss)
        return loss

    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch):
        latents = torch.cat([latents]*2)
        timestep = torch.cat([timestep]*2)
        
        R = torch.cat([batch['R']]*2)
        K = torch.cat([batch['K']]*2)
      
        meta = {
            'K': K,
            'R': R,
        }

        return latents, timestep, prompt_embd, meta

    @torch.no_grad()
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model):
        latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latents_high_res, _timestep, prompt_embd, batch)

        noise_pred = model(
            latents, _timestep, _prompt_embd, meta)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred = self.inference(batch)
        images = ((batch['images']/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)
      
        # compute image & save
        if self.trainer.global_rank == 0:
            self.save_image(images_pred, images, batch['prompt'], batch_idx)

    @torch.no_grad()
    def inference(self, batch):
        images = batch['images']
        bs, m, h, w, _ = images.shape
        device = images.device

        latents= torch.randn(
            bs, m, 4, h//8, w//8, device=device)

        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(self.encode_text(
                prompt, device)[0])
        prompt_embds = torch.stack(prompt_embds, dim=1)

        prompt_null = self.encode_text('', device)[0]
        prompt_embd = torch.cat(
            [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]]*m, dim=1)

            noise_pred = self.forward_cls_free(
                latents, _timestep, prompt_embd, batch, self.mv_base_model)

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
        images_pred = self.decode_latent(
            latents, self.vae)
       
        return images_pred
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images_pred = self.inference(batch)

        images = ((batch['images']/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)
       
       
        scene_id = batch['image_paths'][0].split('/')[2]
        image_id=batch['image_paths'][0].split('/')[-1].split('.')[0].split('_')[0]
        
        output_dir = batch['resume_dir'][0] if 'resume_dir' in batch else os.path.join(self.logger.log_dir, 'images')
        output_dir=os.path.join(output_dir, "{}_{}".format(scene_id, image_id))
        
        os.makedirs(output_dir, exist_ok=True)
        for i in range(images.shape[1]):
            path = os.path.join(output_dir, f'{i}.png')
            im = Image.fromarray(images_pred[0, i])
            im.save(path)
            im = Image.fromarray(images[0, i])
            path = os.path.join(output_dir, f'{i}_natural.png')
            im.save(path)
        with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
            for p in batch['prompt']:
                f.write(p[0]+'\n')

    @torch.no_grad()
    def save_image(self, images_pred, images, prompt, batch_idx):

        img_dir = os.path.join(self.logger.log_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        with open(os.path.join(img_dir, f'{self.global_step}_{batch_idx}.txt'), 'w') as f:
            for p in prompt:
                f.write(p[0]+'\n')
        if images_pred is not None:
            for m_i in range(images_pred.shape[1]):
                im = Image.fromarray(images_pred[0, m_i])
                im.save(os.path.join(
                    img_dir, f'{self.global_step}_{batch_idx}_{m_i}_pred.png'))
                if m_i < images.shape[1]:
                    im = Image.fromarray(
                        images[0, m_i])
                    im.save(os.path.join(
                        img_dir, f'{self.global_step}_{batch_idx}_{m_i}_gt.png'))