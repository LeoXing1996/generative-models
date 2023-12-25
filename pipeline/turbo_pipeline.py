
import copy
import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from .unet_utils import override_forward


def draw_points(img, src_points, tar_points):
    """draw poitns on image,
    src_points, tar_points: coordinates in [h, w] order.
    """
    is_pil = False
    if isinstance(img, Image.Image):
        is_pil = True
        img = np.array(img)

    for src, tar in zip(src_points, tar_points):
        s_x, s_y = src[1] * 2, src[0] * 2
        t_x, t_y = tar[1] * 2, tar[0] * 2

        s_x, s_y = int(s_x), int(s_y)
        t_x, t_y = int(t_x), int(t_y)

        cv2.circle(img, (s_x, s_y), 10, (255, 0, 0), -1)
        cv2.circle(img, (t_x, t_y), 10, (0, 0, 255), -1)

        cv2.arrowedLine(img, (s_x, s_y), (t_x, t_y),
                        (255, 255, 255), 4, tipLength=0.5)

    if is_pil:
        return Image.fromarray(img)
    return img


def write_log(info, target):
    print(info)
    with open(target, 'a+') as file:
        file.write(info)
        if not info.endswith('\n'):
            file.write('\n')


def point_tracking(F0, F1, handle_points, handle_points_init, r_p):
    with torch.no_grad():
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            r1, r2 = int(pi[0]) - r_p, int(pi[0]) + r_p + 1
            c1, c2 = int(pi[1]) - r_p, int(pi[1]) + r_p + 1
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) -
                        F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            # WARNING: no boundary protection right now
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            handle_points[i][0] = pi[0] - r_p + row
            handle_points[i][1] = pi[1] - r_p + col
        return handle_points


def check_handle_reach_target(handle_points, target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(
        map(lambda p, q: (p - q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 2.0).all()


def interpolate_feature_patch(feat, y, x, r):
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    Ia = feat[:, :, y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
    Ib = feat[:, :, y1 - r:y1 + r + 1, x0 - r:x0 + r + 1]
    Ic = feat[:, :, y0 - r:y0 + r + 1, x1 - r:x1 + r + 1]
    Id = feat[:, :, y1 - r:y1 + r + 1, x1 - r:x1 + r + 1]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


@dataclass
class TurboOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray]
    pred_x0_list: List[Image.Image]
    errors: List[float]

    init_latent: Optional[torch.Tensor] = None
    intermedia_feature: Optional[List[torch.Tensor]] = None


class TurboPipeline(StableDiffusionPipeline):

    @classmethod
    def build(cls):
        model_id = 'stabilityai/sd-turbo'
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder='vae').to('cuda', torch.bfloat16)
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder='unet').to('cuda', torch.float16)
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder='text_encoder').to('cuda', torch.float16)
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder='tokenizer')

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder='scheduler')

        pipeline = cls(vae, text_encoder, tokenizer, unet,
                       scheduler, None, None, None, False)

        pipeline.is_drag_unet = False
        return pipeline

    def convert_to_drag(self):
        self.unet.forward = override_forward(self.unet)
        self.is_drag_unet = True

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        save_pred_x0: bool = False,
        **kwargs,
    ):
        if num_inference_steps > 1 and self.is_drag_unet:
            raise ValueError('Only support 1 step for drag')

        if num_images_per_prompt > 1 and self.is_drag_unet:
            raise ValueError('Only support 1 image for drag.')

        pred_x0_list = []
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # to deal with lora scaling and other possible forward hooks
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get(
                "scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        init_latent = latents.clone()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(
                self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop

        self.scheduler.set_timesteps(
            num_inference_steps, device=device, **kwargs)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = len(timesteps) - \
                               num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat(
                    [latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                # predict the noise residual
                if self.is_drag_unet:
                    noise_pred, intermedia_feature = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        return_intermediates=True,
                    )
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                    intermedia_feature = None

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                denoising_output = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                latents = denoising_output['prev_sample']
                pred_x0 = denoising_output['pred_original_sample']

                if save_pred_x0:
                    pred_x0_list.append(pred_x0)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # latents = self.scheduler.step(
                #     noise_pred, t, latents, **extra_step_kwargs, return_dict=True)[0]
        latents = latents.to(self.device, self.vae.dtype)
        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize)

        if save_pred_x0:
            pred_x0_list_new = []
            for pred_x0 in pred_x0_list:
                pred_x0 = pred_x0.to(self.device, self.vae.dtype)
                pred_x0_img = self.vae.decode(
                    pred_x0 / self.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                pred_x0 = self.image_processor.postprocess(
                    pred_x0_img, output_type=output_type, do_denormalize=do_denormalize)
                pred_x0_list_new.append(pred_x0)
            pred_x0_list = pred_x0_list_new

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, )

        return TurboOutput(images=image, pred_x0_list=pred_x0_list, errors=None,
                           init_latent=init_latent, intermedia_feature=intermedia_feature)

    def drag(self,
             init_latent: torch.Tensor,
             src_points: List,
             tar_points: List,
             prompt: str,
             mask: np.ndarray,
             lr: float,
             drag_steps: int,
             layer_idxs: List[int],
             r_m: int,
             r_p: int,
             lam: float,
             super_res_h: int,
             super_res_w: int,
             save_dir: Optional[str] = None,
             return_intermediate: bool = False,
             ):

        device = self._execution_device

        self.scheduler.set_timesteps(1, device=device)
        t = self.scheduler.timesteps[0]

        intermediate = []

        generator = torch.Generator()
        with torch.no_grad():
            # prepare prompt
            prompt_embeds, _ = self.encode_prompt(
                prompt,
                device,
                1,
                False,
                '',
                prompt_embeds=None,
                negative_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )

            init_latent_scaled = self.scheduler.scale_model_input(
                init_latent, t)
            unet_output, F0 = self.forward_unet_feature_for_drag(
                init_latent_scaled.to(dtype=self.unet.dtype),
                t,
                prompt_embeds,
                interp_res_h=super_res_h,
                interp_res_w=super_res_w,
                layer_idx=layer_idxs
            )
            generator.manual_seed(42)
            denoising_output = self.scheduler.step(
                unet_output, t, init_latent, generator=generator,
                return_dict=True)
            # NOTE: reset step_index
            self.scheduler._init_step_index(t)

            pred_xt_1 = denoising_output['prev_sample'].float()
            pred_x0 = denoising_output['pred_original_sample'].float()

        # convert unet, latent, and other variables to fp32 since we use autocast
        unet_ori_type = self.unet.dtype
        self.unet = self.unet.to(torch.float32)
        init_latent = init_latent.float()
        prompt_embeds = prompt_embeds.float()
        F0 = F0.float()

        init_latent.requires_grad_(True)
        optimizer = torch.optim.Adam([init_latent], lr=lr)
        handle_points_init = copy.deepcopy(src_points)
        interp_mask = F.interpolate(mask, (init_latent.shape[2], init_latent.shape[3]),
                                    mode='nearest')

        scaler = torch.cuda.amp.GradScaler()
        for idx in range(drag_steps):
            print(idx)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if save_dir:
                    write_log(f'Mean: {init_latent.mean().item()}',
                              osp.join(save_dir, 'log.txt'))
                init_latent_scaled = self.scheduler.scale_model_input(
                    init_latent, t)
                unet_output, F1 = self.forward_unet_feature_for_drag(
                    init_latent_scaled,
                    t,
                    prompt_embeds,
                    interp_res_h=super_res_h,
                    interp_res_w=super_res_w,
                    layer_idx=layer_idxs
                )
                generator.manual_seed(42)
                denoising_output = self.scheduler.step(
                    unet_output, t, init_latent, generator=generator,
                    return_dict=True)
                self.scheduler._init_step_index(t)
                pred_xt_1_update = denoising_output['prev_sample']
                pred_x0_update = denoising_output['pred_original_sample']

                with torch.no_grad():
                    if save_dir is not None:
                        _img = self.vae.decode(
                            pred_xt_1_update / self.vae.config.scaling_factor,
                            return_dict=False,
                        )[0]
                        do_denormalize = [True] * _img.shape[0]
                        _img = self.image_processor.postprocess(
                            _img, output_type='pil',
                            do_denormalize=do_denormalize)[0]
                        _img = draw_points(_img, src_points, tar_points)
                        _img.save(osp.join(save_dir, f'{idx}.png'))

                if idx != 0:
                    src_points = point_tracking(F0, F1, src_points,
                                                handle_points_init, r_p)
                    print('new handle points', src_points)
                if check_handle_reach_target(src_points, tar_points):
                    break

                loss = 0.0
                for i in range(len(src_points)):
                    pi, ti = src_points[i], tar_points[i]
                    # skip if the distance between target and source is less than 1
                    if (ti - pi).norm() < 2.:
                        continue

                    di = (ti - pi) / (ti - pi).norm()

                    # motion supervision
                    f0_patch = F1[:, :,
                                  int(pi[0]) - r_m:int(pi[0]) + r_m + 1,
                                  int(pi[1]) - r_m:int(pi[1]) + r_m +
                                  1].detach()
                    f1_patch = interpolate_feature_patch(F1, pi[0] + di[0],
                                                         pi[1] + di[1], r_m)
                    loss += ((2 * r_m + 1)**2) * \
                             F.l1_loss(f0_patch, f1_patch)

                # masked region must stay unchanged
                loss += lam * ((pred_xt_1 - pred_xt_1_update) *
                               (1.0 - interp_mask)).abs().sum()
                # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
                # print('loss total=%f' % (loss.item()))
                write_log('loss total=%f' % (loss.item()),
                          osp.join(save_dir, 'log.txt'))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            init_latent = init_latent.to(self.device, self.unet.dtype)
            init_latent_scaled = self.scheduler.scale_model_input(
                init_latent, t)
            noise_pred = self.unet(
                init_latent_scaled,
                t,
                prompt_embeds,
                cross_attention_kwargs=None,
            )
            generator.manual_seed(42)
            denoising_output = self.scheduler.step(
                noise_pred, t, init_latent, generator=generator,
                return_dict=True)
            self.scheduler._init_step_index(t)
            # pred_x0 = denoising_output['pred_original_sample']
            prev_sample = denoising_output['prev_sample']
            prev_sample = prev_sample.to(device, self.vae.dtype)
            image_updated = self.vae.decode(
                prev_sample / self.vae.config.scaling_factor,
                return_dict=False,
            )[0]
            do_denormalize = [True] * image_updated.shape[0]
            image_updated = self.image_processor.postprocess(
                image_updated, output_type='pil', do_denormalize=do_denormalize)[0]
            image_updated = draw_points(image_updated, src_points, tar_points)

        self.unet = self.unet.to(unet_ori_type)
        return init_latent, image_updated

    def forward_unet_feature_for_drag(
            self, latent, t, prompt_embeds, layer_idx=[3],
            interp_res_h=256, interp_res_w=256,
            down_residual=None, mid_residual=None):

        unet_output, all_intermediate_features = self.unet(
            latent,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            return_intermediates=True,
        )

        all_return_features = []
        for idx in layer_idx:
            feat = all_intermediate_features[idx]
            feat = F.interpolate(
                feat, (interp_res_h, interp_res_w), mode='bilinear')
            all_return_features.append(feat)
        return_features = torch.cat(all_return_features, dim=1)
        return unet_output, return_features
