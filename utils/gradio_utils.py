import datetime
import os
import os.path as osp
from copy import deepcopy
from typing import List, Optional

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from mmengine import Config
from PIL import Image
from PIL.ImageOps import exif_transpose

from pipeline import TurboPipeline


def generate_image(prompt: str,
                   pipeline: Optional[TurboPipeline] = None,
                   seed: Optional[int] = None):

    if pipeline is None:
        gr.Info('Build pipeline for the first run...')
        print('Build pipeline for the first run...')
        pipeline = TurboPipeline.build()
        pipeline.convert_to_drag()

    pipeline_kwargs = {}
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        print(f'Set seed as {seed}')
        pipeline_kwargs['generator'] = generator

    output = pipeline(prompt, num_inference_steps=1,
                      guidance_scale=0.0, **pipeline_kwargs)
    img = output.images[0]
    init_latent = output.init_latent

    return img, init_latent, pipeline


def drag_image(pipeline: TurboPipeline,
               source_image: np.ndarray,
               prompt: str,
               points: List,
               mask: np.ndarray,
               #    intermedia_features: List[torch.Tensor],
               init_latent: torch.Tensor,
               lr: float,
               drag_steps: int,
               lam: float,
               save_root: str):
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')

    save_path = osp.join(save_root, now)

    full_h, full_w = source_image.shape[:2]
    sup_res_h = int(0.5*full_h)
    sup_res_w = int(0.5*full_w)

    r_m, r_p = 1, 3

    # process mask
    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (sup_res_h, sup_res_w), mode="nearest")

    # process points
    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor(
            [point[1]/full_h*sup_res_h, point[0]/full_w*sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print('handle points:', handle_points)
    print('target points:', target_points)

    init_latent_orig = init_latent.clone()

    # save init latent and img
    os.makedirs(save_path, exist_ok=True)
    inp_dict = dict(
        init_latent=init_latent_orig,
        mask=mask,
        handle_points=handle_points,
        target_points=target_points,
    )
    torch.save(inp_dict, osp.join(save_path, 'inp_dict.pt'))
    cfg_dict = dict(
        prompt=prompt,
        lr=lr,
        r_m=r_m,
        r_p=r_p,
        lam=lam,
        sup_res_h=sup_res_h,
        sup_res_w=sup_res_w,
    )
    Image.fromarray(source_image.astype(np.uint8)).save(
        osp.join(save_path, 'source_image.png')
    )
    Config(cfg_dict).dump(osp.join(save_path, 'cfg.py'))
    # return init_latent_orig, source_image

    init_latent_updated, img_updated = pipeline.drag(
        init_latent=init_latent_orig,
        prompt=prompt,
        src_points=handle_points,
        tar_points=target_points,
        mask=mask,
        lr=lr,
        drag_steps=drag_steps,
        r_m=r_m,
        r_p=r_p,
        lam=lam,
        super_res_h=sup_res_h,
        super_res_w=sup_res_w,
        save_dir=save_path,
        layer_idxs=[2, 3],
    )
    img_updated = np.array(img_updated)
    # make video
    return init_latent_updated, img_updated


def mask_image(image,
               mask,
               color=[255, 0, 0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose.
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out


def get_points(img,
               sel_pix,
               evt: gr.SelectData):
    # collect the selected point
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1],
                            (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)


def undo_points(original_image,
                mask):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(original_image, 1 - mask,
                                color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = original_image.copy()
    return masked_img, []


def store_img_gen(img):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = np.array(image)
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask
