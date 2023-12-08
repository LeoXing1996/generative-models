import torch
import os
import os.path as osp
from mmengine import Config
from argparse import ArgumentParser
from pipeline import TurboPipeline


parser = ArgumentParser()
parser.add_argument('--src', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--lam', type=float)
parser.add_argument('--save-dir', type=str, default='work_dirs/tmp')
parser.add_argument('--layer-idxs', nargs='+', type=int, default=[3])
args = parser.parse_args()


cfg = Config.fromfile(osp.join(args.src, 'cfg.py'))
input_dict = torch.load(osp.join(args.src, 'inp_dict.pt'))

mask = input_dict['mask']
init_latent = input_dict['init_latent']
handle_points = input_dict['handle_points']
target_points = input_dict['target_points']

pipeline = TurboPipeline.build()
pipeline.convert_to_drag()

os.makedirs(args.save_dir, exist_ok=True)

lr = cfg.lr if args.lr is None else args.lr
lam = cfg.lam if args.lam is None else args.lam

latent, img_updated = pipeline.drag(
    init_latent,
    handle_points,
    target_points,
    cfg.prompt,
    mask,
    lr,
    40,
    args.layer_idxs,
    cfg.r_m,
    cfg.r_p,
    lam,
    cfg.sup_res_h,
    cfg.sup_res_w,
    save_dir=args.save_dir,

)

img_updated.save(f'{args.save_dir}/img_updated.png')
