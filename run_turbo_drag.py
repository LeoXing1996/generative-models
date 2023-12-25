import torch
import os
import os.path as osp
from mmengine import Config
from argparse import ArgumentParser
from pipeline import TurboPipeline


def list_srcs(data_dir):
    return [osp.join(data_dir, f) for f in os.listdir(data_dir)]


parser = ArgumentParser()
parser.add_argument('--data-dir', type=str, default='work_dirs/data')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--lam', type=float, default=0.02)
parser.add_argument('--save-dir-name', type=str, default='res')
parser.add_argument('--layer-idxs', nargs='+', type=int, default=[2, 3])
args = parser.parse_args()

pipeline = TurboPipeline.build()
pipeline.convert_to_drag()


data_dirs = list_srcs(args.data_dir)
# data_dirs = [data_dirs[1]]

for src in data_dirs:
    cfg = Config.fromfile(osp.join(src, 'cfg.py'))
    input_dict = torch.load(osp.join(src, 'inp_dict.pt'))

    save_dir = osp.join(src, args.save_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    mask = input_dict['mask']
    init_latent = input_dict['init_latent']
    handle_points = input_dict['handle_points']
    target_points = input_dict['target_points']

    os.makedirs(args.save_dir_name, exist_ok=True)

    lr = cfg.lr if args.lr is None else args.lr
    lam = cfg.lam if args.lam is None else args.lam

    try:
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
            save_dir=save_dir,
        )

        img_updated.save(f'{save_dir}/img_updated.png')
    except Exception as e:
        print(src, e)
        continue
