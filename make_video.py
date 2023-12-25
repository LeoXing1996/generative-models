import os
import os.path as osp
from tqdm import tqdm

import click
import cv2


def list_srcs(data_dir):
    return [osp.join(data_dir, f) for f in os.listdir(data_dir)]


@click.command()
@click.option('--root', type=str, default='./work_dirs/data')
@click.option('--save-dir-name', type=str, default='res')
@click.option('--n-frames', type=int, default=40)
@click.option('--video-name', type=str, default='res.mp4')
def main(root: str, save_dir_name: str, n_frames: int, video_name: str):
    data_dirs = list_srcs(root)
    for src in tqdm(data_dirs):
        res_dir = osp.join(src, save_dir_name)
        video_dir = osp.join(src, save_dir_name, video_name)
        img_list = []
        for idx in range(0, n_frames):
            img_list.append(cv2.imread(osp.join(res_dir, f'{idx}.png')))
        height, width, layers = img_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_dir, fourcc, 20, (width, height))
        for image in img_list:
            video.write(image)
        video.release()


# root = 'work_dirs/l23-lr005-lam005'
# files = os.listdir(root)
# file_list = []
# img_list = []
# for idx in range(0, 40):
#     img_list.append(cv2.imread(osp.join(root, f'{idx}.png')))

# height, width, layers = img_list[0].shape
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter('res.mp4', fourcc, 20, (width, height))

# for image in img_list:
#     video.write(image)

# video.release()

if __name__ == '__main__':
    main()
