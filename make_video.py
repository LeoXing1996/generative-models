import cv2
import os
import os.path as osp

root = 'work_dirs/l23-lr005-lam005'
files = os.listdir(root)
file_list = []
img_list = []
for idx in range(0, 40):
    img_list.append(cv2.imread(osp.join(root, f'{idx}.png')))

height, width, layers = img_list[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('res.mp4', fourcc, 20, (width, height))

for image in img_list:
    video.write(image)

video.release()
