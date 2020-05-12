import torch
import os
import cv2
import numpy as np
from imageio import imread, imwrite

from singan import SinGAN
from log import TensorboardLogger

# get the available device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# load the single training image
train_img_path = os.path.join('data', '36-white-and-black-boats-on-body-of-water.jpg')
train_img = imread(train_img_path)

# instantiate the logger and the SinGAN
logger = TensorboardLogger('singan_36')
singan = SinGAN(N=11, logger=logger, device=device)

# get the size of the img for later
img_size = train_img.shape[:-1]

# fit SinGAN to it
singan.fit(img=train_img, steps_per_scale=3000)

# after training, save the model in a checkpoint
singan.save_checkpoint()

#singan.load_checkpoint(logger.run_name)

# scale injections
for scale in range(singan.N):
    x = singan.test(target_size=img_size, start_at_scale=scale, injection=train_img)
    imwrite(f'samples/{logger.run_name}/inj_{scale}.jpg', x)

# random sampling for different sizes
for size in [(256, 256), (512, 256), (512, 1024), (64, 64)]:
    x = singan.test(target_size=size)
    imwrite(f'samples/{logger.run_name}/size_{size[0]}x{size[1]}.jpg', x)
