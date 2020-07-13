import torch
import os
from imageio import imread, imwrite

from singan import SinGAN
from log import TensorboardLogger

import argparse

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description='SinGAN - Scale Injections')
parser.add_argument('--run_name', required=True)

parser.add_argument('--not_pretrained', action='store_true')
parser.add_argument('--img')
parser.add_argument('--N', type=int, default=0)
parser.add_argument('--steps_per_scale', type=int, default=2000)

args = parser.parse_args()

# get the available device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# instantiate the logger and the SinGAN
logger = TensorboardLogger(f'singan_{args.run_name}')
singan = SinGAN(N=args.N, logger=logger, device=device)

if args.not_pretrained:
    # load the single training image
    train_img_path = os.path.join('data', args.img)
    train_img = imread(train_img_path)
    # fit SinGAN to it
    singan.fit(img=train_img, steps_per_scale=args.steps_per_scale)
    # after training, save the model in a checkpoint
    singan.save_checkpoint()
else:
    # load the existing checkpoint if possible
    singan.load_checkpoint(logger.run_name)
    train_img = singan.train_img

# get the size of the img for later
img_size = train_img.shape[:-1]

# scale injections
for scale in range(singan.N):
    x = singan.test(target_size=img_size, start_at_scale=scale, injection=train_img)
    imwrite(f'samples/{logger.run_name}/inj_{scale}.jpg', x)
