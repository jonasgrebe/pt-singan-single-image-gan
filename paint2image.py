import torch
import os
from imageio import imread, imwrite

from singan import SinGAN
from log import TensorboardLogger

import argparse

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description='SinGAN - Scale Injections')
parser.add_argument('--run_name', required=True)
parser.add_argument('--paint', required=True)
parser.add_argument('--target_height', type=int, default=None)
parser.add_argument('--target_width', type=int, default=None)

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

img_size = None
if args.not_pretrained:
    # load the single training image
    train_img_path = os.path.join('data', args.img)
    train_img = imread(train_img_path)

    # get the size of the img for later
    img_size = train_img.shape[:-1]

    # fit SinGAN to it
    singan.fit(img=train_img, steps_per_scale=args.steps_per_scale)
    # after training, save the model in a checkpoint
    singan.save_checkpoint()
else:
    # load the existing checkpoint if possible
    singan.load_checkpoint(logger.run_name)


# load the single training image
paint_img_path = os.path.join('data', 'paint', args.paint)
paint_img = imread(paint_img_path)

# get the size of the img for later
paint_size = paint_img.shape[:-1]

if img_size:
    assert img_size == paint_size

if args.target_height is None:
    target_size = paint_size
else:
    assert args.target_width is not None
    target_size = (args.target_height, args.target_width)

# scale injections
for scale in range(singan.N):
    x = singan.test(target_size=target_size, start_at_scale=scale, injection=paint_img)
    imwrite(f'samples/{logger.run_name}/paint_{args.paint.replace(".jpg", "")}_{scale}.jpg', x)
