import torch
import os
from imageio import imread, imwrite
import cv2

from singan import SinGAN
from log import TensorboardLogger

import argparse

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description='SinGAN - Super Resolution')
parser.add_argument('--run_name', required=True)
parser.add_argument('--frames', type=int, required=True)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--start_at_scale', type=int, default=None)
parser.add_argument('--fps', type=int, default=30)

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

target_size = train_img.shape[:-1]
img = train_img

scale_sizes = singan.compute_scale_sizes(target_size)
scale_noise_weights = singan.rmses[::-1]

# starting point of random walk is z_rec
Z_rec = [singan.z_init]+[0]*(singan.N-1)
Z_list = [Z_rec]

# initial walk
Z = Z_rec.copy()
Z_rand = singan.generate_random_noise(scale_sizes)
for n in range(len(Z)):
    Z[n] = Z_rec[n] * args.alpha + (1- args.alpha) * Z_rand[n]
    Z[n] *= singan.hypers["noise_weight"]
    Z[n] *= scale_noise_weights[n]
Z_list.append(Z)

if args.start_at_scale is None:
    start_at_scale = singan.N-1
else:
    start_at_scale = args.start_at_scale

# create random walk trajectory
for t in range(args.frames):

    Z_rand = singan.generate_random_noise(scale_sizes)
    for n in range(len(Z)):
        z_diff = args.beta * (Z_list[-1][n] - Z_list[-2][n]) + (1-args.beta) * Z_rand[n]
        Z[n] = args.alpha * Z_rec[n] + (1-args.alpha) * (Z_list[-1][n] + z_diff)
        Z[n] *= singan.hypers["noise_weight"]
        Z[n] *= scale_noise_weights[n]
    Z_list.append(Z)


# generate frames and save them as a video
video = cv2.VideoWriter(f"samples/{logger.run_name}/video.mp4", -1, args.fps, (target_size[1], target_size[0]))
for t, Z in enumerate(Z_list):
    frame = singan.test(target_size=target_size, Z=Z[-(start_at_scale+1):], injection=img, start_at_scale=args.start_at_scale)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video.write(frame)

video.release()
