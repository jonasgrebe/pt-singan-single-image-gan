from celery import shared_task, current_task
import time
from imageio import imread, imwrite
import sys

from .models import ImageModel
from django.conf import settings
from os import listdir
from os.path import isfile, join

# Don't change the following order
sys.path.append("..")
import singan as sg
from log import TensorboardLogger

# Variables only accessible by Celery
pretrained_models = []
singan = None


def get_images_of_pretrained_models():
    load_pretrained_models()
    return pretrained_models[1]


def load_pretrained_models():
    global pretrained_models

    if len(pretrained_models) == 0:
        checkpoint_dir = "../logs/checkpoints/"
        checkpoint_paths = [checkpoint_dir + f for f in listdir(checkpoint_dir) if isfile(join(checkpoint_dir, f))]

        image_urls = []
        for i, checkpoint_path in enumerate(checkpoint_paths):
            singan = sg.load_pretrained(checkpoint_path)
            if singan.train_img is not None:
                image = singan.train_img
                image_path = 'media/pretrained/%d.jpg' % i
                imwrite(image_path, image)
                image_url = image_path
                image_urls += [image_url]
            else:
                image_urls *= [None]

        pretrained_models = [checkpoint_paths, image_urls]


def get_singan_info(model_id):
    if len(pretrained_models) == 0:
        raise AssertionError("No pretrained models")
    checkpoint_path = pretrained_models[0][model_id]
    singan_temp = sg.load_pretrained(checkpoint_path)
    singan_info = {'N': singan_temp.N, 'r': singan_temp.r, 'train_image_path': pretrained_models[1][model_id]}
    return singan_info


@shared_task
def load_singan(model_id):
    print("Loading pretrained SinGAN...")
    global singan, pretrained_models
    load_pretrained_models()

    if len(pretrained_models) == 0:
        return "No pretrained models"

    print("There are %d pretrained models. Loading model no. %d." % (len(pretrained_models[0]), model_id))

    checkpoint_path = pretrained_models[0][model_id]

    singan = sg.load_pretrained(checkpoint_path)

    return "Successfully loaded pretrained SinGAN"


@shared_task
def train_singan(image_path, N, r, steps_per_scale):
    print("Starting SinGAN training...")
    global singan

    logger = TensorboardLogger(f'singan_web')
    singan = sg.SinGAN(N, logger, r)

    image = imread(image_path)
    singan.fit(img=image, steps_per_scale=steps_per_scale)
    singan.save_checkpoint()

    return "Training finished successfully!"


@shared_task
def generate_random_image(res_x, res_y):
    if not isinstance(singan, sg.SinGAN):
        return "SinGAN not initialized"

    image = singan.test((res_y, res_x))

    image_path = 'media/out/result.jpg'
    imwrite(image_path, image)

    return image_path


@shared_task
def generate_super_res(image_path, upscale_factor):
    if not isinstance(singan, sg.SinGAN):
        return "SinGAN not initialized"

    image = imread(image_path)

    target_width = image.shape[0]
    target_height = image.shape[1]

    target_size = (int(target_width * singan.r ** upscale_factor),
                   int(target_height * singan.r ** upscale_factor))
    image_result = singan.test(target_size=target_size, injection=image, start_at_scale=0)

    result_path = 'media/out/result.jpg'
    imwrite(result_path, image_result)

    return result_path


@shared_task
def generate_paint2image(image_path, start_at_scale):
    return generate_injection(image_path, start_at_scale)


@shared_task
def generate_harmonization(image_path):
    if not isinstance(singan, sg.SinGAN):
        return "SinGAN not initialized"

    start_at_scale = min(4, singan.N - 1)  # TODO: Test this
    return generate_injection(image_path, start_at_scale)


@shared_task
def generate_injection(image_path, start_at_scale):
    if not isinstance(singan, sg.SinGAN):
        return "SinGAN not initialized"

    image = imread(image_path)

    target_width = image.shape[0]
    target_height = image.shape[1]

    target_size = (int(target_width), int(target_height))

    image_result = singan.test(target_size=target_size, injection=image, start_at_scale=start_at_scale)

    result_path = 'media/out/result.jpg'
    imwrite(result_path, image_result)

    return result_path
