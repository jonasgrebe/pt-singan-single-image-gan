from celery import shared_task, current_task
import time
from imageio import imread, imwrite
import sys

from django.templatetags.static import static

# Don't change the following order
sys.path.append("..")
import singan as sg
from log import TensorboardLogger

singan = None


@shared_task
def load_singan(singan_name):
    print("Loading pretrained SinGAN...")

    logger = TensorboardLogger(f'singan_web')
    singan = sg.SinGAN(1, logger, 1.5)

    singan.load_checkpoint(run_name=singan_name)

    return "Successfully loaded pretrained SinGAN"


@shared_task
def train_singan(image_path, N, r, steps_per_scale):
    print("Starting SinGAN training...")
    logger = TensorboardLogger(f'singan_web')
    singan = sg.SinGAN(N, logger, r)

    image = imread(image_path)
    singan.fit(img=image, steps_per_scale=steps_per_scale)
    singan.save_checkpoint()

    return "Training finished successfully!"


@shared_task
def generate_random_image(insert_stage_id, res_x, res_y):
    print("Starting random image generation...")

    if not isinstance(singan, sg.SinGAN):
        return "SinGAN not initialized"

    image = singan.test((res_x, res_y), start_at_scale=insert_stage_id)

    image_path = 'media/out/result.jpg'
    imwrite(image_path, image)
    image_url = static(image_path)

    return image_url
