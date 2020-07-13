from typing import Dict, Any, List, Tuple
import torch
import os
import numpy as np

from imageio import imwrite

from models import SingleScaleGenerator, Discriminator
from utils import freeze, gradient_penalty

from celery import current_task


class SinGAN:

    def __init__(self, N, logger, r=4 / 3, device=torch.device('cpu'), hypers: Dict[str, Any] = {}) -> None:
        # target depth of the SinGAN is (N+1)
        self.N = N
        # scaling factor, > 1
        self.r = r
        # torch device (either a gpu or per default the cpu)
        self.device = device

        # initialize both pyramids empty
        self.g_pyramid = []
        self.d_pyramid = []

        # auxiliary instance variables for paragraph below equation (5) in paper
        self.last_rec = None
        self.rmses = [1.0]

        # default hyperparameters
        self.hypers = {
            'g_lr': 5e-4,  # learning rate for generators
            'd_lr': 5e-4,  # learning rate for discriminators
            'n_blocks': 5,  # number of convblocks in each of the N generators (modules)
            'base_n_channels': 32,  # base number of filters for the coarsest module
            'min_n_channels': 32,  # minimum number of filters in any layer of any module
            'rec_loss_weight': 10.0,  # alpha weight for reconstruction loss
            'grad_penalty_weight': 0.1,  # lambda weight for gradient penalty loss
            'noise_weight': 0.1  # base standard deviation of gaussian noise
        }
        # overwrite them with given hyperparameters
        self.hypers.update(hypers)

        # define input and output (de-)normalization transformations
        self.transform_input = lambda x: (x - 127.5) / 127.5
        self.transform_output = lambda x: (x + 1) * 127.5

        # set the logger
        self.logger = logger
        self.train_img = None

        # for celery task tracking
        self.done_steps = 0
        self.total_steps = None

    def fit(self, img: np.ndarray, steps_per_scale: int = 2000) -> None:
        # initialize task tracking parameters
        self.total_steps = (self.N + 1) * steps_per_scale
        self.update_celery_state()

        # precompute all the sizes of the different scales
        target_size = img.shape[:-1]
        self.train_img = img

        # compute scales sizes
        scale_sizes = self.compute_scale_sizes(target_size)

        # print scales to validate choice of N
        print(scale_sizes)

        # preprocess input image and pack it in a batch
        img = torch.from_numpy(img.transpose(2, 0, 1))
        img = self.transform_input(img)
        img = img.expand(1, 3, target_size[0], target_size[1])

        # fix initial noise map for reconstruction loss computation
        self.z_init = self.generate_random_noise(scale_sizes[:1])[0]

        # training progression
        self.logger.set_scale(self.N + 1)
        for p in range(self.N + 1):
            self.logger.new_scale()

            # double number of initial channels every 4 scales
            n_channels = self.hypers['base_n_channels'] * (2 ** (p // 4))

            # instantiate new models for the next scale
            new_generator = SingleScaleGenerator(n_channels=n_channels,
                                                 min_channels=self.hypers['min_n_channels'],
                                                 n_blocks=self.hypers['n_blocks']).to(self.device)

            new_discriminator = Discriminator(n_channels=n_channels,
                                              min_channels=self.hypers['min_n_channels'],
                                              n_blocks=self.hypers['n_blocks']).to(self.device)

            # initialize weights via copy if possible
            if (p - 1) // 4 == p // 4:
                new_generator.load_state_dict(self.g_pyramid[0].state_dict())
                new_discriminator.load_state_dict(self.d_pyramid[0].state_dict())

            # reset the optimizers
            self.g_optimizer = torch.optim.Adam(new_generator.parameters(), lr=self.hypers['g_lr'], betas=[0.5, 0.999])
            self.d_optimizer = torch.optim.Adam(new_discriminator.parameters(), lr=self.hypers['d_lr'],
                                                betas=[0.5, 0.999])

            # insert new generator and discriminator at the bottom of the pyramids
            self.g_pyramid.insert(0, new_generator)
            self.d_pyramid.insert(0, new_discriminator)

            # fit the currently finest scale
            self.fit_single_scale(img=img, target_size=scale_sizes[p], steps=steps_per_scale)

            # freeze the weights after training
            freeze(self.g_pyramid[0])
            freeze(self.d_pyramid[0])

            # switch them to evaluation mode
            self.g_pyramid[0].eval()
            self.d_pyramid[0].eval()

    def fit_single_scale(self, img: np.ndarray, target_size: Tuple[int, int], steps: int) -> None:
        real = torch.nn.functional.interpolate(img, target_size, mode='bilinear', align_corners=True).float().to(
            self.device)

        # paragraph below equation (5) in paper
        if self.last_rec is not None:
            # compute root mean squared error between upsampled version of rec from last scale and the real target of the current scale
            rmse = torch.sqrt(torch.nn.functional.mse_loss(real,
                                                           torch.nn.functional.interpolate(self.last_rec, target_size,
                                                                                           mode='bilinear',
                                                                                           align_corners=True)))
            self.rmses.insert(0, rmse.detach().cpu().numpy().item())
        print(self.rmses)

        self.logger.set_mode('training')
        for step in range(1, steps + 1):
            self.logger.new_step()

            # ====== train discriminator =======================================
            self.d_pyramid[0].zero_grad()

            # generate a fake image
            fake = self.forward_g_pyramid(target_size=target_size)
            # let the discriminator judge the fake image patches (without any gradient flow through the generator )
            d_fake = self.d_pyramid[0](fake.detach())

            # loss for fake images
            adv_d_fake_loss = torch.mean(d_fake)
            adv_d_fake_loss.backward()

            # let the discriminator judge the real image patches
            d_real = self.d_pyramid[0](real)

            # loss for real images
            adv_d_real_loss = (-1) * torch.mean(d_real)
            adv_d_real_loss.backward()

            # gradient penalty loss
            grad_penalty = gradient_penalty(self.d_pyramid[0], real, fake, self.device) * self.hypers[
                'grad_penalty_weight']
            grad_penalty.backward()

            # make a step against the gradient
            self.d_optimizer.step()

            # ====== train generator ===========================================
            self.g_pyramid[0].zero_grad()

            # let the discriminator judge the fake image patches
            d_fake = self.d_pyramid[0](fake)

            # loss for fake images
            adv_g_loss = (-1) * torch.mean(d_fake)
            adv_g_loss.backward()

            # reconstruct original image with fixed z_init and else no noise
            rec = self.forward_g_pyramid(target_size=target_size, Z=[self.z_init] + [0] * (len(self.g_pyramid) - 1))

            # reconstruction loss
            rec_g_loss = torch.nn.functional.mse_loss(rec, real) * self.hypers['rec_loss_weight']
            rec_g_loss.backward()

            # make a step against the gradient
            self.g_optimizer.step()

            loss_dict = {
                'adv_d_fake_loss': adv_d_fake_loss,
                'adv_d_real_loss': adv_d_real_loss,
                'adv_g_loss': adv_g_loss,
                'rec_g_loss': rec_g_loss,
            }

            self.logger.log_losses(loss_dict)
            print(f'[{self.N - len(self.g_pyramid) + 1}: {step}|{steps}] -', loss_dict)

            # increment counter for task progress tracking
            self.done_steps += 1
            self.update_celery_state()

        # save current reconstruction of this scale temporally
        self.last_rec = rec

        self.logger.set_mode('sampling')
        for step in range(8):
            # sample from learned distribution
            fake = self.test(target_size=target_size)

            # log image
            self.logger.log_images(fake, name=str(step), dataformats='HWC')

            # save image traditionally to file
            run_dir = os.path.join('samples', self.logger.run_name)
            if not os.path.isdir(run_dir):
                os.makedirs(run_dir)
            imwrite(os.path.join(run_dir, f'{self.N - len(self.g_pyramid) + 1}_{step}.jpg'), fake)

        # save reconstruction of real image and downscaled copy of real image
        imwrite(os.path.join(run_dir, f'{self.N - len(self.g_pyramid) + 1}_rec.jpg'),
                self.transform_output(rec[0].detach().cpu().numpy().transpose(1, 2, 0)).astype('uint8'))
        imwrite(os.path.join(run_dir, f'{self.N - len(self.g_pyramid) + 1}_real.jpg'),
                self.transform_output(real[0].detach().cpu().numpy().transpose(1, 2, 0)).astype('uint8'))

    def test(self, target_size: Tuple[int, int], Z: List[torch.Tensor] = None, injection: np.ndarray = None,
             start_at_scale: int = None) -> torch.Tensor:
        # preprocess injection image and pack it in a batch
        if injection is not None:
            injection = torch.from_numpy(injection.transpose(2, 0, 1))
            injection = self.transform_input(injection)
            injection = injection.expand(1, 3, injection.shape[1], injection.shape[2])

        # generate fake image
        x = self.forward_g_pyramid(target_size=target_size, start_at_scale=start_at_scale, injection=injection)[0]
        # convert it to numpy
        x = x.detach().cpu().numpy().transpose(1, 2, 0)
        # denormalize its values and cast to integer
        x = self.transform_output(x).astype('uint8')
        return x

    def forward_g_pyramid(self, target_size: Tuple[int, int], start_at_scale: int = None, Z: List[torch.Tensor] = None,
                          injection: np.ndarray = None) -> torch.Tensor:

        # default starting scale to the coarsest scale
        if start_at_scale is None:
            start_at_scale = len(self.g_pyramid) - 1

        if Z is not None:
            assert len(Z) == start_at_scale + 1

        # compute all image sizes from the start scale upwards
        scale_sizes = self.compute_scale_sizes(target_size, start_at_scale)

        # if no special noise maps are specified, initialize them randomly
        Z_rand = self.generate_random_noise(scale_sizes)

        # if some input noise maps are specified, alter the randomly sampled maps accordingly
        if Z is not None:
            for i, z in enumerate(Z):
                # if a zero is given for the respective scale, then use zero noise
                if type(z) == int and z == 0:
                    Z_rand[i] *= z
                # else if nothing is specified, use the random noise
                elif z is not None:
                    Z_rand[i] = z
        Z = Z_rand

        # inject the image at the starting scale if necessary
        x = injection if start_at_scale != len(self.g_pyramid) - 1 else None

        # for each scale from the starting scale upwards
        for z, size, rmse, generator in zip(Z, scale_sizes, self.rmses[:start_at_scale + 1][::-1],
                                            self.g_pyramid[:start_at_scale + 1][::-1]):

            # make sure that z has the correct size
            z = torch.nn.functional.interpolate(z, size, mode='bilinear', align_corners=True)

            # scale noise by the specified noise weight
            z *= self.hypers['noise_weight']
            # scale noise by the rmse of that scale
            z *= rmse

            if x is not None:
                # upsample the output of the previous scale if necessary
                # (or make sure that the injected image has the correct size)
                x = torch.nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)
            else:
                # if no previous output is given, set it zero
                x = torch.zeros(size=(1, 3,) + size)

            # feed noise map and image through current generator to obtain new image
            x = generator(z.to(self.device), x.to(self.device)).clamp(min=-1, max=1)

        return x

    def generate_random_noise(self, sizes: List[Tuple[int, int]], type: str = 'gaussian'):
        if type == 'gaussian':
            return [torch.randn(size=(1, 3,) + size) for size in sizes]

    def compute_scale_sizes(self, target_size, start_at_scale=None):
        if start_at_scale is None:
            start_at_scale = self.N
        return [tuple((np.array(target_size) // self.r ** n).astype(int)) for n in range(start_at_scale, -1, -1)]

    def save_checkpoint(self) -> None:
        # create the checkpoint directory if it does not exist
        checkpoint_dir = os.path.join(self.logger.log_dir, 'checkpoints')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # create the checkpoint
        checkpoint = {
            'N': self.N,
            'r': self.r,
            'hypers': self.hypers,
            'g_pyramid': self.g_pyramid,
            'd_pyramid': self.d_pyramid,
            'z_init': self.z_init,
            'rmses': self.rmses,
            'train_img': self.train_img,
        }
        # save the checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'{self.logger.run_name}.ckpt'))

    def load_checkpoint(self, run_name: str) -> None:
        # load the checkpoint of SinGAN
        checkpoint_dir = os.path.join(self.logger.log_dir, 'checkpoints')
        checkpoint = torch.load(os.path.join(checkpoint_dir, f'{run_name}.ckpt'))

        # restore the information from the checkpoint
        self.N = checkpoint['N']
        self.r = checkpoint['r']
        self.hypers = checkpoint['hypers']
        self.g_pyramid = checkpoint['g_pyramid']
        self.d_pyramid = checkpoint['d_pyramid']
        self.z_init = checkpoint['z_init']
        self.rmses = checkpoint['rmses']

        if 'train_img' in checkpoint:
            self.train_img = checkpoint['train_img']

        # inform the logger about the restored epoch
        self.logger.set_scale(self.N - len(self.g_pyramid) + 1)


    def update_celery_state(self):
        if current_task:
            current_task.update_state(state='PROGRESS',
                                      meta={'current': self.done_steps,
                                            'total': self.total_steps,
                                            'percent': int((self.done_steps / self.total_steps) * 100)})


def load_pretrained(checkpoint_path) -> SinGAN:
    """Used to load a pretrained SinGAN. It cannot be used for training."""

    # load the checkpoint of SinGAN
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    singan = SinGAN(N=checkpoint['N'], logger=None, r=checkpoint['r'])

    # restore the information from the checkpoint
    singan.hypers = checkpoint['hypers']
    singan.g_pyramid = checkpoint['g_pyramid']
    singan.d_pyramid = checkpoint['d_pyramid']
    singan.z_init = checkpoint['z_init']
    singan.rmses = checkpoint['rmses']

    if 'train_img' in checkpoint:
        singan.train_img = checkpoint['train_img']

    return singan
