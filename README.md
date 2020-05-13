# pt-singan-single-image-gan

Inofficial implementation of the paper __"[SinGAN: Learning a Generative Model from a Single Natural Image](https://arxiv.org/pdf/1905.01164.pdf)"__ as a project for the _Deep Generative Models_ lecture at TU Darmstadt SS2020.

__Notes__
- Refactoring of code regarding the handling of the noise maps Z
- Dynamic RMSE-based adaption of standard deviation at the different scales
- Application examples: Single Image Animation by Random Walk, Paint2Image Translation, Image Super Resolution, Image Editing, Image Harmonization
- Spectral Normalization instead of Gradient Penalty? Other Loss Functions?
- Deformable Convolutions in Generator/Discriminator?


Example results of the current prototype are shown below:

__Single Training Images (512x512)__

Image 0             |  Image 1          | Image 2
:-------------------------:|:-------------------------:|:-------------------------:
![Training-Image-1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_fern/0_real.jpg) | ![Training-Image-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_4581/0_real.jpg) | ![Training-Image-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_36/0_real.jpg)

__Samples__

Sample 0             |  Sample 1          |  Sample 2   |  Sample 3
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Sample-1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_fern/0_0.jpg)  |  ![Sample-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_fern/0_1.jpg) | ![Sample-3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_fern/0_2.jpg) | ![Sample-4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_fern/0_3.jpg)

512x1024 Sample
:-------------------------:|
![Sample-Upscaled](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_fern/size_512x1024.jpg)


Sample 0             |  Sample 1          |  Sample 2   |  Sample 3
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Sample-1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_4581/0_0.jpg)  |  ![Sample-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_4581/0_1.jpg) | ![Sample-3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_4581/0_2.jpg) | ![Sample-4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_4581/0_3.jpg)

512x1024 Sample
:-------------------------:|
![Sample-Upscaled](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_4581/size_512x1024.jpg)

Sample 0             |  Sample 1          |  Sample 2   |  Sample 3
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Sample-1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_36/0_0.jpg)  |  ![Sample-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_36/0_1.jpg) | ![Sample-3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_36/0_2.jpg) | ![Sample-4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_36/0_3.jpg)

512x1024 Sample
:-------------------------:|
![Sample-Upscaled](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_36/size_512x1024.jpg)
