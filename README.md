# pt-singan-single-image-gan

Inofficial implementation of the paper __"[SinGAN: Learning a Generative Model from a Single Natural Image](https://arxiv.org/pdf/1905.01164.pdf)"__ as a project for the _Deep Generative Models_ lecture at TU Darmstadt SS2020.

## Note
- Application examples: Single Image Animation by Random Walk, Paint2Image Translation, Image Super Resolution, Image Editing, Image Harmonization
- Inner Steps in the Official Implementation?
- Spectral Normalization instead of Gradient Penalty? Other Loss Functions?
- Deformable Convolutions in Generator/Discriminator?


## How to use

For each of the exemplary SinGAN applications, we created an easy-to-use python script that can be run directly from the console by specifying the necessary parameters. All of these scripts have in common that they require either just the run_name of a pretrained SinGAN model or the --not_pretrained flag together with the number of scales N and the number of steps per scale. For instance, the following additional command line arguments would train a SinGAN model with 8 scales and 2000 steps per scale:

```console
python application.py [...] --not_pretrained --N 8 --steps_per_scale 2000
```

If the not_pretrained flag is not given but a trained model with the identifier run_name exists, this is used instead.

### Random Sampling

```console
python sample.py --run_name <String> --img 5026-green-fern-plant-during-daytime.jpg -- height <int> --width <int>
```

### Scale Injection

```console
python scale_injections.py --run_name <String> --img 5026-green-fern-plant-during-daytime.jpg
```

### Super Resolution

```console
python scale_injections.py --run_name <String> --img 5026-green-fern-plant-during-daytime.jpg --super_scales <int>
```

### Paint2Image
_Note_: You have to additionally provide a training image via --img if you want to train a new model. The paint images are expected to be found in the data/paint subdirectory.

```console
python paint2image.py --run_name <String> --paint 5026_1.jpg
```

## Example results

### Single training images (512x512 px)

Image 0             |  Image 1          | Image 2    | Image 3
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Training-Image-1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/5026-green-fern-plant-during-daytime.jpg) | ![Training-Image-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/473-brown-rock-wall.jpg)  | ![Training-Image-3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/856-zebra-in-savanna.jpg) | ![Training-Image-4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/220-pile-of-books.jpg)


### Random samples

Sample 0             |  Sample 1          |  Sample 2   |  Sample 3
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Sample-1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/0_0.jpg)  |  ![Sample-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/0_1.jpg) | ![Sample-3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/0_2.jpg) | ![Sample-4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/0_3.jpg)

Sample 4
:-------------------------:
![Sample-5](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/size_512x2048.jpg)


Sample 0             |  Sample 1          |  Sample 2   |  Sample 3
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Sample-1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/0_0.jpg)  |  ![Sample-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/0_1.jpg) | ![Sample-3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/0_2.jpg) | ![Sample-4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/0_3.jpg)

Sample 4
:-------------------------:
![Sample-5](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/size_512x2048.jpg)

## Scale Injections

Scale 0           |  Scale 1       |  Scale 2   |   Scale 3
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Inj-1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_0.jpg)  |  ![Inj-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_1.jpg) | ![Inj-3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_2.jpg) | ![Inj-4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_3.jpg)


Scale 4           |  Scale 5       |  Scale 7   |   Scale 9
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Inj-5](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_4.jpg)  |  ![Inj-6](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_5.jpg) | ![Inj-7](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_7.jpg) | ![Inj-8](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/inj_9.jpg)


Scale 0           |  Scale 1       |  Scale 2   |   Scale 3
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Inj-1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/inj_0.jpg)  |  ![Inj-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/inj_1.jpg) | ![Inj-3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/inj_2.jpg) | ![Inj-4](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/inj_3.jpg)


Scale 4           |  Scale 5       |  Scale 7   |   Scale 9
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Inj-5](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/inj_4.jpg)  |  ![Inj-6](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/inj_5.jpg) | ![Inj-7](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/inj_7.jpg) | ![Inj-8](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/inj_9.jpg)


### Super Resolution (by Factor r)

r<sup>0</sup>        |  r<sup>1</sup>         |  r<sup>2</sup>     |   r<sup>3</sup>  
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![SR-0](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/img_sr_0r.jpg)  |  ![SR-1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/img_sr_1r.jpg) | ![SR-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/img_sr_2r.jpg) | ![SR-3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/img_sr_3r.jpg)

r<sup>0</sup>        |  r<sup>1</sup>         |  r<sup>2</sup>     |   r<sup>3</sup>  
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![SR-0](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/img_sr_0r.jpg)  |  ![SR-1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/img_sr_1r.jpg) | ![SR-2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/img_sr_2r.jpg) | ![SR-3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/img_sr_3r.jpg)


### Paint2Image
Train     |  Paint         |  Scale 8     |   Scale 9
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![p2i_0](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/5026-green-fern-plant-during-daytime.jpg)   |  ![p2i_1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/paint/5026_0.jpg) | ![p2i_2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/paint_5026_0_8.jpg) | ![p2i_3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/paint_5026_0_9.jpg)

Train     |  Paint         |  Scale 8     |   Scale 9
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![p2i_0](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/5026-green-fern-plant-during-daytime.jpg)   |  ![p2i_1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/paint/5026_1.jpg) | ![p2i_2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/paint_5026_1_8.jpg) | ![p2i_3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_5026/paint_5026_1_9.jpg)


Train     |  Paint         |  Scale 7     |   Scale 9
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![p2i_0](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/473-brown-rock-wall.jpg)   |  ![p2i_1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/paint/473_0.jpg) | ![p2i_2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/paint_473_0_7.jpg) | ![p2i_3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/paint_473_0_9.jpg)

Train     |  Paint         |  Scale 6     |   Scale 9
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![p2i_0](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/473-brown-rock-wall.jpg)   |  ![p2i_1](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/data/paint/473_1.jpg) | ![p2i_2](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/paint_473_1_6.jpg) | ![p2i_3](https://github.com/jonasgrebe/pt-singan-single-image-gan/blob/master/samples/singan_473/paint_473_1_9.jpg)



