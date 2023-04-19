<!-- [![DDPM inversion](https://img.shields.io/badge/single%20image-generative%20model-yellow)](https://github.com/topics/single-image-generation) -->
[![Python 3.8](https://img.shields.io/badge/python-3.812+-blue)](https://www.python.org/downloads/release/python-38/)
[![torch](https://img.shields.io/badge/torch-1.13.0+-green)](https://pytorch.org/)


# DDPM_inversion

[Project](https://inbarhub.github.io/DDPM_inversion/) | [Arxiv](https://arxiv.org/abs/2304.06140) | [Supplementary materials](https://inbarhub.github.io/DDPM_inversion/resources/inversion_supp.pdf)
### Official pytorch implementation of the paper: <br>"An Edit Friendly DDPM Noise Space: Inversion and Manipulations"

![](img/teaser.jpg)
Our inversion can be used for text-based editing of real images, either by itself or in combination with other editing methods.
Due to the stochastic manner of our method, we can generate diverse outputs, a feature that is not naturally available with methods relying on the DDIM inversion.

## SinDDM's Applications
SinDDM can also be used for a line of image manipulation tasks, especially image manipluations guided by text, for example:
 ![](imgs/text_guidance.png)


See section 4 in our [paper](https://arxiv.org/pdf/2211.16582.pdf) for more details about our results and experiments.


### Citation
If you use this code for your research, please cite our paper:

```
@article{kulikov2022sinddm,
  title      = {SinDDM: A Single Image Denoising Diffusion Model},
  author     = {Kulikov, Vladimir and Yadin, Shahar and Kleiner, Matan and Michaeli, Tomer},
  journal    = {arXiv preprint arXiv:2211.16582},
  year       = {2022}
}
```

## Table of Contents
* [Random Samples from a Single Example](#Random-Samples-from-a-Single-Example)
* [SinDDM's Applications](#SinDDMs-Applications)
* [Citation](#Citation)
* [Requirements](#Requirements)
* [Repository Structure](#Repository-Structure)
* [Usage Examples](#Usage-Examples)
* [Data and Pretrained Models](#Data-and-Pretrained-Models)
* [Sources](#Sources)


## Requirements 

```
python -m pip install -r requirements.txt
```
This code was tested with python 3.8 and torch 1.13. 

## Repository Structure 
```
├── SinDDM - training and inference code   
├── clip - clip model code
├── datasets - the images used in the paper
├── imgs - images used in this repository readme.md file
├── results - pre-trained models 
├── text2live_util - code for editing via text, based on text2live code 
└── main.py - main python file for initiate model training and for model inference 
```

## Usage Examples 
**Note: This is an early code release which provides full functionality, but is not yet fully organized or optimized. We will be extensively updating this repository in the coming weeks.** 

###  Train
To train a SinDDM model on your own image e.g. `<training_image.png>`, put the desired training image under `./datasets/<training_image>/`, and run

```
python main.py --scope <training_image> --mode train --dataset_folder ./datasets/<training_image>/ --image_name <training_image.png> --results_folder ./results/ 
```

This code will also generate random samples starting from the coarsest scale (s=0) of the trained model.

###  Random sampling
To generate random samples, please first train a SinDDM model on the desired image (as described above) or use a provided pretrained model, then run 

```
python main.py --scope <training_image> --mode sample --dataset_folder ./datasets/<training_image>/ --image_name <training_image.png> --results_folder ./results/ --load_milestone 12
```
To sample images in arbitrary sizes, one can add ```--scale_mul <y> <x>``` argument to generate an image that is `<y>` times as high and `<x>` times as wide as the original image.
 
<!-- ###  Random samples of arbitrary sizes 
To generate random samples of arbitrary sizes, use the '--scale_mul h w' argument.
For example, to generate an image with the width dimension 2 times larger run
```
python main.py --scope <training_image> --mode sample --dataset_folder ./datasets/<training_image>/ --image_name <training_image.png> --results_folder ./results/ --load_milestone 12 --scale_mul 1 2
``` -->

###  Text guided content generation

To guide the generation to create new content using a given text prompt `<text_prompt>`, run 

```
python main.py --scope <training_image> --mode clip_content --clip_text <text_prompt> --strength <s> --fill_factor <f> --dataset_folder ./datasets/<training_image>/ --image_name <training_image.png> --results_folder ./results/ --load_milestone 12
```
Where **_strength_** and **_fill_factor_** are the required controllable parameters explained in the paper.

###  Text guided style generation

To guide the generation to create a new style for the image using a given text prompt `<style_prompt>`, run

```
python main.py --scope <training_image> --mode clip_style_gen --clip_text <style_prompt> --dataset_folder ./datasets/<training_image>/ --image_name <training_image.png> --results_folder ./results/ --load_milestone 12
```
**Note:** One can add the ```--scale_mul <y> <x>``` argument to generate an arbitrary size sample with the given style.

###  Text guided style transfer

To create a new style for a given image, without changing the original image global structure, run

```
python main.py --scope <training_image> --mode clip_style_trans --clip_text <text_style> --dataset_folder ./datasets/<training_image>/ --image_name <training_image.png> --results_folder ./results/ --load_milestone 12
```

###  Text guided ROI
To modify an image in a specified ROI (Region Of Interest) with a given text prompt `<text_prompt>`, run

```
python main.py --scope <training_image> --mode clip_roi --clip_text <text_prompt> --strength <s> --fill_factor <f> --dataset_folder ./datasets/<training_image>/ --image_name <training_image.png> --results_folder ./results/ --load_milestone 12
```
**Note:** A Graphical prompt will open. The user need to select a ROI within the displayed image.

###  ROI guided generation

Here, the user can mark a specific training image ROI and choose where it should appear in the generated samples. If roi_n_tar is passed then the user will be able to choose several target locations.
```
python main.py --scope <training_image> --mode roi --roi_n_tar <n_targets> --dataset_folder ./datasets/<training_image>/ --image_name <training_image.png> --results_folder ./results/ --load_milestone 12
```
A graphical prompt will open and allow the user to choose a ROI from the training image. Then, the user need to choose where it should appear in the resulting samples.
Here as well, one can generate an image with arbitrary shapes using ```--scale_mul <y> <x>```

###  Harmonization

To harmonize a pasted object into an image, place a naively pasted reference image and the selected mask into `./datasets/<training_image>/i2i/` and run

```
python main.py --scope <training_image> --mode harmonization --harm_mask <mask_name> --input_image <naively_pasted_image> --dataset_folder ./datasets/<training_image>/ --image_name <training_image.png> --results_folder ./results/ --load_milestone 12
```

###  Style Transfer

To transfer the style of the training image to a content image, place the content image into `./datasets/<training_image>/i2i/` and run

```
python main.py --scope <training_image> --mode style_transfer --input_image <content_image> --dataset_folder ./datasets/<training_image>/ --image_name <training_image.png> --results_folder ./results/ --load_milestone 12
```

## Data and Pretrained Models
We provide several pre-trained models for you to use under `./results/` directory. More models will be available soon.
 
We provide all the training images we used in our paper under the `./datasets/` directory. All the images we provide are in the dimensions we used for training and are in .png format. 
 
## Sources 

The DDPM code was adapted from the following [pytorch implementation of DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch). 

The modified CLIP model as well as most of the code in `./text2live_util/` directory was taken from the [official Text2live repository](https://github.com/omerbt/Text2LIVE). 
 