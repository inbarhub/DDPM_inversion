<!-- [![DDPM inversion](https://img.shields.io/badge/single%20image-generative%20model-yellow)](https://github.com/topics/single-image-generation) -->
[![Python 3.8](https://img.shields.io/badge/python-3.812+-blue)](https://www.python.org/downloads/release/python-38/)
[![torch](https://img.shields.io/badge/torch-2.0.0+-green)](https://pytorch.org/)


# DDPM_inversion

[Project](https://inbarhub.github.io/DDPM_inversion/) | [Arxiv](https://arxiv.org/abs/2304.06140) | [Supplementary materials](https://inbarhub.github.io/DDPM_inversion/resources/inversion_supp.pdf)
### Official pytorch implementation of the paper: <br>"An Edit Friendly DDPM Noise Space: Inversion and Manipulations"

![](imgs/teaser.jpg)
Our inversion can be used for text-based **editing of real images**, either by itself or in combination with other editing methods.
Due to the stochastic manner of our method, we can generate **diverse outputs**, a feature that is not naturally available with methods relying on the DDIM inversion.

In this repository we support editing using our inversion, prompt-to-prompt (p2p)+our inversion, ddim or [p2p](https://github.com/google/prompt-to-prompt) (with ddim inversion).<br>
**our inversion**: our ddpm inversion following by generating an image with the target prompt 
**prompt-to-prompt (p2p) +our inversion is**: p2p method using our ddpm inversion 
**ddim**: ddim inversion following by generating an image with the target prompt 
**pp2p**: p2p method using our ddim inversion (original paper)

## Table of Contents
* [Requirements](#Requirements)
* [Repository Structure](#Repository-Structure)
* [Usage Examples](#Usage-Examples)
* [Citation](#Citation)

## Requirements 

```
python -m pip install -r requirements.txt
```
This code was tested with python 3.8 and torch 2.0.0. 

## Repository Structure 
```
├── ddpm_inversion - folder contains inversions in order to work on real images: ddim inversion as well as ddom inversion (our method).
├── example_images - folder of input images to be edited
├── imgs - images used in this repository readme.md file
├── prompt_to_prompt - p2p code
├── main_run.py - main python file for real image editing
└── test.yaml - yaml file contains images and prompts to test on
```

A folder named 'results' will be automatically created and all the restuls will be saved to this folder. We also add a timestamp to the saved images in this folder.

## Algorithm Inputs and parameters
Method's inputs: 
```
img_name - the path to the input images
prompt_src - a prompt describing the input image
prompt_tar - the edit prompt
```
These three inputs can be given either as input arguments or via the test.yaml file.

<br>
Method's parametersare: 
```
skip - controlling the adherence to the input image
cfg_tar - the strength of the classifier free guidance
```

These two parameters have default values, as descibe in the paper.

## Usage Examples 
```
python3 main_run.py --mode="XXX" --img_name="example_images/horse_mud.jpg" --prompt_src"a photo of a horse in the mud" --prompt_tar="a photo of a horse in the snow"
or 
python3 main_run.py --mode="XXX"
```
Where XXX can be ```our_inv```,```p2pinv``` (p2p+our inversion),```ddim``` or ```p2p``` (original p2p paper).

In ```our_inv``` and ```p2pinv``` modes we suggest to play with ```skip``` in the range [0,40] and ```cfg_tar``` in the range [7,18].

**For p2pinv and p2p**:
Pay attention that you can play with the corss-and self-attention via ```--xa``` and ```--sa``` arguments. We suggest to set them to (0.6,0.2) and (0.8,0.4) for p2pinv and p2p respectively.

**For ddim and p2p**:
```skip``` is set to be 0.

### Create Your Own Editing with Our Method
(1) Add your image to example_images. <br>
(2) Run ``main_run.py --mode="our_inv"``, play with ``skip`` and ``cfg_tar``. <br>

Example:
```python3 main_run.py --skip=20 --cfg_tar=10 --img_name=gnochi_mirror --cfg_src='a cat is sitting next to a mirro --cfg_tar=a drwaing of a cat sitting next to a mirror'``` 

Instead, you can edit the test.yaml file to load your image and get your prompts.

<!-- ## Sources 

The DDPM code was adapted from the following [pytorch implementation of DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch). 

The modified CLIP model as well as most of the code in `./text2live_util/` directory was taken from the [official Text2live repository](https://github.com/omerbt/Text2LIVE).  -->
 
## Citation
If you use this code for your research, please cite our paper:
```
@article{HubermanSpiegelglas2023,
  title      = {An Edit Friendly DDPM Noise Space: Inversion and Manipulations},
  author     = {Huberman-Spiegelglas, Inbar and Kulikov, Vladimir and Michaeli, Tomer},
  journal    = {arXiv preprint arXiv:2304.06140},
  year       = {2023}
}
```