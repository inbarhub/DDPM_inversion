import PIL
from PIL import Image, ImageDraw ,ImageFont
from matplotlib import pyplot as plt
import torchvision.transforms as T
import os
import torch 
import yaml

def show_torch_img(img):
    img = to_np_image(img)
    plt.imshow(img)
    plt.axis("off")

def to_np_image(all_images):
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
    return all_images

def tensor_to_pil(tensor_imgs):
    if type(tensor_imgs) == list:
        tensor_imgs = torch.cat(tensor_imgs)
    tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
    to_pil = T.ToPILImage()
    pil_imgs = [to_pil(img) for img in tensor_imgs]    
    return pil_imgs

def pil_to_tensor(pil_imgs):
    to_torch = T.ToTensor()
    if type(pil_imgs) == PIL.Image.Image:
        tensor_imgs = to_torch(pil_imgs).unsqueeze(0)*2-1
    elif type(pil_imgs) == list:    
        tensor_imgs = torch.cat([to_torch(pil_imgs).unsqueeze(0)*2-1 for img in pil_imgs]).to(device)
    else:
        raise Exception("Input need to be PIL.Image or list of PIL.Image")
    return tensor_imgs



## TODO implement this
# n = 10
# num_rows = 4
# num_col = n // num_rows
# num_col  = num_col + 1 if n % num_rows else num_col
# num_col
def add_margin(pil_img, top = 0, right = 0, bottom = 0, 
                    left = 0, color = (255,255,255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    
    result.paste(pil_img, (left, top))
    return result

def image_grid(imgs, rows = 1, cols = None, 
                    size = None,
                   titles = None, text_pos = (0, 0)):
    if type(imgs) == list and type(imgs[0]) == torch.Tensor:
        imgs = torch.cat(imgs)
    if type(imgs) == torch.Tensor:
        imgs = tensor_to_pil(imgs)
        
    if not size is None:
        imgs = [img.resize((size,size)) for img in imgs]
    if cols is None:
        cols = len(imgs)
    assert len(imgs) >= rows*cols
    
    top=20
    w, h = imgs[0].size
    delta = 0
    if len(imgs)> 1 and not imgs[1].size[1] == h:
        delta = top
        h = imgs[1].size[1]
    if not titles is  None:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 
                                    size = 20, encoding="unic")
        h = top + h 
    grid = Image.new('RGB', size=(cols*w, rows*h+delta))    
    for i, img in enumerate(imgs):
        
        if not titles is  None:
            img = add_margin(img, top = top, bottom = 0,left=0)
            draw = ImageDraw.Draw(img)
            draw.text(text_pos, titles[i],(0,0,0), 
            font = font)
        if not delta == 0 and i > 0:
           grid.paste(img, box=(i%cols*w, i//cols*h+delta))
        else:
            grid.paste(img, box=(i%cols*w, i//cols*h))
        
    return grid    


"""
input_folder - dataset folder
"""
def load_dataset(input_folder):
    # full_file_names = glob.glob(input_folder)
    # class_names = [x[0] for x in os.walk(input_folder)]
    class_names = next(os.walk(input_folder))[1]
    class_names[:] = [d for d in class_names if not d[0] == '.']
    file_names=[]
    for class_name in class_names:
        cur_path = os.path.join(input_folder, class_name)
        filenames = next(os.walk(cur_path), (None, None, []))[2]
        filenames = [f for f in filenames if not f[0] == '.']
        file_names.append(filenames)
    return class_names, file_names


def create_prompts_from_class(class_name):
    enc_prompt = f'a photo of a {class_name}'
    if class_name=='penguin':
        dec_prompts = [f'an embroidery of a {class_name}',
                    f'a video-game of a {class_name}',
                    f'a tattoo of a {class_name}',
                    f'a photo of a bustard',
                    f'a pattern of a crane',]
        
    if class_name=='husky':
        dec_prompts = ['a pattern of a husky',
                    'a tattoo of a husky',
                    'a toy of a husky',
                    'a graffiti of a poodle',
                    'an embroidery of a cat',]
        
    if class_name=='goldfish':
        dec_prompts = ['a graffiti of a goldfish',
                    'an origami of a goldfish',
                    'a photo of a goldfish',
                    'a tattoo of a parrotfish',
                    'an embroidery of a tuna',]
        
    if class_name=='cat':
        dec_prompts = ['a pattern of a cat',
                    'a graffiti of a cat',
                    'a toy of a cat',
                    'an origami of a poodle',
                    'an embroidery of a bear',]   
             
    if class_name=='jeep':
        dec_prompts = ['a graffiti of a jeep',
                    'an image of a jeep',
                    'a tattoo of a jeep',
                    'a cartoon of a pickup',
                    'a deviantart of a tractor',]     

    if class_name=='castle':
        dec_prompts = ['an image of a castle',
                    'a sculpture of a castle',
                    'a tattoo of a castle',
                    'an embroidery of a temple',
                    'a photo of a church',]  

    if class_name=='pizza':
        dec_prompts = ['a tattoo of a pizza',
                    'a cartoon of a pizza',
                    'a graffiti of a pizza',
                    'an image of a baloon',
                    'an embroidery of a cake',]  

    if class_name=='violin':
        dec_prompts = ['a video-game of a violin',
                    'a sculpture of a violin',
                    'a toy of a violin',
                    'an embroidery of a viola',
                    'a pattern of a cello',]  

    if class_name=='panda':
        dec_prompts = ['a toy of a panda',
                    'a tattoo of a panda',
                    'an origami of a panda',
                    'an embroidery of a bear',
                    'a graffiti of a leopard',]  

    if class_name=='hummingbird':
        dec_prompts = ['an art of a hummingbird',
                    'a video-game of a hummingbird',
                    'a toy of a hummingbird',
                    'a deviantart of a pigeon',
                    'a graffiti of an egret',] 

    return enc_prompt, dec_prompts


def dataset_from_yaml(yaml_location):
    with open(yaml_location, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded