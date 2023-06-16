import argparse
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import os
from prompt_to_prompt.ptp_classes import AttentionStore, AttentionReplace, AttentionRefine, EmptyControl,load_512
from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable, view_images
from ddm_inversion.inversion_utils import  inversion_forward_process, inversion_reverse_process
from ddm_inversion.utils import image_grid,dataset_from_yaml

from torch import autocast, inference_mode
from ddm_inversion.ddim_inversion import ddim_inversion

import calendar
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=3.5)
    parser.add_argument("--cfg_tar", type=float, default=15)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--dataset_yaml",  default="test.yaml")
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--mode",  default="p2pinv", help="modes: our_inv,p2pinv,p2pddim,ddim")
    parser.add_argument("--skip",  type=int, default=36)
    parser.add_argument("--xa", type=float, default=0.6)
    parser.add_argument("--sa", type=float, default=0.2)
    
    args = parser.parse_args()
    full_data = dataset_from_yaml(args.dataset_yaml)

    # create scheduler
    # load diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    # model_id = "stable_diff_local" # load local save of model (for internet problems)

    device = f"cuda:{args.device_num}"

    cfg_scale_src = args.cfg_src
    cfg_scale_tar_list = [args.cfg_tar]
    eta = args.eta # = 1
    skip_zs = [args.skip]
    xa_sa_string = f'_xa_{args.xa}_sa{args.sa}_' if args.mode=='p2pinv' else '_'

    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    # load/reload model:
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    for i in range(len(full_data)):
        current_image_data = full_data[i]
        image_path = current_image_data['init_img']
        image_path = '.' + image_path 
        image_folder = image_path.split('/')[1] # after '.'
        prompt_src = current_image_data.get('source_prompt', "") # default empty string
        prompt_tar_list = current_image_data['target_prompts']

        if args.mode=="p2pddim" or args.mode=="ddim":
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            ldm_stable.scheduler = scheduler
        else:
            ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
            
        ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)

        # load image
        offsets=(0,0,0,0)
        x0 = load_512(image_path, *offsets, device)

        # vae encode image
        with autocast("cuda"), inference_mode():
            w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()

        # find Zs and wts - forward process
        if args.mode=="p2pddim" or args.mode=="ddim":
            wT = ddim_inversion(ldm_stable, w0, prompt_src, cfg_scale_src)
        else:
            wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=args.num_diffusion_steps)

        # iterate over decoder prompts
        for k in range(len(prompt_tar_list)):
            prompt_tar = prompt_tar_list[k]
            save_path = os.path.join(f'./results/', args.mode+xa_sa_string+str(time_stamp), image_path.split(sep='.')[0], 'src_' + prompt_src.replace(" ", "_"), 'dec_' + prompt_tar.replace(" ", "_"))
            os.makedirs(save_path, exist_ok=True)

            # Check if number of words in encoder and decoder text are equal
            src_tar_len_eq = (len(prompt_src.split(" ")) == len(prompt_tar.split(" ")))

            for cfg_scale_tar in cfg_scale_tar_list:
                for skip in skip_zs:    
                    if args.mode=="our_inv":
                        # reverse process (via Zs and wT)
                        controller = AttentionStore()
                        register_attention_control(ldm_stable, controller)
                        w0, _ = inversion_reverse_process(ldm_stable, xT=wts[skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[skip:], controller=controller)

                    elif args.mode=="p2pinv":
                        # inversion with attention replace
                        cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                        prompts = [prompt_src, prompt_tar]
                        if src_tar_len_eq:
                            controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)
                        else:
                            # Should use Refine for target prompts with different number of tokens
                            controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)

                        register_attention_control(ldm_stable, controller)
                        w0, _ = inversion_reverse_process(ldm_stable, xT=wts[skip], etas=eta, prompts=prompts, cfg_scales=cfg_scale_list, prog_bar=True, zs=zs[skip:], controller=controller)
                        w0 = w0[1].unsqueeze(0)

                    elif args.mode=="p2pddim" or args.mode=="ddim":
                        # only z=0
                        if skip != 0:
                            continue
                        prompts = [prompt_src, prompt_tar]
                        if args.mode=="p2pddim":
                            if src_tar_len_eq:
                                controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
                            # Should use Refine for target prompts with different number of tokens
                            else:
                                controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
                        else:
                            controller = EmptyControl()

                        register_attention_control(ldm_stable, controller)
                        # perform ddim inversion
                        cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
                        w0, latent = text2image_ldm_stable(ldm_stable, prompts, controller, args.num_diffusion_steps, cfg_scale_list, None, wT)
                        w0 = w0[1:2]
                    else:
                        raise NotImplementedError
                    
                    # vae decode image
                    with autocast("cuda"), inference_mode():
                        x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
                    if x0_dec.dim()<4:
                        x0_dec = x0_dec[None,:,:,:]
                    img = image_grid(x0_dec)
                       
                    # same output
                    current_GMT = time.gmtime()
                    time_stamp_name = calendar.timegm(current_GMT)
                    image_name_png = f'cfg_d_{cfg_scale_tar}_' + f'skip_{skip}_{time_stamp_name}' + ".png"

                    save_full_path = os.path.join(save_path, image_name_png)
                    img.save(save_full_path)

