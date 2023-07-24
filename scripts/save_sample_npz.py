import argparse, os, sys, glob
import cv2
import torch
import numpy as np
import io
import warnings
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMConsistencySampler, ConsistentSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from torchvision.transforms import Resize

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False,idx = 0):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # ema_params = model.ema_parameters
    # for key in list(sd.keys()):
    #     key_split = key.split('_')
    #     if len(key_split) == 2 and key_split[0] == 'ema':
    #         padded_key = "{:03}".format(int(key_split[-1]))
    #         ema_params[idx][padded_key] = sd[key]
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
        
    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=40,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--consistent",
        action='store_true',
        help = 'use multistep consistent sampling'
    )
    parser.add_argument(
        "--ddimconsistent",
        action='store_true',
        help = 'use ddim consistency sampling'
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=6,
        help="use safety checker",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="use safety checker",
    )
    parser.add_argument(
        "--ema_idx",
        type=int,
        help="index for select ema model, if None us online model ",
    )
    parser.add_argument(
        "--begin_ckpt",
        type = int,
        default = 0
    )
    parser.add_argument(
        "--end_ckpt",
        type = int,
        default = 0
    )
    parser.add_argument(
        "--start_code_path",
        type = str,
        default='/home/ubuntu/exp/stablediffusion/test512/start_code.pt'

    )
    parser.add_argument(
        "--run_name",
        type = str,

    )


    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)
    for ckpt in range(opt.begin_ckpt,opt.end_ckpt+1):
        config = OmegaConf.load(f"{opt.config}")
        path = os.path.join(opt.ckpt_root,f"epoch={ckpt:06d}.ckpt")
        if os.path.exists(path):
            model = load_model_from_config(config, f"{path}")
        else:
            raise FileNotFoundError('ckpt not exists')
            print(f"Instantiate from config")
            model = instantiate_from_config(config.model)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        if opt.consistent:
            sampler = ConsistentSolverSampler(model,sampler = 'multistep')
        elif opt.ddimconsistent:
            sampler = DDIMConsistencySampler(model)
        elif opt.dpm_solver:
            sampler = DPMSolverSampler(model)
        elif opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        if not opt.from_file:
            prompt = opt.prompt
            assert prompt is not None


        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
            if os.path.exists(opt.start_code_path):
                temp_code = torch.load(opt.start_code_path)
                assert start_code.shape == temp_code.shape
                start_code = temp_code
            else:
                torch.save(start_code,opt.start_code_path)
                warnings.warn('start_code is not fixed across attempts',UserWarning)


        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        num_iterations = opt.dataset_size//batch_size
        human_dict = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope(idx = opt.ema_idx):
                    tic = time.time()
                    all_samples = list()
                    id = 0
                    out_dir_name = os.path.join(outpath,f"epoch={ckpt:06d}",opt.run_name)
                    os.makedirs(out_dir_name,exist_ok=True)
                    if os.path.exists(os.path.join(out_dir_name,'current_itr.txt')):
                        with open(opt.current_itr_path,'r') as f:
                            current_itr = int(f.read())
                    else:
                        current_itr = 0   
                    for n in trange(current_itr,num_iterations, desc="Sampling"):
                        data = []
                        for _ in range(batch_size):
                            data.append(prompt.replace('*',human_dict[id%10]))
                            id+=1
                        data = [data]
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code,
                                                            model_type = opt.model_type)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            # why clamp (x_samples_ddim + 1.0) / 2.0? 
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1)
                            x_samples_ddim = (x_samples_ddim*255).to(torch.uint8).numpy()
                            # for idx_img in range(x_samples_ddim.shape[0]):
                            #     cv2.imwrite(os.path.join(outpath, f'grid-{grid_count:04}.png'), x_samples_ddim[idx_img, :, :, ::-1])
                            #     grid_count += 1
                        
                            with open(os.path.join(out_dir_name,f"samples_{n}.npz"), "wb") as fout:
                                io_buffer = io.BytesIO()
                                # samples =  torch.clip(x_samples_ddim* 255., 0, 255).to(torch.uint8)
                                resize = Resize((32,32))
                                resized_samples = [np.array(resize(Image.fromarray(x_samples_ddim[i]))) for i in range(x_samples_ddim.shape[0])]
                                np.savez_compressed(io_buffer, samples=np.stack(resized_samples))
                                fout.write(io_buffer.getvalue())
                            with open(os.path.join(out_dir_name,'current_itr.txt'),'w') as f:
                                f.write(f'{n}')
                            torch.cuda.empty_cache()
                            # test code
                            # shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            # samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                            #                                  conditioning=c,
                            #                                  batch_size=opt.n_samples,
                            #                                  shape=shape,
                            #                                  verbose=False,
                            #                                  unconditional_guidance_scale=opt.scale,
                            #                                  unconditional_conditioning=uc,
                            #                                  eta=opt.ddim_eta,
                            #                                  x_T=torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device))

                            # x_samples_ddim = model.decode_first_stage(samples_ddim)
                            # # why clamp (x_samples_ddim + 1.0) / 2.0? 
                            # x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            # x_samples_ddim = x_samples_ddim.permute(0,2,3,1)
                            # samples =  torch.clip(x_samples_ddim* 255., 0, 255).to(torch.uint8)
                            # cv2.imwrite("test.png", samples.cpu()[0].numpy())


                            # with open(os.path.join(outpath, f"samples_{n}.npz"), "wb") as fout:
                            #     io_buffer = io.BytesIO()
                            #     samples =  torch.clip(x_samples_ddim* 255., 0, 255).to(torch.uint8)
                            #     np.savez_compressed(io_buffer, samples=samples.cpu())
                            #     fout.write(io_buffer.getvalue())

                            # x_samples_ddim = x_samples_ddim.cpu().numpy()

                            # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                            # x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            # x_checked_image_torch = samples.permute(0, 3, 1, 2)
    
                            # if not opt.skip_save:
                            #     for x_sample in x_checked_image_torch:
                            #         x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            #         img = Image.fromarray(x_sample.astype(np.uint8))
                            #         img = put_watermark(img, wm_encoder)
                            #         img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            #         base_count += 1

                    #         if not opt.skip_grid:
                    #             all_samples.append(x_samples_ddim)

                    # if not opt.skip_grid:
                    #     # additionally, save as grid
                    #     grid = torch.stack(all_samples, 0)
                    #     grid = rearrange(grid, 'n b h w c-> (n b) h w c')
                    #     grid = make_grid(grid, nrow=n_rows)

                    #     # to image
                    #     grid = 255. * grid.numpy()
                    #     img = Image.fromarray(grid.astype(np.uint8))
                    #     # img = put_watermark(img, wm_encoder)
                        
                    #     grid_count += 1

                    toc = time.time()

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")


if __name__ == "__main__":
    main()
