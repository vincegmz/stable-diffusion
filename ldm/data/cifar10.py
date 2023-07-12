import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light

import json
import random
def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())


class Cifar10Base(Dataset):
    def __init__(self, data_root,label_path,mode, process_images, config=None):
        self.label_path = label_path
        self.data_root = data_root
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        # self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = process_images  # if False we skip loading & processing images and self.data contains filepaths
        self._prepare()
        # self._prepare_synset_to_human()
        # self._prepare_idx_to_synset()
        # self._prepare_human_to_integer_label()
        self._load(mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # img = Image.fromarray(self.data[i]['image'])
        # img.save('~/Downloads')
        return self.data[i]

    def _prepare(self):
        self.datadir =os.path.join(self.data_root,'imgs')
        self.train_list = os.path.join(self.data_root, "trainList.txt")
        self.val_list = os.path.join(self.data_root, "valList.txt")

        #TODO: Add random_crop to cifar 10
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
                                    default=True)
        with open(self.label_path) as f:
            self.labels = json.load(f)
        if not os.path.exists(self.train_list):
            def numerical_sort_key(file_path):
                id = file_path.split('/')[-1][:-4]
                try:
                    return int(id)
                except ValueError:
                    return id
            all_paths = sorted(list(self.labels.keys()),key = numerical_sort_key)
            num_imgs = 50000
            num_train = int(num_imgs*0.8)
            train_idx = random.sample(range(num_imgs),num_train)
            val_idx = set(range(num_train))-set(train_idx)
            train_paths = [all_paths[i] for i in train_idx]
            val_paths = [all_paths[i] for i in val_idx]
            train_paths = "\n".join(train_paths)+"\n"
            val_paths = "\n".join(val_paths)+"\n"
            with open(self.train_list,'w') as f:
                f.write(train_paths)

            with open(self.val_list,'w') as f:
                f.write(val_paths)


    def _filter_relpaths(self, relpaths):
        ignore = set([
            "n06596364_9591.JPEG",
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            self.synset2idx = synset2idx(path_to_yaml=self.idx2syn)
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or
                not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)

    def _prepare_idx_to_synset(self):
        URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
        self.idx2syn = os.path.join(self.root, "index_synset.yaml")
        if (not os.path.exists(self.idx2syn)):
            download(URL, self.idx2syn)

    def _prepare_human_to_integer_label(self):
        URL = "https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1"
        self.human2integer = os.path.join(self.root, "imagenet1000_clsidx_to_labels.txt")
        if (not os.path.exists(self.human2integer)):
            download(URL, self.human2integer)
        with open(self.human2integer, "r") as f:
            lines = f.read().splitlines()
            assert len(lines) == 1000
            self.human2integer_dict = dict()
            for line in lines:
                value, key = line.split(":")
                self.human2integer_dict[key] = int(value)

    def _load(self,mode):
        if mode =='train':
            with open(self.train_list, "r") as f:
                self.relpaths = f.read().splitlines()
        elif mode == 'val':
            with open(self.val_list, "r") as f:
                self.relpaths = f.read().splitlines()
        else:
            raise NotImplementedError()
        
        self.synsets = [None for _ in self.relpaths]
        self.abspaths = [os.path.join(os.path.dirname(self.data_root), p) for p in self.relpaths]
        self.class_labels = [int(self.labels[relpath]) for relpath in self.relpaths]
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
        self.human_labels = [human_dict[s] for s in self.class_labels]

        labels = {
            "relpath": np.array(self.relpaths),
            # "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
            "caption":np.array(['']*len(self.relpaths))
        }

        if self.process_images:
            self.size = retrieve(self.config, "size", default=256)
            self.data = ImagePaths(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   random_crop=self.random_crop
                                   )
        else:
            self.data = self.abspaths


class Cifar10Train(Cifar10Base):
    def __init__(self, process_images=True, data_root='~/dataset/cifar-10-batches-py',**kwargs):
        label_path = os.path.join(data_root,'label.json')
        super().__init__(data_root,label_path,'train', process_images,**kwargs)


class Cifar10Validation(Cifar10Base):

    def __init__(self, process_images=True, data_root='~/dataset/cifar-10-batches-py',**kwargs):
        label_path = os.path.join(data_root,'label.json')
        super().__init__(data_root,label_path,'val', process_images,**kwargs)


class Cifar10TrainOneImage(Cifar10Base):
    def __init__(self, process_images=True, data_root='~/dataset/cifar-10-batches-py',**kwargs):
        label_path = os.path.join(data_root,'label.json')
        super().__init__(data_root,label_path,'train', process_images,**kwargs)
    def __len__(self):
        return 1000

    def __getitem__(self, i):
        return self.data[0]

class Cifar10ValidationOneImage(Cifar10Base):

    def __init__(self, process_images=True, data_root='~/dataset/cifar-10-batches-py',**kwargs):
        label_path = os.path.join(data_root,'label.json')
        super().__init__(data_root,label_path,'train', process_images,**kwargs)
    def __len__(self):
        return 100

    def __getitem__(self, i):
        return self.data[0]





class ImageNetSR(Dataset):
    def __init__(self, size=None,
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.base = self.get_base()
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fn = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": PIL.Image.NEAREST,
            "pil_bilinear": PIL.Image.BILINEAR,
            "pil_bicubic": PIL.Image.BICUBIC,
            "pil_box": PIL.Image.BOX,
            "pil_hamming": PIL.Image.HAMMING,
            "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = self.base[i]
        image = Image.open(example["file_path_"])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)

        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]

        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        else:
            LR_image = self.degradation_process(image=image)["image"]

        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)

        return example


class ImageNetSRTrain(ImageNetSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_train_hr_indices.p", "rb") as f:
            indices = pickle.load(f)
        dset = ImageNetTrain(process_images=False,)
        return Subset(dset, indices)


class ImageNetSRValidation(ImageNetSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_val_hr_indices.p", "rb") as f:
            indices = pickle.load(f)
        dset = ImageNetValidation(process_images=False,)
        return Subset(dset, indices)


class LSUNChurchesTrain(Cifar10Base):
    def __init__(self, **kwargs):
        super().__init__(txt_file="~/dataset/cifar-10-batches-py/labels.json", data_root="~/dataset/cifar-10-batches-py/imgs", **kwargs)


class LSUNChurchesValidation(Cifar10Base):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="~/dataset/cifar-10-batches-py/labels.json", data_root="~/dataset/cifar-10-batches-py/imgs",
                         flip_p=flip_p, **kwargs)