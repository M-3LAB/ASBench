import cv2
import os
import numpy as np
from einops import rearrange
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import imgaug.augmenters as iaa

from .perlin import rand_perlin_2d_np

from typing import List
import glob
import matplotlib.pyplot as plt
import random
from PIL import Image
from torch.utils.data import Dataset
from einops import rearrange
from data.data_utils import perlin_noise
from multiprocessing import Value, Lock
from data.cutpaste import cut_paste_normal,cut_paste_scar
from data.CutOut import Cutout
from data.NSA_generation import patch_ex
from data.FPI_generation import synthesize_anomalies_pil
from data.Fractal_Aug import FAG
# from torchvision.transforms import ToTensor, Normalize, Compose, ToPILImage, RandomHorizontalFlip

# from data.perlin import rand_perlin_2d_np
# from data.utils import torch_seed

# from typing import Union, List, Tuple
# import matplotlib.pyplot as plt
counter = Value('i',0)
counter_lock = Lock()
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class MemSegDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, to_memory: bool = False, 
        resize: List[int] = [256, 256], imagesize: int = 224,
        texture_source_dir: str = None, structure_grid_size: str = 8,
        transparency_range: List[float] = [0.15, 1.],
        perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
        use_mask: bool = True, bg_threshold: float = 100, bg_reverse: bool = False, percent: float = 0.5
    ):
        # mode
        self.is_train = is_train 
        self.to_memory = to_memory
        self.percent = percent
        # load image file list
        self.datadir = datadir
        self.target = target
        self.file_list = glob.glob(os.path.join(self.datadir, self.target, 'train/*/*' if is_train else 'test/*/*'))
        
        # synthetic anomaly
        if self.is_train and not self.to_memory:
            # load texture image file list    
            self.texture_source_file_list = glob.glob(os.path.join(texture_source_dir,'*/*')) if texture_source_dir else None
            # perlin noise
            self.perlin_scale = perlin_scale
            self.min_perlin_scale = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold
            
            # structure
            self.structure_grid_size = structure_grid_size
            
            # anomaly mixing
            self.transparency_range = transparency_range
            
            # mask setting
            self.use_mask = use_mask
            self.bg_threshold = bg_threshold
            self.bg_reverse = bg_reverse
            
        # transform
        self.resize = list(resize)
        self.transform_img = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.anomaly_switch = False
        
    def __getitem__(self, idx):
        use_mask = True
        bg_threshold = 100
        bg_reverse = False
        file_path = self.file_list[idx]
        # print(file_path)
        # image
        img = Image.open(file_path).convert("RGB").resize(self.resize)
        img = np.array(img)
        image_path = file_path 
        # target
        target = 0 if 'good' in self.file_list[idx] else 1
        
        # mask
        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            mask = Image.open(file_path.replace('test','ground_truth').replace('.png','.png')).resize(self.resize)
            mask = np.array(mask)
        
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly < self.percent:
            self.anomaly_switch = True
        else:
            self.anomaly_switch = False
        ## anomaly source
        if self.is_train and not self.to_memory:
            if self.anomaly_switch:
                img, mask = self.generate_anomaly(img=img, texture_img_list=self.texture_source_file_list)
                target = 1
                Image.fromarray(img.astype(np.uint8)).convert('RGB').save("MemSeg_aug.png")
                Image.fromarray((mask*255).astype(np.uint8)).save("MemSeg_mask.png")
                mask = torch.Tensor(mask)

        # mask = np.expand_dims(mask, axis=0)
        img = self.transform_img(img)
        mask = self.transform_mask(mask).squeeze()
        # mask = mask.unsqueeze(0)
        # print(img.shape)
        # print(mask.shape)
        return img, mask, target, image_path
        
        
    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        
        return aug
        
    def generate_anomaly(self, img: np.ndarray, texture_img_list: list = None) -> List[np.ndarray]:
        '''
        step 1. generate mask
            - target foreground mask
            - perlin noise mask
            
        step 2. generate texture or structure anomaly
            - texture: load DTD
            - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation, 
            and hue on the input image  𝐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid 
            and randomly arranged to obtain the disordered image  𝐼 
            
        step 3. blending image and anomaly source
        '''
        
        # step 1. generate mask
        img_size = img.shape[:-1] # H x W
        
        ## target foreground mask
        if self.use_mask:
            target_foreground_mask = self.generate_target_foreground_mask(img=img)
        else:
            target_foreground_mask = np.ones(self.resize)
        
        ## perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask(img_size=img_size)
        
        ## mask
        mask = perlin_noise_mask * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)
        # step 2. generate texture or structure anomaly
        ## anomaly source
        anomaly_source_img = self.anomaly_source(img=img, texture_img_list=texture_img_list)
        
        ## mask anomaly parts
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
        
        # step 3. blending image and anomaly source
        anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img
        
        return (anomaly_source_img.astype(np.uint8), mask)
    
    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(img_gray, self.bg_threshold, 255, cv2.THRESH_BINARY)
        target_background_mask = target_background_mask.astype(np.bool_).astype(np.int32)

        # invert mask for foreground mask
        if self.bg_reverse:
            target_foreground_mask = target_background_mask
        else:
            target_foreground_mask = -(target_background_mask - 1)
        # print(self.bg_reverse)
        # print(self.bg_threshold)
        return target_foreground_mask
    
    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise        
        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))
        
        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)
        
        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold, 
            np.ones_like(perlin_noise), 
            np.zeros_like(perlin_noise)
        )
        
        return mask_noise
    
    def anomaly_source(self, img: np.ndarray, texture_img_list: list = None) -> np.ndarray:
        p = np.random.uniform() if texture_img_list else 1.0
        if p < 0.5:
            idx = np.random.choice(len(texture_img_list))
            img_size = img.shape[:-1] # H x W
            anomaly_source_img = self._texture_source(img_size=img_size, texture_img_path=texture_img_list[idx])
        else:
            anomaly_source_img = self._structure_source(img=img)
            
        return anomaly_source_img
        
    def _texture_source(self, img_size: tuple, texture_img_path: str) -> np.ndarray:
        texture_source_img = cv2.imread(texture_img_path)
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img, dsize=img_size).astype(np.float32)
        
        return texture_source_img
        
    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        structure_source_img = self.rand_augment()(image=img)
        
        img_size = img.shape[:-1] # H x W
        
        assert img_size[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = img_size[1] // self.structure_grid_size
        grid_h = img_size[0] // self.structure_grid_size
        
        structure_source_img = rearrange(
            tensor  = structure_source_img, 
            pattern = '(h gh) (w gw) c -> (h w) gw gh c',
            gw      = grid_w, 
            gh      = grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor  = structure_source_img[disordered_idx], 
            pattern = '(h w) gw gh c -> (h gh) (w gw) c',
            h       = self.structure_grid_size,
            w       = self.structure_grid_size
        ).astype(np.float32)
        
        return structure_source_img
        
    def __len__(self):
        return len(self.file_list)
    

class MVTecDataset(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[288, 288],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        imagesize=256,
        percent=0.5
    ):
        super().__init__()
        with counter_lock:
            counter.value = 0
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.category = category
        self.percent = percent
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.CenterCrop(imagesize),
                    transforms.ToTensor(),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)
        image_path = self.mvtec_paths[index]
        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
        dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

        # perlin_noise implementation
        aug_image, aug_mask = perlin_noise(image, dtd_image, aug_prob=self.percent)
        aug_image = aug_image*255
#         aug_mask = aug_mask.squeeze()
#         no_anomaly = torch.rand(1).numpy()[0]
# #         if no_anomaly<self.percent:
# #             Image.fromarray((aug_image*255).astype(np.uint8)).convert('RGB').save("DestSeg_aug.png")
# #             Image.fromarray((aug_mask*255).astype(np.uint8)).save("DestSeg_mask.png")
        if(np.sum(aug_mask)==0):
            target=0
        else:
            target=1
        aug_image = self.final_preprocessing(aug_image.astype(np.uint8))
        aug_mask = aug_mask.squeeze()
        aug_mask = self.transform_mask(aug_mask.astype(np.float32)).squeeze()
        return aug_image,aug_mask,target,image_path
#         else:
#             aug_mask = np.zeros((256,256), np.float32)
# #             image.save("DestSeg_aug.png")
# #             Image.fromarray((aug_mask*255).astype(np.uint8)).save("DestSeg_mask.png")
#             target=0
#             image = np.array(image).astype(np.uint8)
#             aug_image = self.final_preprocessing(image)
#             aug_mask = self.transform_mask(aug_mask)
#             return aug_image,aug_mask,target,image_path

class MVTecDataset_CutPaste(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[288, 288],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        imagesize=256,
        percent=0.5
    ):
        super().__init__()
        with counter_lock:
            counter.value = 0
        self.percent = percent
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.category = category
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.CenterCrop(imagesize),
                    transforms.ToTensor(),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)
        image_path = self.mvtec_paths[index]
        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
        dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

        # perlin_noise implementation
        aug_image, aug_mask = cut_paste_normal(image, area_ratio=[0.02,0.15], aspect_ratio=0.3, color_jitter=None)
        aug_mask = aug_mask/255
        aug_mask = aug_mask.squeeze()
        aug_image = np.array(aug_image)
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly<self.percent:
#             Image.fromarray((aug_image).astype(np.uint8)).convert('RGB').save("CutPaste_aug.png")
#             Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=1
            aug_image = self.final_preprocessing(aug_image.astype(np.uint8))
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path
        else:
            aug_mask = np.zeros((256,256), np.float32)
#             image.save("CutPaste_aug.png")
#             Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=0
            image = np.array(image).astype(np.uint8)
            aug_image = self.final_preprocessing(image)
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path

class MVTecDataset_CutPaste_Scar(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[288, 288],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        imagesize=256,
        percent=0.5
    ):
        super().__init__()
        with counter_lock:
            counter.value = 0
        self.percent = percent
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.category = category
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.CenterCrop(imagesize),
                    transforms.ToTensor(),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)
        image_path = self.mvtec_paths[index]
        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
        dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

        # perlin_noise implementation
        aug_image, aug_mask = cut_paste_scar(image, width=[2, 16], height=[10, 25], rotation=[-30, 30])
        aug_mask = aug_mask/255
        aug_mask = aug_mask.squeeze()
        aug_image = np.array(aug_image)
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly<self.percent:
            Image.fromarray((aug_image).astype(np.uint8)).convert('RGB').save("CutPaste_aug.png")
            Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=1
            aug_image = self.final_preprocessing(aug_image.astype(np.uint8))
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path
        else:
            aug_mask = np.zeros((256,256), np.float32)
            image.save("CutPaste_aug.png")
            Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=0
            image = np.array(image).astype(np.uint8)
            aug_image = self.final_preprocessing(image)
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path

class MVTecDataset_NSA(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[288, 288],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        imagesize=256,
        percent=0.5,
    ):
        super().__init__()
        with counter_lock:
            counter.value = 0
        self.percent = percent
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.category = category
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.CenterCrop(imagesize),
                    transforms.ToTensor(),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)
        image_path = self.mvtec_paths[index]
        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
        dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

        # perlin_noise implementation
        aug_image, aug_mask = patch_ex(image,target=self.category)
        #aug_mask = aug_mask/255
        aug_mask = aug_mask.squeeze()
        aug_image = np.array(aug_image)
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly<self.percent:
            Image.fromarray((aug_image).astype(np.uint8)).convert('RGB').save("CutPaste_aug.png")
            Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=1
            aug_image = self.final_preprocessing(aug_image.astype(np.uint8))
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path
        else:
            aug_mask = np.zeros((256,256), np.float32)
            image.save("CutPaste_aug.png")
            Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=0
            image = np.array(image).astype(np.uint8)
            aug_image = self.final_preprocessing(image)
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path
        
class MVTecDataset_FPI(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[288, 288],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        imagesize=256,
        percent=0.5
    ):
        super().__init__()
        with counter_lock:
            counter.value = 0
        self.percent = percent
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.category = category
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.CenterCrop(imagesize),
                    transforms.ToTensor(),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)
        image_path = self.mvtec_paths[index]
        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
        dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

        # perlin_noise implementation
        aug_image,aug_mask = synthesize_anomalies_pil(image)
        aug_mask = aug_mask.squeeze()
        aug_image = np.array(aug_image)
        aug_mask = aug_mask.astype(np.float32)
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly<self.percent:
#             Image.fromarray((aug_image).astype(np.uint8)).convert('RGB').save("CutPaste_aug.png")
#             Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=1
            aug_image = self.final_preprocessing(aug_image.astype(np.uint8))
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path
        else:
            aug_mask = np.zeros((256,256), np.float32)
#             image.save("CutPaste_aug.png")
#             Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=0
            image = np.array(image).astype(np.uint8)
            aug_image = self.final_preprocessing(image)
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path
        
class MVTecDataset_Fractal(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[288, 288],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        imagesize=256,
        percent=0.5
    ):
        super().__init__()
        with counter_lock:
            counter.value = 0
        self.percent = percent
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.category = category
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.CenterCrop(imagesize),
                    transforms.ToTensor(),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)
        image_path = self.mvtec_paths[index]
        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
        dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

        # perlin_noise implementation
        fag = FAG(load_size=256)
        aug_image, aug_mask = fag(image)
        aug_mask = aug_mask/255
        aug_mask = aug_mask.squeeze()
        aug_image = np.array(aug_image)
        aug_mask = aug_mask.astype(np.float32)
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly<self.percent:
            target=1
            aug_image = self.final_preprocessing(aug_image.astype(np.uint8))
            aug_mask = self.transform_mask(aug_mask).squeeze()
            return aug_image,aug_mask,target,image_path
        else:
            aug_mask = np.zeros((256,256), np.float32)
            target=0
            image = np.array(image).astype(np.uint8)
            aug_image = self.final_preprocessing(image)
            aug_mask = self.transform_mask(aug_mask).squeeze()
            return aug_image,aug_mask,target,image_path

class MVTecDataset_CutOut(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[288, 288],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        imagesize=256,
        percent=0.5
    ):
        super().__init__()
        with counter_lock:
            counter.value = 0
        self.percent = percent
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.category = category
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.CenterCrop(imagesize),
                    transforms.ToTensor(),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)
        image_path = self.mvtec_paths[index]
        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
        dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

        cutout = Cutout(n_holes=1, length=random.randint(20, 50))
        aug_image,aug_mask = cutout(image)
        aug_mask = aug_mask.squeeze()
        aug_image = np.array(aug_image)*255
        aug_mask = aug_mask.astype(np.float32)
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly<self.percent:
            Image.fromarray((aug_image).astype(np.uint8)).convert('RGB').save("CutPaste_aug.png")
            Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=1
            aug_image = self.final_preprocessing(aug_image.astype(np.uint8))
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path
        else:
            aug_mask = np.zeros((256,256), np.float32)
            image.save("CutPaste_aug.png")
            Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=0
            image = np.array(image).astype(np.uint8)
            aug_image = self.final_preprocessing(image)
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path

class MemSegDataset_RealNet(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, to_memory: bool = False, 
        resize: List[int] = [256, 256], imagesize: int = 224,
        texture_source_dir: str = None, structure_grid_size: str = 8,
        transparency_range: List[float] = [0.15, 1.],
        perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
        use_mask: bool = True, bg_threshold: float = 100, bg_reverse: bool = False, percent: float = 0.5
    ):
        # mode
        self.is_train = is_train 
        self.to_memory = to_memory
        self.percent = percent
        # load image file list
        self.datadir = datadir
        self.target = target
        self.file_list = glob.glob(os.path.join(self.datadir, self.target, 'train/*/*' if is_train else 'test/*/*'))
        
        # synthetic anomaly
        if self.is_train and not self.to_memory:
            # load texture image file list    
            self.texture_source_file_list = glob.glob(os.path.join(texture_source_dir,self.target,'*')) if texture_source_dir else None
        
            # perlin noise
            self.perlin_scale = perlin_scale
            self.min_perlin_scale = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold
            
            # structure
            self.structure_grid_size = structure_grid_size
            
            # anomaly mixing
            self.transparency_range = transparency_range
            
            # mask setting
            self.use_mask = use_mask
            self.bg_threshold = bg_threshold
            self.bg_reverse = bg_reverse
            
        # transform
        self.resize = list(resize)
        self.transform_img = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.anomaly_switch = False
        
    def __getitem__(self, idx):
        
        file_path = self.file_list[idx]
        # print(file_path)
        # image
        img = Image.open(file_path).convert("RGB").resize(self.resize)
        img = np.array(img)
        image_path = file_path 
        # target
        target = 0 if 'good' in self.file_list[idx] else 1
        
        # mask
        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            mask = Image.open(file_path.replace('test','ground_truth').replace('.png','_mask.png')).resize(self.resize)
            mask = np.array(mask)
        
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly < self.percent:
            self.anomaly_switch = True
        else:
            self.anomaly_switch = False
        ## anomaly source
        if self.is_train and not self.to_memory:
            if self.anomaly_switch:
                img, mask = self.generate_anomaly(img=img, texture_img_list=self.texture_source_file_list)
                target = 1
#                 Image.fromarray(img.astype(np.uint8)).convert('RGB').save("MemSeg_aug.png")
#                 Image.fromarray((mask*255).astype(np.uint8)).save("MemSeg_mask.png")
                mask = torch.Tensor(mask)

        img = self.transform_img(img)
        mask = self.transform_mask(mask).squeeze()
        return img, mask, target, image_path
        
        
    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        
        return aug
        
    def generate_anomaly(self, img: np.ndarray, texture_img_list: list = None) -> List[np.ndarray]:
        '''
        step 1. generate mask
            - target foreground mask
            - perlin noise mask
            
        step 2. generate texture or structure anomaly
            - texture: load DTD
            - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation, 
            and hue on the input image  𝐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid 
            and randomly arranged to obtain the disordered image  𝐼 
            
        step 3. blending image and anomaly source
        '''
        
        # step 1. generate mask
        img_size = img.shape[:-1] # H x W
        
        ## target foreground mask
        if self.use_mask:
            target_foreground_mask = self.generate_target_foreground_mask(img=img)
        else:
            target_foreground_mask = np.ones(self.resize)
        
        ## perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask(img_size=img_size)
        
        ## mask
        mask = perlin_noise_mask * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)
        # step 2. generate texture or structure anomaly
        ## anomaly source
        anomaly_source_img = self.anomaly_source(img=img, texture_img_list=texture_img_list)
        
        ## mask anomaly parts
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
        
        # step 3. blending image and anomaly source
        anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img
        
        return (anomaly_source_img.astype(np.uint8), mask)
    
    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(img_gray, self.bg_threshold, 255, cv2.THRESH_BINARY)
        target_background_mask = target_background_mask.astype(np.bool_).astype(np.int32)

        # invert mask for foreground mask
        if self.bg_reverse:
            target_foreground_mask = target_background_mask
        else:
            target_foreground_mask = -(target_background_mask - 1)
        # print(self.bg_reverse)
        # print(self.bg_threshold)
        return target_foreground_mask
    
    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise        
        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))
        
        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)
        
        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold, 
            np.ones_like(perlin_noise), 
            np.zeros_like(perlin_noise)
        )
        
        return mask_noise
    
    def anomaly_source(self, img: np.ndarray, texture_img_list: list = None) -> np.ndarray:
        p = np.random.uniform() if texture_img_list else 1.0
        if p < 0.5:
            idx = np.random.choice(len(texture_img_list))
            img_size = img.shape[:-1] # H x W
            anomaly_source_img = self._texture_source(img_size=img_size, texture_img_path=texture_img_list[idx])
        else:
            anomaly_source_img = self._structure_source(img=img)
            
        return anomaly_source_img
        
    def _texture_source(self, img_size: tuple, texture_img_path: str) -> np.ndarray:
        texture_source_img = cv2.imread(texture_img_path)
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img, dsize=img_size).astype(np.float32)
        
        return texture_source_img
        
    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        structure_source_img = self.rand_augment()(image=img)
        
        img_size = img.shape[:-1] # H x W
        
        assert img_size[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = img_size[1] // self.structure_grid_size
        grid_h = img_size[0] // self.structure_grid_size
        
        structure_source_img = rearrange(
            tensor  = structure_source_img, 
            pattern = '(h gh) (w gw) c -> (h w) gw gh c',
            gw      = grid_w, 
            gh      = grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor  = structure_source_img[disordered_idx], 
            pattern = '(h w) gw gh c -> (h gh) (w gw) c',
            h       = self.structure_grid_size,
            w       = self.structure_grid_size
        ).astype(np.float32)
        
        return structure_source_img
        
    def __len__(self):
        return len(self.file_list)
    
class MemSegDataset_RealNet(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, to_memory: bool = False, 
        resize: List[int] = [256, 256], imagesize: int = 224,
        texture_source_dir: str = None, structure_grid_size: str = 8,
        transparency_range: List[float] = [0.15, 1.],
        perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
        use_mask: bool = True, bg_threshold: float = 100, bg_reverse: bool = False, percent: float = 0.5
    ):
        # mode
        self.is_train = is_train 
        self.to_memory = to_memory
        self.percent = percent
        # load image file list
        self.datadir = datadir
        self.target = target
        self.file_list = glob.glob(os.path.join(self.datadir, self.target, 'train/*/*' if is_train else 'test/*/*'))
        
        # synthetic anomaly
        if self.is_train and not self.to_memory:
            # load texture image file list    
            self.texture_source_file_list = glob.glob(os.path.join(texture_source_dir,self.target,'*')) if texture_source_dir else None
            # perlin noise
            self.perlin_scale = perlin_scale
            self.min_perlin_scale = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold
            
            # structure
            self.structure_grid_size = structure_grid_size
            
            # anomaly mixing
            self.transparency_range = transparency_range
            
            # mask setting
            self.use_mask = use_mask
            self.bg_threshold = bg_threshold
            self.bg_reverse = bg_reverse
            
        # transform
        self.resize = list(resize)
        self.transform_img = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.anomaly_switch = False
        
    def __getitem__(self, idx):
        
        file_path = self.file_list[idx]
        # print(file_path)
        # image
        img = Image.open(file_path).convert("RGB").resize(self.resize)
        img = np.array(img)
        image_path = file_path 
        # target
        target = 0 if 'good' in self.file_list[idx] else 1
        
        # mask
        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            mask = Image.open(file_path.replace('test','ground_truth').replace('.png','_mask.png')).resize(self.resize)
            mask = np.array(mask)
        
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly < self.percent:
            self.anomaly_switch = True
        else:
            self.anomaly_switch = False
        ## anomaly source
        if self.is_train and not self.to_memory:
            if self.anomaly_switch:
                img, mask = self.generate_anomaly(img=img, texture_img_list=self.texture_source_file_list)
                target = 1
                Image.fromarray(img.astype(np.uint8)).convert('RGB').save("MemSeg_aug.png")
                Image.fromarray((mask*255).astype(np.uint8)).save("MemSeg_mask.png")
                mask = torch.Tensor(mask)

        img = self.transform_img(img)
        mask = self.transform_mask(mask).squeeze()
        return img, mask, target, image_path
        
        
    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        
        return aug
        
    def generate_anomaly(self, img: np.ndarray, texture_img_list: list = None) -> List[np.ndarray]:
        '''
        step 1. generate mask
            - target foreground mask
            - perlin noise mask
            
        step 2. generate texture or structure anomaly
            - texture: load DTD
            - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation, 
            and hue on the input image  𝐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid 
            and randomly arranged to obtain the disordered image  𝐼 
            
        step 3. blending image and anomaly source
        '''
        
        # step 1. generate mask
        img_size = img.shape[:-1] # H x W
        
        ## target foreground mask
        if self.use_mask:
            target_foreground_mask = self.generate_target_foreground_mask(img=img)
        else:
            target_foreground_mask = np.ones(self.resize)
        
        ## perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask(img_size=img_size)
        
        ## mask
        mask = perlin_noise_mask * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)
        # step 2. generate texture or structure anomaly
        ## anomaly source
        anomaly_source_img = self.anomaly_source(img=img, texture_img_list=texture_img_list)
        
        ## mask anomaly parts
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
        
        # step 3. blending image and anomaly source
        anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img
        
        return (anomaly_source_img.astype(np.uint8), mask)
    
    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(img_gray, self.bg_threshold, 255, cv2.THRESH_BINARY)
        target_background_mask = target_background_mask.astype(np.bool_).astype(np.int32)

        # invert mask for foreground mask
        if self.bg_reverse:
            target_foreground_mask = target_background_mask
        else:
            target_foreground_mask = -(target_background_mask - 1)
        # print(self.bg_reverse)
        # print(self.bg_threshold)
        return target_foreground_mask
    
    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise        
        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))
        
        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)
        
        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold, 
            np.ones_like(perlin_noise), 
            np.zeros_like(perlin_noise)
        )
        
        return mask_noise
    
    def anomaly_source(self, img: np.ndarray, texture_img_list: list = None) -> np.ndarray:
        p = np.random.uniform() if texture_img_list else 1.0
        if p < 1:
            idx = np.random.choice(len(texture_img_list))
            img_size = img.shape[:-1] # H x W
            print(texture_img_list[idx])
            anomaly_source_img = self._texture_source(img_size=img_size, texture_img_path=texture_img_list[idx])
        else:
            anomaly_source_img = self._structure_source(img=img)
            
        return anomaly_source_img
        
    def _texture_source(self, img_size: tuple, texture_img_path: str) -> np.ndarray:
        texture_source_img = cv2.imread(texture_img_path)
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img, dsize=img_size).astype(np.float32)
        
        return texture_source_img
        
    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        structure_source_img = self.rand_augment()(image=img)
        
        img_size = img.shape[:-1] # H x W
        
        assert img_size[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = img_size[1] // self.structure_grid_size
        grid_h = img_size[0] // self.structure_grid_size
        
        structure_source_img = rearrange(
            tensor  = structure_source_img, 
            pattern = '(h gh) (w gw) c -> (h w) gw gh c',
            gw      = grid_w, 
            gh      = grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor  = structure_source_img[disordered_idx], 
            pattern = '(h w) gw gh c -> (h gh) (w gw) c',
            h       = self.structure_grid_size,
            w       = self.structure_grid_size
        ).astype(np.float32)
        
        return structure_source_img
        
    def __len__(self):
        return len(self.file_list)
    
class MVTecDataset_AD(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        aug_dir,
        origin_dir,
        mask_dir,
        resize_shape=[288, 288],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        imagesize=256,
        percent=0.5
    ):
        super().__init__()
        with counter_lock:
            counter.value = 0
        self.percent = percent
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.category = category
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.CenterCrop(imagesize),
                    transforms.ToTensor(),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)
#         self.aug_image_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/bottle_aug'+"/*.png"))
#         self.origin_image_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/bottle_origin'+"/*.png"))
#         self.mask_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/bottle_mask'+"/*.png"))
        self.aug_image_path = sorted(glob.glob(aug_dir+"/*.jpg"))
        self.origin_image_path = sorted(glob.glob(origin_dir+"/*.jpg"))
        self.mask_path = sorted(glob.glob(mask_dir+"/*.jpg"))
        print(self.aug_image_path[0])
        print(self.origin_image_path[0])
    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        index = torch.randint(0, len(self.aug_image_path), (1,)).item()
        image = Image.open(self.origin_image_path[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)
        image_path = self.origin_image_path[index]
        aug_image = Image.open(self.aug_image_path[index]).convert("RGB")
        aug_image = aug_image.resize(self.resize_shape, Image.BILINEAR)
        aug_mask = Image.open(self.mask_path[index]).convert("L")
        aug_mask = aug_mask.resize(self.resize_shape, Image.BILINEAR)
        # perlin_noise implementation
#         aug_image, aug_mask = cut_paste_normal(image, area_ratio=[0.02,0.15], aspect_ratio=0.3, color_jitter=None)
#         aug_mask = aug_mask/255
#         aug_mask = aug_mask.squeeze()
        aug_image = np.array(aug_image)
        aug_mask = np.array(aug_mask)
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly<self.percent:
#             Image.fromarray((aug_image).astype(np.uint8)).convert('RGB').save("CutPaste_aug111.png")
#             Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=1
            aug_image = self.final_preprocessing(aug_image.astype(np.uint8))
            aug_mask = self.transform_mask(aug_mask).squeeze()
            return aug_image,aug_mask,target,image_path
        else:
            aug_mask = np.zeros((256,256), np.float32)
#             image.save("CutPaste_aug.png")
#             Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=0
            image = np.array(image).astype(np.uint8)
            aug_image = self.final_preprocessing(image)
            aug_mask = self.transform_mask(aug_mask).squeeze()
            return aug_image,aug_mask,target,image_path
        
        
class MVTecDataset_DFMGAN(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        aug_dir,
        origin_dir,
        mask_dir,
        resize_shape=[288, 288],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        imagesize=256,
        percent=0.5
    ):
        super().__init__()
        with counter_lock:
            counter.value = 0
        self.percent = percent
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.category = category
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.CenterCrop(imagesize),
                    transforms.ToTensor(),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)
#         self.aug_image_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/bottle_aug'+"/*.png"))
#         self.origin_image_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/bottle_origin'+"/*.png"))
#         self.mask_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/bottle_mask'+"/*.png"))
        self.aug_image_path = sorted(glob.glob(aug_dir+"/*.png"))
        self.origin_image_path = sorted(glob.glob(origin_dir+"/*.png"))
        self.mask_path = sorted(glob.glob(mask_dir+"/*.png"))
        print(self.aug_image_path[0])
        print(self.origin_image_path[0])
    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        index = torch.randint(0, len(self.aug_image_path), (1,)).item()
        image = Image.open(self.origin_image_path[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)
        image_path = self.origin_image_path[index]
        aug_image = Image.open(self.aug_image_path[index]).convert("RGB")
        aug_image = aug_image.resize(self.resize_shape, Image.BILINEAR)
        aug_mask = Image.open(self.mask_path[index]).convert("L")
        aug_mask = aug_mask.resize(self.resize_shape, Image.BILINEAR)
        # perlin_noise implementation
#         aug_image, aug_mask = cut_paste_normal(image, area_ratio=[0.02,0.15], aspect_ratio=0.3, color_jitter=None)
#         aug_mask = aug_mask/255
#         aug_mask = aug_mask.squeeze()
        aug_image = np.array(aug_image)
        aug_mask = np.array(aug_mask)
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly<self.percent:
#             Image.fromarray((aug_image).astype(np.uint8)).convert('RGB').save("CutPaste_aug111.png")
#             Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=1
            aug_image = self.final_preprocessing(aug_image.astype(np.uint8))
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path
        else:
            aug_mask = np.zeros((256,256), np.float32)
#             image.save("CutPaste_aug.png")
#             Image.fromarray((aug_mask*255).astype(np.uint8)).save("CutPaste_mask.png")
            target=0
            image = np.array(image).astype(np.uint8)
            aug_image = self.final_preprocessing(image)
            aug_mask = self.transform_mask(aug_mask)
            return aug_image,aug_mask,target,image_path