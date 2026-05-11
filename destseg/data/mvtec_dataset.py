import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import imgaug.augmenters as iaa
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from einops import rearrange
from data.data_utils import perlin_noise
from multiprocessing import Value, Lock
from data.cutpaste import CutPasteNormal,cut_paste_normal,cut_paste_scar
from data.CutOut import Cutout
from data.NSA_generation import patch_ex
from data.FPI_generation import synthesize_anomalies_pil
from data.Fractal_Aug import *
from torchvision.transforms import ToTensor, Normalize, Compose, ToPILImage, RandomHorizontalFlip

from data.perlin import rand_perlin_2d_np
from data.utils import torch_seed

from typing import Union, List, Tuple
import matplotlib.pyplot as plt
counter = Value('i',0)
counter_lock = Lock()
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        percent=0.5,
    ):
        super().__init__()
        with counter_lock:
            counter.value = 0
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.category = category
        self.percent = percent
        if is_train:
            # self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            # self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            # self.rotate_90 = rotate_90
            # self.random_rotate = random_rotate
            print(f"训练模式，mvtec_dir: {mvtec_dir}")
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            print(f"找到的mvtec文件数量: {len(self.mvtec_paths)}")
            print(f"dtd_dir: {dtd_dir}")
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            print(f"找到的dtd文件数量: {len(self.dtd_paths)}")
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)
    

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)
        image_path = self.mvtec_paths[index]
        if self.is_train:
            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

            # perlin_noise implementation
            aug_image, aug_mask = perlin_noise(image, dtd_image, aug_prob=self.percent)
            # print(type(aug_image))
            # print(aug_image.shape)
            # print(aug_mask.shape)
            # temp_counter = 0
            # with counter_lock:
            #     temp_counter = counter.value
            #     counter.value += 1
            #     if(temp_counter<=1000):
            #         temp_counter=temp_counter+1
            #         save_folder1 = 'Perlin_test11/'+self.category
            #         # save_folder2 = 'datasets/test11/Perlin_test11_mask/'+self.category
            #         if not os.path.exists(save_folder1):
            #             os.makedirs(save_folder1)
            #         if not os.path.exists(save_folder2):
            #             os.makedirs(save_folder2)
            #         filename1 = f'generation_image_{temp_counter}.jpg'
            #         filename2 = f'mask_image_{temp_counter}.jpg'
            #         filename3 = f'origin_image_{temp_counter}.jpg'
            #         file_path1 = os.path.join(save_folder1, filename1)
            #         file_path2 = os.path.join(save_folder1, filename2)
            #         file_path3 = os.path.join(save_folder1, filename3)
            #         plt.imsave(file_path1, aug_image)
            #         mask_saved = aug_mask.squeeze()
            #         mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
            #         mask_saved.save(file_path2)
            #         image.save(file_path3)
            # if(np.sum(aug_mask)==0):
            #     print("0")
            # else:
            #     print("1")
            aug_image = self.final_preprocessing(aug_image)            
            image = self.final_preprocessing(image)
            return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
        else:
            image = self.final_preprocessing(image)
            dir_path, file_name = os.path.split(self.mvtec_paths[index])
            base_dir = os.path.basename(dir_path)
            if base_dir == "good":
                mask = torch.zeros_like(image[:1])
            else:
                mask_path = os.path.join(dir_path, "../../ground_truth/")
                mask_path = os.path.join(mask_path, base_dir)
                mask_file_name = file_name.split(".")[0] + "_mask.png"
                if 'BTAD' in mask_path:
                    mask_file_name = file_name.split(".")[0] + ".png"
                mask_path = os.path.join(mask_path, mask_file_name)
                mask = Image.open(mask_path)
                mask = self.mask_preprocessing(mask)
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )
            return {"img": image, "mask": mask,"image_path":image_path}

        
class MVTecDataset_CutOut(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        percent = 0.5,
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
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        print(self.mvtec_paths[index])
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
        dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

        cutout = Cutout(n_holes=1, length=random.randint(20, 50))
        aug_image,aug_mask = cutout(image)
        # temp_counter = 0
        # with counter_lock:
        #     temp_counter = counter.value
        #     counter.value += 1
        # #aug_image, aug_mask = perlin_noise(image, dtd_image, aug_prob=1.0)
        #     if(temp_counter<=1000):
        #         temp_counter=temp_counter+1
        #         save_folder1 = 'datasets/test11/CutOut_test11/'+self.category
        #         save_folder2 = 'datasets/test11/CutOut_test11_mask/'+self.category
        #         if not os.path.exists(save_folder1):
        #             os.makedirs(save_folder1)
        #         if not os.path.exists(save_folder2):
        #             os.makedirs(save_folder2)
        #         filename1 = f'generation_image_{temp_counter}.jpg'
        #         filename2 = f'mask_image_{temp_counter}.jpg'
        #         file_path1 = os.path.join(save_folder1, filename1)
        #         file_path2 = os.path.join(save_folder2, filename2)
        #         plt.imsave(file_path1, aug_image)
        #         mask_saved = aug_mask.squeeze()
        #         mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
        #         mask_saved.save(file_path2)
        no_anomaly = torch.rand(1).numpy()[0]
        aug_image = self.final_preprocessing(aug_image)
        image = self.final_preprocessing(image)
        if(no_anomaly<self.percent):
            return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
        else:
            aug_mask = np.zeros((256,256), np.float32)
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return{"img_aug":image,"img_origin":image,"mask":aug_mask}
        
class MVTecDataset_CutPaste_Scar(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        percent = 0.5,
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
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
        dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

        aug_image, aug_mask = cut_paste_scar(image, width=[2, 16], height=[10, 25], rotation=[-30, 30])
        aug_image = np.array(aug_image).astype(np.float32)
        aug_image = aug_image/255
        aug_mask = aug_mask/255
        temp_counter = 0
        with counter_lock:
            temp_counter = counter.value
            counter.value += 1
            if(temp_counter<=1000):
                temp_counter=temp_counter+1
                save_folder1 = 'datasets/test11/CutPaste_Scar_test11/'+self.category
                save_folder2 = 'datasets/test11/CutPaste_Scar_test11_mask/'+self.category
                if not os.path.exists(save_folder1):
                    os.makedirs(save_folder1)
                if not os.path.exists(save_folder2):
                    os.makedirs(save_folder2)
                filename1 = f'generation_image_{temp_counter}.jpg'
                filename2 = f'mask_image_{temp_counter}.jpg'
                file_path1 = os.path.join(save_folder1, filename1)
                file_path2 = os.path.join(save_folder2, filename2)
                plt.imsave(file_path1, aug_image)
                mask_saved = aug_mask.squeeze()
                mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
                mask_saved.save(file_path2)
        no_anomaly = torch.rand(1).numpy()[0]
        aug_image = self.final_preprocessing(aug_image)
        image = self.final_preprocessing(image)
        if(no_anomaly<self.percent):
            return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
        else:
            aug_mask = np.zeros((256,256), np.float32)
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return{"img_aug":image,"img_origin":image,"mask":aug_mask}
        

class MVTecDataset_NSA(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
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
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        if self.is_train:
            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

            # perlin_noise implementation
            aug_image, aug_mask = patch_ex(image,self.category)
            aug_image = np.array(aug_image).astype(np.float32)
            aug_mask = aug_mask.transpose(2,0,1)
            # print(aug_image.shape)
            # print(aug_mask.shape)
            aug_image = aug_image/255

            temp_counter = 0
            with counter_lock:
                temp_counter = counter.value
                counter.value += 1
                if(temp_counter<=1000):
                    temp_counter=temp_counter+1
                    save_folder1 = 'datasets/test11/NSA_test11/'+self.category
                    save_folder2 = 'datasets/test11/NSA_test11_mask/'+self.category
                    if not os.path.exists(save_folder1):
                        os.makedirs(save_folder1)
                    if not os.path.exists(save_folder2):
                        os.makedirs(save_folder2)
                    filename1 = f'generation_image_{temp_counter}.jpg'
                    filename2 = f'mask_image_{temp_counter}.jpg'
                    file_path1 = os.path.join(save_folder1, filename1)
                    file_path2 = os.path.join(save_folder2, filename2)
                    plt.imsave(file_path1, aug_image)
                    mask_saved = aug_mask.squeeze()
                    mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
                    mask_saved.save(file_path2)
            no_anomaly = torch.rand(1).numpy()[0]
            aug_image = self.final_preprocessing(aug_image)
            image = self.final_preprocessing(image)
            if(no_anomaly<self.percent):
                return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
            else:
                aug_mask = np.zeros((256,256), np.float32)
                aug_mask = np.expand_dims(aug_mask, axis=0)
                return{"img_aug":image,"img_origin":image,"mask":aug_mask}

        
class MVTecDataset_FPI(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        percent=1,
        rotate_90=False,
        random_rotate=0,
    ):
        super().__init__()
        with counter_lock:
            counter.value = 0
        self.resize_shape = resize_shape
        self.percent = percent
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
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        if self.is_train:
            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

            # perlin_noise implementation
            aug_image,aug_mask = synthesize_anomalies_pil(image)
            aug_image = np.array(aug_image).astype(np.float32)
            # print(aug_image.shape)
            # print(aug_mask.shape)
            aug_image = aug_image/255
            # aug_mask = aug_mask/255
            temp_counter = 0
            with counter_lock:
                temp_counter = counter.value
                counter.value += 1
                if(temp_counter<=1000):
                    temp_counter=temp_counter+1
                    save_folder1 = 'datasets/test11/FPI_test11/'+self.category
                    save_folder2 = 'datasets/test11/FPI_test11_mask/'+self.category
                    if not os.path.exists(save_folder1):
                        os.makedirs(save_folder1)
                    if not os.path.exists(save_folder2):
                        os.makedirs(save_folder2)
                    filename1 = f'generation_image_{temp_counter}.jpg'
                    filename2 = f'mask_image_{temp_counter}.jpg'
                    file_path1 = os.path.join(save_folder1, filename1)
                    file_path2 = os.path.join(save_folder2, filename2)
                    plt.imsave(file_path1, aug_image)
                    mask_saved = aug_mask.squeeze()
                    mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
                    mask_saved.save(file_path2)
            aug_image = self.final_preprocessing(aug_image)
            
            image = self.final_preprocessing(image)
            no_anomaly = torch.rand(1).numpy()[0]
            if(no_anomaly<self.percent):########在10.11号之前是大于
                return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
            else:
                aug_mask = np.zeros((256,256), np.float32)
                aug_mask = np.expand_dims(aug_mask, axis=0)
                return{"img_aug":image,"img_origin":image,"mask":aug_mask}

class MemSegDataset(Dataset):
    def __init__(self, datadir: str,percent: float, target: str, is_train: bool,category,
                to_memory: bool=False, resize: Tuple[int, int]=(224, 224),imagesize: int = 224,
                texture_source_dir: str=None, structure_grid_size: str=8,
                transparency_range: List[float] =[0.15, 1.],
                perlin_scale: int=6, min_perlin_scale: int=0, 
                perlin_noise_threshold: float=0.5, use_mask: bool = True, bg_threshold: float = 100, bg_reverse: bool = False):
        
        # Mode
        with counter_lock:
            counter.value = 0
        self.is_train = is_train 
        self.percent = percent
        self.to_memory = to_memory
        self.category = category
        # load image file list
        self.datadir = datadir
        self.target = target
        self.file_list = glob.glob(os.path.join(self.datadir, self.target, 'train/*/*' if is_train else 'test/*/*'))
        
        # synthetic anomaly
        if self.is_train and not self.to_memory:
            self.texture_source_file_list = glob.glob(os.path.join(texture_source_dir,'*/*')) if texture_source_dir else None
            self.perlin_scale = perlin_scale
            self.min_perlin_scale = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold
            
            self.structure_grid_size = structure_grid_size
            
            self.transparency_range = transparency_range
            
            self.use_mask = use_mask
            self.bg_threshold = bg_threshold
            self.bg_reverse = bg_reverse

        # transform
        self.resize = list(resize)
        self.transform_img = [
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
        # Synthetic anomaly switch
        self.anomaly_switch = True

    def __getitem__(self, idx):
        # print(self.use_mask)
        # if self.target=="hazelnut":
        #     self.use_mask=True,
        #     self.bg_threshold=50,
        #     self.bg_reverse=True
        # elif self.target=="leather"or self.target=="tile"or self.target =="wood"or self.target=="grid"or self.target=="carpet":
        #     self.use_mask=False,
        #     self.bg_threshold=None,
        #     self.bg_reverse=None
        # elif self.target=="metal_nut":
        #     self.use_mask=True,
        #     self.bg_threshold=40,
        #     self.bg_reverse=True
        # elif self.target=="cable":
        #     self.use_mask=False,
        #     self.bg_threshold=150,
        #     self.bg_reverse=True
        # elif self.target=="capsule":
        #     self.use_mask=True,
        #     self.bg_threshold=120,
        #     self.bg_reverse=False
        # elif self.target=="transistor":
        #     self.use_mask=True,
        #     self.bg_threshold=90,
        #     self.bg_reverse=False
        # elif self.target=="bottle":
        #     self.use_mask=True,
        #     self.bg_threshold=250,
        #     self.bg_reverse=False
        # elif self.target=="screw":
        #     self.use_mask=True,
        #     self.bg_threshold=110,
        #     self.bg_reverse=False
        # elif self.target=="zipper":
        #     self.use_mask=True,
        #     self.bg_threshold=100,
        #     self.bg_reverse=False
        # elif self.target=="pill":
        #     self.use_mask=True,
        #     self.bg_threshold=100,
        #     self.bg_reverse=True
        # elif self.target=="toothbrush":
        #     self.use_mask=True,
        #     self.bg_threshold=30,
        #     self.bg_reverse=True
        self.use_mask = True, 
        self.bg_threshold = 100, 
        self.bg_reverse = False
        file_path = self.file_list[idx]
        img = Image.open(file_path).convert("RGB").resize(self.resize)
        img = np.array(img)
        
        # target
        target = 0 if 'good' in self.file_list[idx] else 1
        
        # mask
        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            mask = Image.open(file_path.replace('test','ground_truth').replace('.png','_mask.png')).resize(self.resize)
            mask = np.array(mask)
        
        no_anomaly = torch.rand(1).numpy()[0]
        if self.is_train and not self.to_memory:
            if no_anomaly < self.percent:
                # print("anomaly!")
                aug_img, mask = self.generate_anomaly(img=img, texture_img_list=self.texture_source_file_list)
                target = 1
                mask = np.expand_dims(mask, axis=0)
                temp_counter = 0
                with counter_lock:
                    temp_counter = counter.value
                    counter.value += 1
                    if(temp_counter<=1000):
                        temp_counter=temp_counter+1
                        save_folder1 = 'datasets/test11/MemSeg_BTAD'+'/'+self.category
                        save_folder2 = 'datasets/test11/MemSeg_BTAD_mask'+'/'+self.category
                        if not os.path.exists(save_folder1):
                            os.makedirs(save_folder1)
                        if not os.path.exists(save_folder2):
                            os.makedirs(save_folder2)
                        filename1 = f'generation_image_{temp_counter}.jpg'
                        filename2 = f'mask_image_{temp_counter}.jpg'
                        file_path1 = os.path.join(save_folder1, filename1)
                        file_path2 = os.path.join(save_folder2, filename2)
                        plt.imsave(file_path1, aug_img)
                        mask_saved = mask.squeeze()
                        mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
                        mask_saved.save(file_path2)
                    elif temp_counter==1001:
                        print("1000 finished.")
                aug_img = self.transform_img(aug_img)
                img = self.transform_img(img)
                return {"img_aug": aug_img, "img_origin": img, "mask": mask}
            else:        
                # print("no_anomaly!")
                mask = np.expand_dims(mask, axis=0)
                img = self.transform_img(img)
                # mask = torch.Tensor(mask).to(torch.int64)
                return {"img_aug": img, "img_origin": img, "mask": mask}


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
        if self.use_mask[0]:
            target_foreground_mask = self.generate_target_foreground_mask(img=img)
            # print(self.use_mask[0])
            # print("use_mask")
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
        
        _, target_background_mask = cv2.threshold(img_gray, self.bg_threshold[0], 255, cv2.THRESH_BINARY)
        target_background_mask = target_background_mask.astype(np.bool).astype(np.int)

        # invert mask for foreground mask
        if self.bg_reverse:
            target_foreground_mask = target_background_mask
        else:
            target_foreground_mask = -(target_background_mask - 1)
        
        return target_foreground_mask
    
    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise        
        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))
        
        # apply affine transform
        # rot = iaa.Affine(rotate=(-90, 90))
        # perlin_noise = rot(image=perlin_noise)
        
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
        structure_source_img = img
        
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
    
class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, category, percent, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.percent = percent
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.category = category
        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
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

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = Image.open(anomaly_source_path).convert("RGB")
        anomaly_source_img = anomaly_source_img.resize(self.resize_shape, Image.BILINEAR)

        # anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr =np.array(anomaly_source_img).astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > self.percent:
            print("wrong for Multi")
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        # image = cv2.imread(image_path)
        # image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        image = np.array(image).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        #augmented_image = np.transpose(augmented_image, (2, 0, 1))
        #image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        # print(image.shape)
        # print(augmented_image.shape)
        # print(anomaly_mask.shape)
        temp_counter = 0
        with counter_lock:
            temp_counter = counter.value
            counter.value += 1

            # if(temp_counter<=5000):
            #     temp_counter=temp_counter+1
            #     save_folder1 = 'datasets/DRAEM_xxc/'+self.category
            #     save_folder2 = 'datasets/DRAEM_xxc_mask/'+self.category
            #     if not os.path.exists(save_folder1):
            #         os.makedirs(save_folder1)
            #     if not os.path.exists(save_folder2):
            #         os.makedirs(save_folder2)
            #     filename1 = f'generation_image_{temp_counter}.jpg'
            #     filename2 = f'mask_image_{temp_counter}.jpg'
            #     file_path1 = os.path.join(save_folder1, filename1)
            #     file_path2 = os.path.join(save_folder2, filename2)
            #     plt.imsave(file_path1, augmented_image)
            #     mask_saved = anomaly_mask.squeeze()
            #     mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
            #     mask_saved.save(file_path2)
            # elif temp_counter==1001:
            #     print("1000 finished.")
        augmented_image = self.final_preprocessing(augmented_image)
        image = self.final_preprocessing(image)
 # return {"img_aug": img, "img_origin": img, "mask": mask}
        return {"img_aug": augmented_image, "img_origin":image, "mask":anomaly_mask}

class MVTecDataset_Fractal(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
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
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        if self.is_train:
            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

            # perlin_noise implementation
            fag = FAG(load_size=256)
            aug_image, aug_mask = fag(image)
            aug_image = np.array(aug_image).astype(np.float32)
            aug_image = aug_image/255
            aug_mask = aug_mask/255

            temp_counter = 0
            with counter_lock:
                temp_counter = counter.value
                counter.value += 1
                if(temp_counter<=1000):
                    temp_counter=temp_counter+1
                    save_folder1 = 'datasets/test11/Fractal_BTAD/'+self.category
                    save_folder2 = 'datasets/test11/Fractal_BTAD_mask/'+self.category
                    if not os.path.exists(save_folder1):
                        os.makedirs(save_folder1)
                    if not os.path.exists(save_folder2):
                        os.makedirs(save_folder2)
                    filename1 = f'generation_image_{temp_counter}.jpg'
                    filename2 = f'mask_image_{temp_counter}.jpg'
                    file_path1 = os.path.join(save_folder1, filename1)
                    file_path2 = os.path.join(save_folder2, filename2)
                    plt.imsave(file_path1, aug_image)
                    mask_saved = aug_mask.squeeze()
                    mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
                    mask_saved.save(file_path2)
            aug_image = self.final_preprocessing(aug_image)
            image = self.final_preprocessing(image)
            # return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
            no_anomaly = torch.rand(1).numpy()[0]
            if(no_anomaly<self.percent): ########在10.11号之前是大于
                return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
            else:
                aug_mask = np.zeros((256,256), np.float32)
                aug_mask = np.expand_dims(aug_mask, axis=0)
                return{"img_aug":image,"img_origin":image,"mask":aug_mask}
            
class MVTecDataset_CutPaste(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        category,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        percent=0.5,
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
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
        dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

        aug_image, aug_mask = cut_paste_normal(image, area_ratio=[0.02,0.15], aspect_ratio=0.3, color_jitter=None)
        aug_image = aug_image/255
        aug_mask = aug_mask/255
        temp_counter = 0
        with counter_lock:
            temp_counter = counter.value
            counter.value += 1
            if(temp_counter<=1000):
                temp_counter=temp_counter+1
                save_folder1 = 'datasets/test11/CutPaste_test11/'+self.category
                save_folder2 = 'datasets/test11/CutPaste_test11_mask/'+self.category
                if not os.path.exists(save_folder1):
                    os.makedirs(save_folder1)
                if not os.path.exists(save_folder2):
                    os.makedirs(save_folder2)
                filename1 = f'generation_image_{temp_counter}.jpg'
                filename2 = f'mask_image_{temp_counter}.jpg'
                file_path1 = os.path.join(save_folder1, filename1)
                file_path2 = os.path.join(save_folder2, filename2)
                plt.imsave(file_path1, aug_image)
                mask_saved = aug_mask.squeeze()
                mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
                mask_saved.save(file_path2)
        aug_image = self.final_preprocessing(aug_image)
        image = self.final_preprocessing(image)
        no_anomaly = torch.rand(1).numpy()[0]
        if(no_anomaly<self.percent):
            return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
        else:
            aug_mask = np.zeros((256,256), np.float32)
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return{"img_aug":image,"img_origin":image,"mask":aug_mask}
        
class MemSegDataset_RealNet(Dataset):
    def __init__(self, datadir: str,percent: float, target: str, is_train: bool,category,
                to_memory: bool=False, resize: Tuple[int, int]=(256, 256),imagesize: int = 256,
                texture_source_dir: str=None, structure_grid_size: str=8,
                transparency_range: List[float] =[0.15, 1.],
                perlin_scale: int=6, min_perlin_scale: int=0, 
                perlin_noise_threshold: float=0.5, use_mask: bool = True, bg_threshold: float = 100, bg_reverse: bool = False):
        
        # Mode
        with counter_lock:
            counter.value = 0
        self.is_train = is_train 
        self.percent = percent
        self.to_memory = to_memory
        self.category = category
        # load image file list
        self.datadir = datadir
        self.target = target
        self.file_list = glob.glob(os.path.join(self.datadir, self.target, 'train/*/*' if is_train else 'test/*/*'))
        #print(self.file_list)
        # synthetic anomaly
        if self.is_train and not self.to_memory:
            self.texture_source_file_list = glob.glob(os.path.join(texture_source_dir,self.category,'*')) if texture_source_dir else None
            self.perlin_scale = perlin_scale
            self.min_perlin_scale = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold
            
            self.structure_grid_size = structure_grid_size
            
            self.transparency_range = transparency_range
            
            self.use_mask = use_mask
            self.bg_threshold = bg_threshold
            self.bg_reverse = bg_reverse
            
        # transform
        self.resize = list(resize)
        self.transform_img = [
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
        # Synthetic anomaly switch
        self.anomaly_switch = True

    def __getitem__(self, idx):
        # if self.target=="hazelnut":
        #     self.use_mask=True,
        #     self.bg_threshold=50,
        #     self.bg_reverse=True
        # elif self.target=="leather"or self.target=="tile"or self.target =="wood"or self.target=="grid"or self.target=="carpet":
        #     self.use_mask=False,
        #     self.bg_threshold=None,
        #     self.bg_reverse=None
        # elif self.target=="metal_nut":
        #     self.use_mask=True,
        #     self.bg_threshold=40,
        #     self.bg_reverse=True
        # elif self.target=="cable":
        #     self.use_mask=False,
        #     self.bg_threshold=150,
        #     self.bg_reverse=True
        # elif self.target=="capsule":
        #     self.use_mask=True,
        #     self.bg_threshold=120,
        #     self.bg_reverse=False
        # elif self.target=="transistor":
        #     self.use_mask=True,
        #     self.bg_threshold=90,
        #     self.bg_reverse=False
        # elif self.target=="bottle":
        #     self.use_mask=True,
        #     self.bg_threshold=250,
        #     self.bg_reverse=False
        # elif self.target=="screw":
        #     self.use_mask=True,
        #     self.bg_threshold=110,
        #     self.bg_reverse=False
        # elif self.target=="zipper":
        #     self.use_mask=True,
        #     self.bg_threshold=100,
        #     self.bg_reverse=False
        # elif self.target=="pill":
        #     self.use_mask=True,
        #     self.bg_threshold=100,
        #     self.bg_reverse=True
        # elif self.target=="toothbrush":
        #     self.use_mask=True,
        #     self.bg_threshold=30,
        #     self.bg_reverse=True
        self.use_mask = True, 
        self.bg_threshold = 100, 
        self.bg_reverse = False
        file_path = self.file_list[idx]
        img = Image.open(file_path).convert("RGB").resize(self.resize)
        img = np.array(img)
        
        # target
        target = 0 if 'good' in self.file_list[idx] else 1
        
        # mask
        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            mask = Image.open(file_path.replace('test','ground_truth').replace('.png','_mask.png')).resize(self.resize)
            mask = np.array(mask)
        no_anomaly = torch.rand(1).numpy()[0]
        if self.is_train and not self.to_memory:
            if no_anomaly < self.percent:
                # print("anomaly!")
                aug_img, mask = self.generate_anomaly(img=img, texture_img_list=self.texture_source_file_list)
                target = 1
                mask = np.expand_dims(mask, axis=0)
                temp_counter = 0
                with counter_lock:
                    temp_counter = counter.value
                    counter.value += 1
                    if(temp_counter<=5000):
                        temp_counter=temp_counter+1
                        save_folder1 = 'datasets/RealNet2_xxc/'+self.category
                        save_folder2 = 'datasets/RealNet2_xxc_mask/'+self.category
                        if not os.path.exists(save_folder1):
                            os.makedirs(save_folder1)
                        if not os.path.exists(save_folder2):
                            os.makedirs(save_folder2)
                        filename1 = f'generation_image_{temp_counter}.jpg'
                        filename2 = f'mask_image_{temp_counter}.jpg'
                        file_path1 = os.path.join(save_folder1, filename1)
                        file_path2 = os.path.join(save_folder2, filename2)
                        plt.imsave(file_path1, aug_img)
                        mask_saved = mask.squeeze()
                        mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
                        mask_saved.save(file_path2)
                    elif temp_counter==1001:
                        print("1000 finished.")
                    aug_img = self.transform_img(aug_img)
                    img = self.transform_img(img)
                    return {"img_aug": aug_img, "img_origin": img, "mask": mask}
            else:        
                # print("no_anomaly!")
                mask = np.expand_dims(mask, axis=0)
                img = self.transform_img(img)
                # mask = torch.Tensor(mask).to(torch.int64)
                return {"img_aug": img, "img_origin": img, "mask": mask}


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
        if self.use_mask[0]:
            # print("use_mask")
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
        _, target_background_mask = cv2.threshold(img_gray, self.bg_threshold[0], 255, cv2.THRESH_BINARY)
        target_background_mask = target_background_mask.astype(np.bool).astype(np.int)

        # invert mask for foreground mask
        if self.bg_reverse:
            target_foreground_mask = target_background_mask
        else:
            target_foreground_mask = -(target_background_mask - 1)
        
        return target_foreground_mask
    
    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise        
        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))
        
        # apply affine transform
        # rot = iaa.Affine(rotate=(-90, 90))
        # perlin_noise = rot(image=perlin_noise)
        
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
        structure_source_img = img
        
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
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        ad_dir=None,
        ad_mask_dir=None,
        ad_ori_dir=None,
        rotate_90=False,
        random_rotate=0,
        percent=0.5,
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
            self.ad_paths = sorted(glob.glob(ad_dir + "/*.jpg"))
            self.ad_ori_paths = sorted(glob.glob(ad_ori_dir + "/*.jpg"))
            self.ad_mask_paths = sorted(glob.glob(ad_mask_dir + "/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)
    

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        aug_index = torch.randint(0, len(self.ad_paths), (1,)).item()
        aug_ori_image = Image.open(self.ad_ori_paths[aug_index]).convert("RGB")
        aug_image = Image.open(self.ad_paths[aug_index]).convert("RGB")
        aug_mask = Image.open(self.ad_mask_paths[aug_index]).convert("L")
        aug_mask = np.array(aug_mask)/255
        aug_image = np.array(aug_image)
        temp_counter = 0
        with counter_lock:
            temp_counter = counter.value
            counter.value += 1
            if(temp_counter<=1000):
                temp_counter=temp_counter+1
                save_folder1 = 'datasets/test11/AD_test11/'+self.category
                save_folder2 = 'datasets/test11/AD_test11_mask/'+self.category
                if not os.path.exists(save_folder1):
                    os.makedirs(save_folder1)
                if not os.path.exists(save_folder2):
                    os.makedirs(save_folder2)
                filename1 = f'generation_image_{temp_counter}.jpg'
                filename2 = f'mask_image_{temp_counter}.jpg'
                file_path1 = os.path.join(save_folder1, filename1)
                file_path2 = os.path.join(save_folder2, filename2)
                plt.imsave(file_path1, aug_image)
                mask_saved = Image.fromarray((aug_mask * 255).astype(np.uint8))
                mask_saved.save(file_path2)
        aug_image = self.final_preprocessing(aug_image)            
        image = self.final_preprocessing(image)
        aug_ori_image = self.final_preprocessing(aug_ori_image)
        no_anomaly = torch.rand(1).numpy()[0]
        # print(image.shape,aug_image.shape,aug_ori_image.shape,aug_mask.shape)
        if(no_anomaly<self.percent):
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return {"img_aug": aug_image, "img_origin": aug_ori_image, "mask": aug_mask}
        else:
            aug_mask = np.zeros((256,256), np.float32)
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return{"img_aug":image,"img_origin":image,"mask":aug_mask}