import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import random
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from perlin import rand_perlin_2d_np
from data_utils import perlin_noise
from PIL import Image
from CutOut import Cutout
from Fractal_Aug import FAG
from NSA_generation import patch_ex
from FPI_generation import synthesize_anomalies_pil
from cutpaste import cut_paste_normal,cut_paste_scar
from typing import List
from torchvision import transforms
from einops import rearrange
# np.set_printoptions(threshold=np.inf)
class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        print(f"测试数据集中的样本数:", len(self.images))
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            if 'BTAD' in mask_path:
                mask_file_name = file_name.split(".")[0]+".png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx,'image_path':img_path}

        return sample

class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, percent,resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.percent = percent
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
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > self.percent:
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
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
            'augmented_image': augmented_image} #, 'has_anomaly': has_anomaly, 'idx': idx
#         image_saved = np.transpose(image, (1,2,0))
#         augmented_image_saved = np.transpose(augmented_image, (1,2,0))
#         image_saved = Image.fromarray((image_saved*255).astype(np.uint8))
#         image_saved.save('image.png')
#         augmented_image_saved = Image.fromarray((augmented_image_saved * 255).astype(np.uint8))
#         augmented_image_saved = augmented_image_saved.convert('RGB') 
#         augmented_image_saved.save('image2.png')
#         mask_saved = anomaly_mask.squeeze()
#         mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
#         mask_saved = mask_saved.convert('RGB') 
#         mask_saved.save('mask.png')


        return sample

class MVTecDataset_DestSeg(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        percent,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
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
        self.percent=percent
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))

    def __len__(self):
        return len(self.mvtec_paths)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def __getitem__(self, index):
        # global counter
        aug = self.randAugmenter()
        image = cv2.imread(self.mvtec_paths[index])
        image = cv2.resize(image, dsize = (self.resize_shape[0], self.resize_shape[1]))

        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = cv2.imread(self.dtd_paths[dtd_index])
        dtd_image = cv2.resize(dtd_image, dsize = (self.resize_shape[0], self.resize_shape[1]))
        dtd_image = aug(image=dtd_image)

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)
            
        aug_image, aug_mask = perlin_noise(image, dtd_image,aug_prob=1)
        aug_mask = aug_mask.astype(np.float32)
        image = image/255
#         image_saved = np.array(image)
#         aug_image_saved=np.array(aug_image)
#         image_saved = Image.fromarray((image_saved*255).astype(np.uint8))
#         image_saved.save('destseg1.png')
#         augmented_image_saved = Image.fromarray((aug_image_saved * 255).astype(np.uint8))
#         augmented_image_saved = augmented_image_saved.convert('RGB') 
#         augmented_image_saved.save('destseg2.png')
#         mask_saved = aug_mask.squeeze()
#         mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
#         mask_saved = mask_saved.convert('RGB')
#         mask_saved.save('destseg3.png')
        aug_image = np.transpose(aug_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        aug_image = aug_image.astype(np.float32)
        image = image.astype(np.float32)

        no_anomaly = torch.rand(1).numpy()[0]
        if(no_anomaly<self.percent):
            return {"augmented_image": aug_image, "image": image, "anomaly_mask": aug_mask}
        else:
            aug_mask = np.zeros((256,256), np.float32)
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return{"augmented_image":image,"image":image,"anomaly_mask":aug_mask}
#         return {"augmented_image": aug_image, "image": image, "anomaly_mask": aug_mask}


class MVTecDataset_CutOut(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        percent=0.5,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.percent = percent
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))


    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        # global counter
        image = cv2.imread(self.mvtec_paths[index])
        image = cv2.resize(image,dsize=(self.resize_shape[0], self.resize_shape[1]))
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = cv2.imread(self.dtd_paths[dtd_index])
        dtd_image = cv2.resize(dtd_image,dsize=(self.resize_shape[0], self.resize_shape[1]))

        cutout = Cutout(n_holes=1, length=random.randint(20, 50))
        aug_image,aug_mask = cutout(image)

        aug_mask=aug_mask.astype(np.float32)
        image = image/255
#         image_saved = Image.fromarray((image*255).astype(np.uint8))
#         image_saved.save('imagedestseg.png')
#         augmented_image_saved = Image.fromarray((aug_image * 255).astype(np.uint8))
#         augmented_image_saved = augmented_image_saved.convert('RGB') 
#         augmented_image_saved.save('image2dest.png')
#         mask_saved = aug_mask.squeeze()
#         mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
#         mask_saved = mask_saved.convert('RGB') 
#         mask_saved.save('maskdest.png')
        aug_image = np.transpose(aug_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        aug_image = aug_image.astype(np.float32)
        image=image.astype(np.float32)

        no_anomaly = torch.rand(1).numpy()[0]
        if(no_anomaly < self.percent):
            return {"augmented_image": aug_image, "image": image, "anomaly_mask": aug_mask}
        else:
            aug_mask = np.zeros((256,256), np.float32)
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return{"augmented_image":image,"image":image,"anomaly_mask":aug_mask}


class MVTecDataset_Fractal(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        percent=0.5,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.percent = percent
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))


    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        # global counter
        image = cv2.imread(self.mvtec_paths[index])
        image = cv2.resize(image,dsize=(self.resize_shape[0], self.resize_shape[1]))

        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = cv2.imread(self.dtd_paths[dtd_index])
        dtd_image = cv2.resize(dtd_image,dsize=(self.resize_shape[0], self.resize_shape[1]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        fag = FAG(load_size=256)
        aug_image, aug_mask = fag(image)

        aug_mask=aug_mask.astype(np.float32)
        image = image/255
        aug_image = aug_image/255
        aug_mask= aug_mask/255
#         image_saved = Image.fromarray((image*255).astype(np.uint8))
#         image_saved.save('image destseg11.png')
#         augmented_image_saved = Image.fromarray((aug_image * 255).astype(np.uint8))
#         augmented_image_saved = augmented_image_saved.convert('RGB') 
#         augmented_image_saved.save('image2dest11.png')
#         mask_saved = aug_mask.squeeze()
#         mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
#         mask_saved = mask_saved.convert('RGB') 
#         mask_saved.save('maskdest111.png')
        aug_image = np.transpose(aug_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        aug_image = aug_image.astype(np.float32)
        image=image.astype(np.float32)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly < self.percent:
            return {"augmented_image": aug_image, "image": image, "anomaly_mask": aug_mask}
        else:
            aug_mask = np.zeros((256,256), np.float32)
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return{"augmented_image":image,"image":image,"anomaly_mask":aug_mask}

class MVTecDataset_FPI(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        percent = 0.5,
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.percent = percent
        self.is_train = is_train
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))


    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        # global counter
        image = cv2.imread(self.mvtec_paths[index])
        image = cv2.resize(image,dsize=(self.resize_shape[0], self.resize_shape[1]))


        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = cv2.imread(self.dtd_paths[dtd_index])
        dtd_image = cv2.resize(dtd_image,dsize=(self.resize_shape[0], self.resize_shape[1]))


        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        aug_image,aug_mask = synthesize_anomalies_pil(image)
        aug_image = np.array(aug_image).astype(np.float32)
        image = image/255
        aug_image=aug_image/255


#         image_saved = Image.fromarray((image*255).astype(np.uint8))
#         image_saved.save('image2FPI.png')
#         augmented_image_saved = Image.fromarray((aug_image * 255).astype(np.uint8))
#         augmented_image_saved = augmented_image_saved.convert('RGB') 
#         augmented_image_saved.save('image2FPI_aug.png')
#         mask_saved = aug_mask.squeeze()
#         mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
#         mask_saved = mask_saved.convert('RGB') 
#         mask_saved.save('mask2FPI.png')

        aug_image = np.transpose(aug_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        aug_image = aug_image.astype(np.float32)
        image=image.astype(np.float32)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly<self.percent:
            return {"augmented_image": aug_image, "image": image, "anomaly_mask": aug_mask}
        else:
            aug_mask = np.zeros((256,256), np.float32)
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return{"augmented_image":image,"image":image,"anomaly_mask":aug_mask}

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
        percent=0.5,
        rotate_90=False,
        random_rotate=0,
        
    ):
        super().__init__()
        self.percent = percent 
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.category = category
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))


    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        # global counter
        image = cv2.imread(self.mvtec_paths[index])
        image = cv2.resize(image, dsize=(self.resize_shape[0], self.resize_shape[1]))

        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = cv2.imread(self.dtd_paths[dtd_index])
        dtd_image = cv2.resize(dtd_image, dsize=(self.resize_shape[0], self.resize_shape[1]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        aug_image, aug_mask = patch_ex(image,target=self.category)
        image = image / 255
        aug_image=aug_image/255
#         image_saved = Image.fromarray((image*255).astype(np.uint8))
#         image_saved.save('image2NSA.png')
#         augmented_image_saved = Image.fromarray((aug_image * 255).astype(np.uint8))
#         augmented_image_saved = augmented_image_saved.convert('RGB') 
#         augmented_image_saved.save('image2NSA_aug.png')
#         mask_saved = aug_mask.squeeze()
#         mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
#         mask_saved = mask_saved.convert('RGB') 
#         mask_saved.save('mask2NSA.png')
        aug_image = aug_image.astype(np.float32)
        image=image.astype(np.float32)
        aug_mask = aug_mask.astype(np.float32)
        aug_image = np.transpose(aug_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly < self.percent:
            return {"augmented_image": aug_image, "image": image, "anomaly_mask": aug_mask}
        else:
            aug_mask = np.zeros((256,256), np.float32)
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return{"augmented_image":image,"image":image,"anomaly_mask":aug_mask}


class MVTecDataset_CutPaste(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        percent=0.5
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.percent = percent
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))


    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.mvtec_paths[index])
        image = cv2.resize(image,dsize=(self.resize_shape[0], self.resize_shape[1]))
        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = cv2.imread(self.dtd_paths[dtd_index])
        dtd_image = cv2.resize(dtd_image,dsize=(self.resize_shape[0], self.resize_shape[1]))
        
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = Image.fromarray(image)
        aug_image, aug_mask = cut_paste_normal(image, area_ratio=[0.02,0.15], aspect_ratio=0.3, color_jitter=None)
        aug_image = np.array(aug_image).astype(np.float32)
        image = np.array(image).astype(np.float32)
        image = image/255
        aug_image = aug_image/255
        aug_mask = aug_mask/255
        aug_mask = aug_mask.astype(np.float32)

#         image_saved = Image.fromarray((image*255).astype(np.uint8))
#         image_saved.save('image2cutpaste.png')
#         augmented_image_saved = Image.fromarray((aug_image * 255).astype(np.uint8))
#         augmented_image_saved = augmented_image_saved.convert('RGB') 
#         augmented_image_saved.save('image2cutpaste_aug.png')
#         mask_saved = aug_mask.squeeze()
#         mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
#         mask_saved = mask_saved.convert('RGB') 
#         mask_saved.save('mask2cutpaste.png')

        aug_image = np.transpose(aug_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly<self.percent:
            return {"augmented_image": aug_image, "image": image, "anomaly_mask": aug_mask}
        else:
            aug_mask = np.zeros((256,256), np.float32)
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return{"augmented_image":image,"image":image,"anomaly_mask":aug_mask}

class MVTecDataset_MemSeg(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, percent:float, to_memory: bool = False, 
        resize: List[int] = [256, 256], imagesize: int = 224,
        texture_source_dir: str = None, structure_grid_size: str = 8,
        transparency_range: List[float] = [0.15, 1.],
        perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
        use_mask: bool = True, bg_threshold: float = 100, bg_reverse: bool = False
    ):
        self.is_train = is_train 
        self.percent = percent 
        self.to_memory = to_memory
        self.datadir = datadir
        self.target = target
        self.file_list = glob.glob(os.path.join(self.datadir, self.target, 'train/*/*' if is_train else 'test/*/*'))
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
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
            
        self.resize = list(resize)
        self.anomaly_switch = True
        
    def __getitem__(self, idx):
#         if self.target=="hazelnut":
#             self.use_mask=True,
#             self.bg_threshold=50,
#             self.bg_reverse=True
#         elif self.target=="leather"or self.target=="tile"or self.target =="wood"or self.target=="grid"or self.target=="carpet":
#             self.use_mask=False,
#             self.bg_threshold=None,
#             self.bg_reverse=None
#         elif self.target=="metal_nut":
#             self.use_mask=True,
#             self.bg_threshold=40,
#             self.bg_reverse=True
#         elif self.target=="cable":
#             self.use_mask=False,
#             self.bg_threshold=150,
#             self.bg_reverse=True
#         elif self.target=="capsule":
#             self.use_mask=True,
#             self.bg_threshold=120,
#             self.bg_reverse=False
#         elif self.target=="transistor":
#             self.use_mask=True,
#             self.bg_threshold=90,
#             self.bg_reverse=False
#         elif self.target=="bottle":
#             self.use_mask=True,
#             self.bg_threshold=250,
#             self.bg_reverse=False
#         elif self.target=="screw":
#             self.use_mask=True,
#             self.bg_threshold=110,
#             self.bg_reverse=False
#         elif self.target=="zipper":
#             self.use_mask=True,
#             self.bg_threshold=100,
#             self.bg_reverse=False
#         elif self.target=="pill":
#             self.use_mask=True,
#             self.bg_threshold=100,
#             self.bg_reverse=True
#         elif self.target=="toothbrush":
#             self.use_mask=True,
#             self.bg_threshold=30,
#             self.bg_reverse=True
        self.use_mask = True,
        self.bg_threshold = 100,
        self.bg_reverse = False
        file_path = self.file_list[idx]
        img = Image.open(file_path).convert("RGB").resize(self.resize)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         image = Image.open(file_path).convert("RGB").resize(self.resize)
#         image = np.array(image)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            img = self.rot(image=img)
        target = 0 if 'good' in self.file_list[idx] else 1
        
        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            mask = cv2.imread(file_path.replace('test','ground_truth').replace('.png','_mask.png')).resize(self.resize)
            mask = np.array(mask)
            
        no_anomaly = torch.rand(1).numpy()[0]
        if(no_anomaly<self.percent):
            self.anomaly_switch = True
        else:
            self.anomaly_switch = False
        if self.is_train and not self.to_memory:
            if self.anomaly_switch:
                image, mask = self.generate_anomaly(img=img, texture_img_list=self.texture_source_file_list)
                target = 1
                image=image/255
#                 aug_image_saved = Image.fromarray((image*255).astype(np.uint8))
#                 aug_image_saved.save('augmented_mem_image.png', 'PNG') 
                image = np.transpose(image, (2, 0, 1))
                img=img/255

                # mask=mask/255
#                 img_saved=Image.fromarray((img*255).astype(np.uint8))
#                 img_saved.save('mem_image_ori.png')
#                 mask_saved=Image.fromarray(mask*255)
#                 mask_saved=mask_saved.convert('RGB')
#                 mask_saved.save('mem_mask.png')
                image = np.array(image).astype(np.float32)
                img = np.array(img).astype(np.float32)
                mask = np.array(mask)
                mask = np.expand_dims(mask, axis=0)
                img = np.transpose(img, (2, 0, 1))
                return {"augmented_image": image, "image": img, "anomaly_mask": mask}
            else:
                img=img/255
#                 img_saved=Image.fromarray((img*255).astype(np.uint8))
#                 img_saved.save('mem_image_ori_false.png')
#                 mask_saved=Image.fromarray(mask*255)
#                 mask_saved=mask_saved.convert('RGB')
#                 mask_saved.save('mem_mask_false.png')
                img = np.array(img).astype(np.float32)
                mask = np.array(mask)
                mask = np.expand_dims(mask, axis=0)
                img = np.transpose(img, (2, 0, 1))
                return {"augmented_image": img, "image": img, "anomaly_mask": mask}
        
        
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
        if self.use_mask[0]:
            target_foreground_mask = self.generate_target_foreground_mask(img=img)
#             print("use_mask")
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
        target_background_mask = target_background_mask.astype(np.bool_).astype(int)

        if self.bg_reverse:
            target_foreground_mask = target_background_mask
        else:
            target_foreground_mask = -(target_background_mask - 1)
        
        return target_foreground_mask
    
    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
  
        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))
        
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold, 
            np.ones_like(perlin_noise), 
            np.zeros_like(perlin_noise)
        )
        
        return mask_noise
    
    def anomaly_source(self, img: np.ndarray, texture_img_list: list = None) -> np.ndarray:
        p = np.random.uniform() if texture_img_list else 1.0
        ##########memseg改在这里改
        if p > 0.5:
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
    
class MVTecDataset_CutPaste_Scar(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        percent=0.5,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
        self.percent = percent
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))


    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.mvtec_paths[index])
        image = cv2.resize(image,dsize=(self.resize_shape[0], self.resize_shape[1]))
        dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
        dtd_image = cv2.imread(self.dtd_paths[dtd_index])
        dtd_image = cv2.resize(dtd_image,dsize=(self.resize_shape[0], self.resize_shape[1]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image=Image.fromarray(image)
        aug_image, aug_mask = cut_paste_scar(image, width=[2, 16], height=[10, 25], rotation=[-30, 30])
        aug_image = np.array(aug_image).astype(np.float32)
        image= np.array(image).astype(np.float32)
        image=image/255
        aug_image=aug_image/255
        aug_mask=aug_mask/255
        aug_mask=aug_mask.astype(np.float32)

#         image_saved = Image.fromarray((image*255).astype(np.uint8))
#         image_saved.save('image2cutpaste.png')
#         augmented_image_saved = Image.fromarray((aug_image * 255).astype(np.uint8))
#         augmented_image_saved = augmented_image_saved.convert('RGB') 
#         augmented_image_saved.save('image2cutpaste_aug.png')
#         mask_saved = aug_mask.squeeze()
#         mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
#         mask_saved = mask_saved.convert('RGB') 
#         mask_saved.save('mask2cutpaste.png')

        aug_image = np.transpose(aug_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly<self.percent:
            return {"augmented_image": aug_image, "image": image, "anomaly_mask": aug_mask}
        else:
            aug_mask = np.zeros((256,256), np.float32)
            aug_mask = np.expand_dims(aug_mask, axis=0)
            return{"augmented_image":image,"image":image,"anomaly_mask":aug_mask}

class MVTecDataset_RealNet(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, percent:float, to_memory: bool = False, 
        resize: List[int] = [256, 256], imagesize: int = 224,
        texture_source_dir: str = None, structure_grid_size: str = 8,
        transparency_range: List[float] = [0.15, 1.],
        perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
        use_mask: bool = True, bg_threshold: float = 100, bg_reverse: bool = False
    ):
        self.is_train = is_train 
        self.percent = percent 
        self.to_memory = to_memory
        self.datadir = datadir
        self.target = target
        self.file_list = glob.glob(os.path.join(self.datadir, self.target, 'train/*/*' if is_train else 'test/*/*'))
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        if self.is_train and not self.to_memory:   
            self.texture_source_file_list = glob.glob(os.path.join(texture_source_dir,target,'*')) if texture_source_dir else None
            self.perlin_scale = perlin_scale
            self.min_perlin_scale = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold
            self.structure_grid_size = structure_grid_size
            self.transparency_range = transparency_range
            self.use_mask = use_mask
            self.bg_threshold = bg_threshold
            self.bg_reverse = bg_reverse
            
        self.resize = list(resize)
        self.anomaly_switch = True
        
    def __getitem__(self, idx):

        self.use_mask=True,
        self.bg_threshold=100,
        self.bg_reverse=False

        file_path = self.file_list[idx]
        img = Image.open(file_path).convert("RGB").resize(self.resize)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         image = Image.open(file_path).convert("RGB").resize(self.resize)
#         image = np.array(image)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
#         do_aug_orig = torch.rand(1).numpy()[0] > 0.7
#         if do_aug_orig:
#             img = self.rot(image=img)
        target = 0 if 'good' in self.file_list[idx] else 1
        
        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            mask = cv2.imread(file_path.replace('test','ground_truth').replace('.png','_mask.png')).resize(self.resize)
            mask = np.array(mask)
            
        no_anomaly = torch.rand(1).numpy()[0]
        if(no_anomaly<self.percent):
            self.anomaly_switch = True
        else:
            self.anomaly_switch = False
        if self.is_train and not self.to_memory:
            if self.anomaly_switch:
                image, mask = self.generate_anomaly(img=img, texture_img_list=self.texture_source_file_list)
                target = 1
                image=image/255
#                 aug_image_saved = Image.fromarray((image*255).astype(np.uint8))
#                 aug_image_saved.save('augmented_mem_image.png', 'PNG') 
                image = np.transpose(image, (2, 0, 1))
                img=img/255

                # mask=mask/255
#                 img_saved=Image.fromarray((img*255).astype(np.uint8))
#                 img_saved.save('mem_image_ori.png')
#                 mask_saved=Image.fromarray(mask*255)
#                 mask_saved=mask_saved.convert('RGB')
#                 mask_saved.save('mem_mask.png')
                image = np.array(image).astype(np.float32)
                img = np.array(img).astype(np.float32)
                mask = np.array(mask)
                mask = np.expand_dims(mask, axis=0)
                img = np.transpose(img, (2, 0, 1))
                return {"augmented_image": image, "image": img, "anomaly_mask": mask}
            else:
                img=img/255
                img = np.array(img).astype(np.float32)
                mask = np.array(mask)
                mask = np.expand_dims(mask, axis=0)
                img = np.transpose(img, (2, 0, 1))
                return {"augmented_image": img, "image": img, "anomaly_mask": mask}
        
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
        if self.use_mask[0]:
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
        target_background_mask = target_background_mask.astype(np.bool_).astype(int)

        if self.bg_reverse:
            target_foreground_mask = target_background_mask
        else:
            target_foreground_mask = -(target_background_mask - 1)
        
        return target_foreground_mask
    
    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
  
        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))
        
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold, 
            np.ones_like(perlin_noise), 
            np.zeros_like(perlin_noise)
        )
        
        return mask_noise
    
    def anomaly_source(self, img: np.ndarray, texture_img_list: list = None) -> np.ndarray:
        p = np.random.uniform() if texture_img_list else 1.0
        ##########memseg改在这里改
        if p > 0:
            idx = np.random.choice(len(texture_img_list))
            img_size = img.shape[:-1] # H x W
#             print(texture_img_list[idx])
            anomaly_source_img = self._texture_source(img_size=img_size, texture_img_path=texture_img_list[idx])
        else:
            anomaly_source_img = self._structure_source(img=img)
            
        return anomaly_source_img
        
    def _texture_source(self, img_size: tuple, texture_img_path: str) -> np.ndarray:
        texture_source_img = cv2.imread(texture_img_path)
#         texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
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
    
class MVTec_Anomaly_Detection(Dataset):
    def __init__(self, args,sample_name,length=5000,anomaly_id=None,recon=False):
        self.recon=recon
        self.good_path='%s/%s/train/good'%('/opt/ml/code/luohan/ml-destseg-main/datasets/mvtec/',sample_name)
        self.good_files=[os.path.join(self.good_path,i) for i in os.listdir(self.good_path)]
        self.root_dir = '%s/%s'%('/opt/ml/code/luohan/anomalydiffusion-master/generated_dataset',sample_name)
        self.anomaly_names=os.listdir(self.root_dir)
        if anomaly_id!=None:
            self.anomaly_names=self.anomaly_names[anomaly_id:anomaly_id+1]
            print('training subsets',self.anomaly_names)
        l=len(self.anomaly_names)
        self.anomaly_num = l
        self.img_paths=[]
        self.mask_paths=[]
        for idx,anomaly in enumerate(self.anomaly_names):
            img_path=[]
            mask_path=[]
            for i in range(min(len(os.listdir(os.path.join(self.root_dir,anomaly,'mask'))),500)):
                img_path.append(os.path.join(self.root_dir,anomaly,'image','%d.jpg'%i))
                mask_path.append(os.path.join(self.root_dir,anomaly,'mask','%d.jpg'%i))
            self.img_paths.append(img_path.copy())
            self.mask_paths.append(mask_path.copy())
        for i in range(l):
            print(len(self.img_paths[i]),len(self.mask_paths[i]))
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256])
        ])
        self.length=length
        if self.length is None:
            self.length=len(self.good_files)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if random.random()>0.5:
            image=self.loader(Image.open(self.good_files[idx%len(self.good_files)]).convert('RGB'))
            mask=torch.zeros((1,image.size(-2),image.size(-1)))
            has_anomaly = np.array([0], dtype=np.float32)
            sample = {'image': image,'augmented_image':image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomay_id': -1}
        else:
            anomaly_id=random.randint(0,self.anomaly_num-1)
            img_path=self.img_paths[anomaly_id][idx% len(self.mask_paths[anomaly_id])]
            image = self.loader(Image.open(img_path).convert('RGB'))
            mask_path = self.mask_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
            mask = self.loader(Image.open(mask_path).convert('L'))
            mask=(mask>0.5).float()
            if mask.sum()==0:
                has_anomaly = np.array([0], dtype=np.float32)
                anomaly_id=-1
            else:
                has_anomaly = np.array([1], dtype=np.float32)
            sample = {'image': image, 'augmented_image':image,'has_anomaly': has_anomaly, 'mask': mask, 'anomay_id': anomaly_id}
            if self.recon:
                img_path = self.img_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
                img_path=img_path.replace('image','recon')
                ori_image = self.loader(Image.open(img_path).convert('RGB'))
                sample['source']=ori_image
        return sample



class MVTecDRAEMTrainDataset_AD(Dataset):

    def __init__(self, root_dir, anomaly_source_path,percent,aug_dir,origin_dir,mask_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

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
        self.percent = percent
#         self.aug_dir = aug_dir
#         self.origin_dir = origin_dir
#         self.mask_dir = mask_dir
#         self.aug_image_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/carpet_aug'+"/*.png"))
#         self.origin_image_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/carpet_origin'+"/*.png"))
#         self.mask_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/carpet_mask'+"/*.png"))
        self.aug_image_path = sorted(glob.glob(aug_dir+"/*.jpg"))
        self.origin_image_path = sorted(glob.glob(origin_dir+"/*.jpg"))
        self.mask_path = sorted(glob.glob(mask_dir+"/*.jpg"))
        print(self.aug_image_path[0])
        print(self.origin_image_path[0])
        print(self.mask_path[0])
    def __len__(self):
#         print(len(self.image_paths))
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.aug_image_path), (1,)).item()
#         print(len(aug_image_path))
        augmented_image = cv2.imread(self.aug_image_path[idx])
#         print("test:::::::::",self.aug_image_path[idx],self.origin_image_path[idx],self.mask_path[idx])
        image = cv2.imread(self.origin_image_path[idx])
#         anomaly_mask = cv2.imread(mask_path[idx], cv2.IMREAD_GRAYSCALE)
        anomaly_mask = Image.open(self.mask_path[idx]).convert("L")
        anomaly_mask = np.array(anomaly_mask).astype(np.uint8)
        anomaly_mask = anomaly_mask/255
        augmented_image = cv2.resize(augmented_image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        has_anomaly = 1
        
        augmented_image = np.array(augmented_image).astype(np.float32)
        image = np.array(image).astype(np.float32)
        augmented_image=augmented_image/255
        image = image /255
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.expand_dims(anomaly_mask, axis=0)
#         anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
#         print(image.shape)
#         print(augmented_image.shape)
#         print(anomaly_mask.shape)
        no_anomaly = torch.rand(1).numpy()[0]
        if(no_anomaly<self.percent):
            sample = {'image': image, "anomaly_mask": anomaly_mask,
            'augmented_image': augmented_image}
        else:
            anomaly_mask = np.zeros((256,256), np.float32)
            anomaly_mask = np.expand_dims(anomaly_mask, axis=0)
            sample = {'image': image, "anomaly_mask": anomaly_mask,
                'augmented_image': image}
        image_saved = np.transpose(image, (1,2,0))
        augmented_image_saved = np.transpose(augmented_image, (1,2,0))
        image_saved = Image.fromarray((image_saved*255).astype(np.uint8))
        image_saved.save('image.png')
        augmented_image_saved = Image.fromarray((augmented_image_saved * 255).astype(np.uint8))
        augmented_image_saved = augmented_image_saved.convert('RGB') 
        augmented_image_saved.save('image2.png')
        mask_saved = anomaly_mask.squeeze()
        mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
        mask_saved = mask_saved.convert('RGB') 
        mask_saved.save('mask.png')

        return sample
    
    
class MVTecDRAEMTrainDataset_DFMGAN(Dataset):

    def __init__(self, root_dir, anomaly_source_path,percent,aug_dir,origin_dir,mask_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

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
        self.percent = percent
#         self.aug_image_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/carpet_aug'+"/*.png"))
#         self.origin_image_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/carpet_origin'+"/*.png"))
#         self.mask_path = sorted(glob.glob('/opt/ml/code/luohan/DFM_data_new/carpet_mask'+"/*.png"))
        self.aug_image_path = sorted(glob.glob(aug_dir+"/*.png"))
        self.origin_image_path = sorted(glob.glob(origin_dir+"/*.png"))
        self.mask_path = sorted(glob.glob(mask_dir+"/*.png"))
        print(self.aug_image_path[0])
        print(self.origin_image_path[0])
    def __len__(self):
        return len(self.image_paths)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def __getitem__(self, idx):
        
        idx = torch.randint(0, len(self.aug_image_path), (1,)).item()
#         print(len(aug_image_path))
        augmented_image = cv2.imread(self.aug_image_path[idx])
        image = cv2.imread(self.origin_image_path[idx])
#         anomaly_mask = cv2.imread(mask_path[idx], cv2.IMREAD_GRAYSCALE)
        anomaly_mask = Image.open(self.mask_path[idx]).convert("L")
        anomaly_mask = np.array(anomaly_mask).astype(np.uint8)
        anomaly_mask = anomaly_mask/255
        augmented_image = cv2.resize(augmented_image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        has_anomaly = 1
        
        augmented_image = np.array(augmented_image).astype(np.float32)
        image = np.array(image).astype(np.float32)
        augmented_image=augmented_image/255
        image = image /255
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.expand_dims(anomaly_mask, axis=0)
#         anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
#         print(image.shape)
#         print(augmented_image.shape)
#         print(anomaly_mask.shape)
        no_anomaly = torch.rand(1).numpy()[0]
        if(no_anomaly<self.percent):
            sample = {'image': image, "anomaly_mask": anomaly_mask,
            'augmented_image': augmented_image}
        else:
            anomaly_mask = np.zeros((256,256), np.float32)
            anomaly_mask = np.expand_dims(anomaly_mask, axis=0)
            sample = {'image': image, "anomaly_mask": anomaly_mask,
                'augmented_image': image}
        image_saved = np.transpose(image, (1,2,0))
        augmented_image_saved = np.transpose(augmented_image, (1,2,0))
        image_saved = Image.fromarray((image_saved*255).astype(np.uint8))
        image_saved.save('image.png')
        augmented_image_saved = Image.fromarray((augmented_image_saved * 255).astype(np.uint8))
        augmented_image_saved = augmented_image_saved.convert('RGB') 
        augmented_image_saved.save('image2.png')
        mask_saved = anomaly_mask.squeeze()
        mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
        mask_saved = mask_saved.convert('RGB') 
        mask_saved.save('mask.png')

        return sample