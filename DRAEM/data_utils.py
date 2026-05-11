import math

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
import random

from torchvision import transforms

from PIL import Image,ImageDraw

"""The scripts here are copied from DRAEM: https://github.com/VitjanZ/DRAEM"""


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(
    shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3
):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: cv2.resize(
        np.repeat(
            np.repeat(
                gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]], d[0], axis=0
            ),
            d[1],
            axis=1,
        ),
        dsize=(shape[1], shape[0]),
    )
    dot = lambda grad, shift: (
        np.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            axis=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]
    )


rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


def perlin_noise(image, dtd_image, aug_prob=1.0):
    image = np.array(image, dtype=np.float32)
    dtd_image = np.array(dtd_image, dtype=np.float32)
    shape = image.shape[:2]
    min_perlin_scale, max_perlin_scale = 0, 6
    t_x = torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0]
    t_y = torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0]
    perlin_scalex, perlin_scaley = 2**t_x, 2**t_y

    perlin_noise = rand_perlin_2d_np(shape, (perlin_scalex, perlin_scaley))

    perlin_noise = rot(images=perlin_noise)
    perlin_noise = np.expand_dims(perlin_noise, axis=2)
    threshold = 0.5
    perlin_thr = np.where(
        perlin_noise > threshold,
        np.ones_like(perlin_noise),
        np.zeros_like(perlin_noise),
    )

    img_thr = dtd_image * perlin_thr / 255.0
    image = image / 255.0

    beta = torch.rand(1).numpy()[0] * 0.8
    image_aug = (
        image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
    )
    image_aug = image_aug.astype(np.float32)

    no_anomaly = torch.rand(1).numpy()[0]

    if no_anomaly > aug_prob:
        return image, np.zeros_like(perlin_thr)
    else:
        msk = (perlin_thr).astype(np.float32)
        msk = msk.transpose(2, 0, 1)

        return image_aug, msk

def cut_paste_collate_fn(batch):
    # cutPaste return 2 tuples of tuples we convert them into a list of tuples
    img_types = list(zip(*batch))
#     print(list(zip(*batch)))
    return [torch.stack(imgs) for imgs in img_types]
    

class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""
    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform
        
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness = colorJitter,
                                                      contrast = colorJitter,
                                                      saturation = colorJitter,
                                                      hue = colorJitter)
    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img
    
class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """
    def __init__(self, area_ratio=[0.02,0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        #TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        # h = img.size[0]
        # w = img.size[1]
        # print((img.shape))
        
        h = img.size[0]
        w = img.size[1]
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h
        
        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()
        
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))
        
        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)
        
        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))
        
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()
        augmented.paste(patch, insert_box)
        ##img = np.array(img)
        #augmented = np.array(augmented)
        #print(img.shape)
        return super().__call__(img, augmented)

class CutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """
    def __init__(self, width=[2,16], height=[10,25], rotation=[-45,45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation
    
    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        
        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)
        
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg,expand=True)
        
        #paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")
        
        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)
        img = np.array(img)
        augmented = np.array(augmented)
        # print(img.shape)
        return super().__call__(img, augmented)
    
class CutPasteUnion(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)
    
    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)

class CutPaste3Way(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)
    
    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)
        
        return org, cutpaste_normal, cutpaste_scar



def cut_paste_normal(image, area_ratio, aspect_ratio, color_jitter=None):
    h, w = image.size
    ratio_area = random.uniform(area_ratio[0], area_ratio[1]) * w * h
    
    log_ratio = torch.log(torch.tensor((aspect_ratio, 1/aspect_ratio)))
    aspect = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
    
    cut_w = int(round(math.sqrt(ratio_area * aspect)))
    cut_h = int(round(math.sqrt(ratio_area / aspect)))
    
    # 随机选择裁剪区域的位置
    from_location_h = int(random.uniform(0, h - cut_h))
    from_location_w = int(random.uniform(0, w - cut_w))
    box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
    

    patch = image.crop(box)
    # patch.save('patch.png', 'PNG')
    # 如果有颜色抖动变换，则应用它
    if color_jitter:
        patch = color_jitter(patch)
    
    # 随机选择粘贴区域的位置
    to_location_h = int(random.uniform(0, h - cut_h))
    to_location_w = int(random.uniform(0, w - cut_w))
    insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
            # 创建一个与原图同size的单通道掩码
    mask = Image.new('L', (w, h), 0)  # 创建一个单通道的黑色图像
    mask.paste(Image.new('L', (cut_w, cut_h), 255), insert_box)  # 粘贴白色区域
    
    augmented_image = image.copy()
    augmented_image.paste(patch, insert_box)
    augmented_image.save('augmented_image.png', 'PNG')
    mask.save('mask.png', 'PNG')
    # 将PIL图像转换为numpy数组
    msk = np.array(mask).astype(np.float32)
    msk = np.expand_dims(msk, axis=0)

    return augmented_image, msk

def cut_paste_scar(image, width, height, rotation, color_jitter=None):
    # 获取图像的高度和宽度
    h, w = image.size
    #print(h,w)
    # 随机选择裁剪区域的宽度和高度
    cut_w = random.uniform(width[0], width[1])
    cut_h = random.uniform(height[0], height[1])
    #print("cut_w:", cut_w, "cut_h:", cut_h)  # 新增打印语句
    
    # 随机选择裁剪区域的位置
    from_location_h = int(random.uniform(0, h - cut_h))
    from_location_w = int(random.uniform(0, w - cut_w))
    box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
    
    # 裁剪图像
    patch = image.crop(box)
    
    # 如果有颜色抖动变换，则应用它
    if color_jitter:
        patch = color_jitter(patch)
    
    # 随机选择旋转角度
    rot_deg = random.uniform(rotation[0], rotation[1])
    # 旋转图像，并确保旋转后的图像填充背景
    patch = patch.convert("RGBA").rotate(rot_deg, expand=True)
    patch.save('scar_patchmask.png','PNG')
    # 获取旋转后的图像尺寸
    new_w, new_h = patch.size
    #print("new_w:", new_w, "new_h:", new_h)
    # 随机选择粘贴区域的位置
    to_location_h = int(random.uniform(0, h - new_h))
    to_location_w = int(random.uniform(0, w - new_w))
    insert_box = [to_location_w, to_location_h, to_location_w + new_w, to_location_h + new_h]
    #print("insert_box:", insert_box)  # 新增打印语句
    # 创建一个与原图同size的单通道掩码
    mask = Image.new('L', (w, h), 0)  # 创建一个单通道的黑色图像
    mask.save('scar_mask1.png', 'PNG')
    white_area = Image.new('L', (int(new_w), int(new_h)), 255)  # 新增：保存要粘贴的白色区域图像
    white_area.save('scar_white.png', 'PNG')
    #print("white_area size:", white_area.size)  # 新增打印语句
    mask.paste(white_area, insert_box)  # 粘贴白色区域
    # 裁剪图像
    # 创建一个新的图像以进行粘贴操作
    augmented_image = image.copy()
    # 使用掩码进行粘贴
    augmented_image.paste(patch, insert_box)
    augmented_image.save('scar_augmented_image.png', 'PNG')
    mask.save('scar_mask.png', 'PNG')
    # 将PIL图像转换为numpy数组
    
    msk = np.array(mask).astype(np.float32)
    msk = np.expand_dims(msk, axis=0)
    #print(msk.shape)
    # 返回增强后的图像和掩码
    return augmented_image, msk


# 使用示例：
# image = Image.open("your_image.jpg")
# augmented_image, mask = cut_paste_scar(image, width=[2, 16], height=[10, 25], rotation=[-45, 45])
# augmented_image.show()  # 显示增强后的图像
def cut_out(image, area_ratio, aspect_ratio, color_jitter=None):
    # 获取图像的高度和宽度
    h, w = image.size
    
    # 随机选择一个区域大小
    ratio_area = random.uniform(area_ratio[0], area_ratio[1]) * w * h
    
    # 在对数空间中采样宽高比
    log_ratio = torch.log(torch.tensor((aspect_ratio, 1/aspect_ratio)))
    aspect = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
    
    # 计算裁剪区域的大小
    cut_w = int(round(math.sqrt(ratio_area * aspect)))
    cut_h = int(round(math.sqrt(ratio_area / aspect)))
    
    # 随机选择裁剪区域的位置
    from_location_h = int(random.uniform(0, h - cut_h))
    from_location_w = int(random.uniform(0, w - cut_w))
    box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
    
    # 创建一个与原图同size的单通道掩码
    mask = Image.new('L', (w, h), 0)  # 创建一个单通道的黑色图像
    mask.paste(Image.new('L', (cut_w, cut_h), 255), box)  # 粘贴白色区域
    
    # 创建一个新的图像以进行粘贴操作
    augmented_image = image.copy()
    # 将PIL图像转换为numpy数组
    msk = np.array(mask).astype(np.float32)
    msk = np.expand_dims(msk, axis=0)
    #print(msk.shape)
    
    # 返回增强后的图像和掩码
    return augmented_image, msk
