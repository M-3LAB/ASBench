import numpy as np
import random
from PIL import Image

def calc_distance(xy0, xy1):
    delta_X = (xy0[0] - xy1[0])**2
    delta_Y = (xy0[1] - xy1[1])**2
    return (delta_X + delta_Y)**0.5

def create_mask(im, center, radius):
    dims = im.shape
    mask = np.zeros(dims[:2])  # 注意这里只创建一个二维数组
    for i in range(dims[0]):
        for j in range(dims[1]):
            dist_i = calc_distance([i, j], center)
            if dist_i < radius:
                mask[i, j] = 1
    # 将 mask 转换为 (256, 256, 1) 的形状
    mask = mask[..., np.newaxis]
    return mask


def synthesize_anomalies_pil(pil_img):
    # 将PIL图像转换为numpy数组
    
    im = np.array(pil_img)

    
    dims = im.shape
    core = np.array(dims[:2]) / 2  # 核心区域的宽度（只考虑前两个维度）
    offset = core / 2  # 核心区域的中心偏移
    
    min_radius = np.round(0.05 * dims[0])
    max_radius = np.round(0.10 * dims[0])
    
    center = [np.random.randint(offset[i], offset[i] + core[i]) for i in range(2)]
    radius = np.random.randint(min_radius, max_radius)
    
    mask_i = create_mask(im, center, radius)
    
    intensity_range = np.max(im) - np.min(im)
    intensity = np.random.uniform(0.2 * intensity_range, 0.3 * intensity_range)
    if np.random.randint(2):  # 随机符号
        intensity *= -1
    
    add = mask_i * intensity
    im_out = im + add
    
    # 将结果转换回PIL图像
    im_out_pil = Image.fromarray(np.uint8(im_out.squeeze()))
    mask_i = mask_i.transpose(2, 0, 1)
    #print(mask_i.shape)
    #print(im_out.shape)
    return im_out_pil, mask_i

# # 调用函数
# image = Image.open('000.png')
# aug_image,aug_mask = synthesize_anomalies_pil(image)
# aug_image.save("FPI_generated.png")
