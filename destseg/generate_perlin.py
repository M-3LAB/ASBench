import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Value, Lock
from data.data_utils import perlin_noise

# Initialize counter and lock for image saving
counter = Value('i', 0)
counter_lock = Lock()

def save_perlin_images(image, aug_image, aug_mask, dtd_image, category, save_folder='Perlin_test11'):
    """
    Save the original image, augmented image, mask, and DTD image.
    Args:
        image (PIL.Image): Original image.
        aug_image (np.ndarray): Augmented image generated using Perlin noise.
        aug_mask (np.ndarray): Mask corresponding to the augmented image.
        dtd_image (PIL.Image): DTD texture image.
        category (str): Category name for saving in a subfolder.
        save_folder (str): Base folder to save the images.
    """
    with counter_lock:
        temp_counter = counter.value
        counter.value += 1
    if temp_counter <= 200:
        save_folder_path = os.path.join(save_folder, category)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        # File paths
        filename1 = f'generation_image_{temp_counter}.jpg'
        filename2 = f'mask_image_{temp_counter}.jpg'
        filename3 = f'origin_image_{temp_counter}.jpg'
        filename4 = f'dtd_image_{temp_counter}.jpg'  # New filename for DTD image
        file_path1 = os.path.join(save_folder_path, filename1)
        file_path2 = os.path.join(save_folder_path, filename2)
        file_path3 = os.path.join(save_folder_path, filename3)
        file_path4 = os.path.join(save_folder_path, filename4)  # New file path for DTD image
        # Save augmented image
        plt.imsave(file_path1, aug_image)
        # Save mask
        mask_saved = aug_mask.squeeze()
        mask_saved = Image.fromarray((mask_saved * 255).astype(np.uint8))
        mask_saved.save(file_path2)
        # Save original image
        image.save(file_path3)
        # Save DTD image
        dtd_image.save(file_path4)

def generate_and_save_perlin(image_path, dtd_image_path, category, percent=1, resize_shape=(256, 256)):
    """
    Generate Perlin noise-based augmented image and mask, and save the results.
    Args:
        image_path (str): Path to the original image.
        dtd_image_path (str): Path to the DTD texture image.
        category (str): Category name for saving in a subfolder.
        percent (float): Probability of augmentation.
        resize_shape (tuple): Size to resize the images.
    """
    # Load and resize images
    image = Image.open(image_path).convert("RGB")
    image = image.resize(resize_shape, Image.BILINEAR)
    dtd_image = Image.open(dtd_image_path).convert("RGB")
    dtd_image = dtd_image.resize(resize_shape, Image.BILINEAR)
    # Generate Perlin noise-based augmentation
    aug_image, aug_mask = perlin_noise(image, dtd_image, aug_prob=percent)
    # Save the images
    save_perlin_images(image, aug_image, aug_mask, dtd_image, category)

def main():
    # Define paths
    mvtec_dir = "/cluster/home/zqyeleven/destseg_perlin/datasets/mvtec/hazelnut/train/good/"
    dtd_dir = "/cluster/home/zqyeleven/destseg_perlin/datasets/dtd/images/*/"
    category = "good"  # Set the category name
    # Get all MVTec and DTD image paths
    mvtec_image_paths = sorted(glob.glob(os.path.join(mvtec_dir, "*.png")))
    dtd_image_paths = sorted(glob.glob(os.path.join(dtd_dir, "*.jpg")))
    # Ensure there are images to process
    if not mvtec_image_paths or not dtd_image_paths:
        print("No images found in the specified directories.")
        return
    # Iterate through MVTec images and randomly select DTD images for augmentation
    for mvtec_image_path in mvtec_image_paths:
        dtd_image_path = np.random.choice(dtd_image_paths)  # Randomly select a DTD image
        generate_and_save_perlin(mvtec_image_path, dtd_image_path, category)

if __name__ == "__main__":
    main()