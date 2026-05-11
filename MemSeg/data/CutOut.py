import numpy as np
from PIL import Image

class Cutout:
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): PIL image.
        Returns:
            PIL.Image: Image with n_holes of dimension length x length cut out of it.
        """
        img = img.resize((256,256))
        width, height = img.size
        mask = np.ones((height, width), np.float32)
        img = np.array(img)

        for n in range(self.n_holes):
            y = np.random.randint(height)
            x = np.random.randint(width)

            y1 = np.clip(y - self.length // 2, 0, height)
            y2 = np.clip(y + self.length // 2, 0, height)
            x1 = np.clip(x - self.length // 2, 0, width)
            x2 = np.clip(x + self.length // 2, 0, width)

            mask[y1:y2, x1:x2] = 0

        img = img * np.expand_dims(mask, axis=-1)
        img = img/255
        mask = 1 - mask
        mask = np.expand_dims(mask, axis=0)
        # print(img.shape)
        # print(img)
        # print(mask.shape)
        return img,mask


