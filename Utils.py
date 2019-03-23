import os
import numpy as np
from PIL import Image

def load_images(path):
    images = []
    for file in os.listdir(path):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_arr = np.array(Image.open(os.path.join(path, file)), dtype=np.float64)
            images.append(img_arr)
    return images