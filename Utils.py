import os
import numpy as np
from PIL import Image
import json
from HaarLikeFeature import HaarLikeFeature

def load_images(path):
    images = []
    for file in os.listdir(path):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_arr = np.array(Image.open(os.path.join(path, file)), dtype=np.float64)
            images.append(img_arr)
    return images


def load_database(path):

    features = []
    with open(path,'r') as f:
        files = json.load(f)

    for fitur in files:
        for t,xy in enumerate(fitur['type']):
            if t == 0:
                x = xy
            else:
                y = xy
        for p, posisi in enumerate(fitur['top_left']):
            if p == 0:
                j = posisi
            else:
                k = posisi
        features.append(HaarLikeFeature((x,y),(j,k),fitur['width'],fitur['height'],fitur['threshold'],fitur['polarity'],fitur['weight']))

    return features
