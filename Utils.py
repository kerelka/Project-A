import os
import numpy as np
from PIL import Image
import json
from HaarLikeFeature import HaarLikeFeature

def load_images(path):
    images = []
    for file in os.listdir(path):
        if file.endswith('.pgm') or file.endswith('.png'):
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


def load_banned_index(path):
    banned_index = []

    with open(path,'r') as f:
        files = json.load(f)

    for file in files:
        banned_index.append(file)

    return banned_index


def load_features(features,num_classifier,idx):
    classifiers = []

    for i in range(num_classifier):
        # stage 1 cascade (2) = 0-1
        if idx == 1:
            classifiers.append(features[i])
        # stage 2 cascade (10) = 2-11
        elif idx == 2:
            classifiers.append(features[i + 2])
        # stage 3 cascade (20) = 12-31
        elif idx == 3:
            classifiers.append(features[i + 12])
        # stage 4 cascade (20) = 32-51
        elif idx == 4:
            classifiers.append(features[i + 32])
        # stage 5 cascade (30) = 52-81
        elif idx == 5:
            classifiers.append(features[i + 52])
        # stage 6 cascade (30) = 82-112
        elif idx == 6:
            classifiers.append(features[i + 82])
        else:
            break

    return classifiers