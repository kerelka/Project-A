import numpy as np
from PIL import Image
import os

def load_images(path):
    images = []
    for file in os.listdir(path):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_arr = np.array(Image.open(os.path.join(path, file)), dtype=np.float64)
            images.append(img_arr)
    return images


#path data training
positif_data_training = 'trainset/faces'
negatif_data_training = 'trainset/non-faces'

#load data training
print('Load data training positif...')
faces_data = load_images(positif_data_training)
print(str(len(faces_data))+' Has been loaded.\nLoad data training negatif...')
non_faces_data = load_images(negatif_data_training)

print(str(len(non_faces_data))+' Has been Loaded.')