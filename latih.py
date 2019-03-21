import numpy as np
from PIL import Image
from HaarLikeFeature import HaarLikeFeature, FeatureTypes
import os
import Cascade as cas

def load_images(path):
    images = []
    for file in os.listdir(path):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_arr = np.array(Image.open(os.path.join(path, file)), dtype=np.float64)
            images.append(img_arr)
    return images


def to_integral_image(img_arr):
    integral_array = np.zeros((img_arr.shape[0]+1,img_arr.shape[1]+1))

    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            integral_array[x+1,y+1] = img_arr[x,y] + integral_array[x+1,y] + integral_array[x,y+1] - integral_array[x,y]

    return integral_array


def create_features(img_height, img_width, min_feature_width = 1, max_feature_width = -1, min_feature_height = 1, max_feature_height = -1):
    features = []
    #define maximum feature
    max_feature_width = img_width if max_feature_width == -1 else max_feature_width
    max_feature_height= img_height if max_feature_height == -1 else max_feature_height

    #create every type, every scale, and every location of feature
    for feature in FeatureTypes:
        feature_start_width = max(min_feature_width, feature[0])
        for feature_width in range(feature_start_width,max_feature_width,feature[0]):
            feature_start_height = max(min_feature_height,feature[1])
            for feature_height in range(feature_start_height,max_feature_height,feature[1]):
                for x in range(img_width - feature_width):
                    for y in range(img_height - feature_height):
                        features.append(HaarLikeFeature(feature,(x,y),feature_width, feature_height, 0, 1))
                        features.append(HaarLikeFeature(feature,(x,y), feature_width, feature_height, 0, -1))

    print(str(len(features))+' Has been created.')

    return features



#path data training
positif_data_training = 'trainset/faces'
negatif_data_training = 'trainset/non-faces'

#load data training
print('Load data training positif...')
faces_data = load_images(positif_data_training)
faces_ii_data = list(map(to_integral_image,faces_data))
print(str(len(faces_ii_data))+' Has been loaded.\nLoad data training negatif...')
non_faces_data = load_images(negatif_data_training)
non_faces_ii_data = list(map(to_integral_image,non_faces_data))
print(str(len(non_faces_ii_data))+' Has been Loaded.')

img_height, img_width = faces_ii_data[0].shape

#create features
features = create_features(img_height=19,img_width=19)
#cascade => stage of bunch classifiers, alpha => weights every classifier
cascade, alpha = cas.cascade_latih(faces_ii_data,non_faces_ii_data,features)
