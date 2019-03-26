import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
from HaarLikeFeature import HaarLikeFeature, FeatureTypes


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


def learn(faces_ii_data, non_faces_ii_data, features, jumlah_classifier, votes, banned_index):

    jumlah_positif = len(faces_ii_data)
    jumlah_negatif = len(non_faces_ii_data)
    jumlah_image = jumlah_positif + jumlah_negatif

    images = faces_ii_data + non_faces_ii_data
    labels = np.hstack((np.ones(jumlah_positif),np.zeros(jumlah_negatif)))

    feature_index = list(range(len(features)))

    #inisialisasi bobot
    bobot_positif = (np.ones(jumlah_positif) * 1) / (2 * jumlah_positif)
    bobot_negatif = (np.ones(jumlah_negatif) * 1) / (2 * jumlah_negatif)
    bobot = np.hstack((bobot_positif, bobot_negatif))

    jumlah_fitur = len(features)
    classifiers = []

    for _ in range(jumlah_classifier):

        #inisial error
        classification_errors = np.zeros(len(features))

        #normalisasi bobot
        bobot = bobot / np.sum(bobot)

        #cari error setiap feature
        for f in tqdm(range(len(feature_index))):
            f_idx = feature_index[f]
            if f not in banned_index:
                error = sum(map(lambda img_idx: bobot[img_idx] if labels[img_idx] != votes[img_idx,f_idx] else 0, range(jumlah_image)))
                classification_errors[f] = error
            else:
                classification_errors[f] = 10

        #pilih feature dengan error terkecil
        min_error_index = np.argmin(classification_errors)
        error_pilihan = classification_errors[min_error_index]
        feature_pilihan_index = feature_index[min_error_index]
        feature_pilihan = features[feature_pilihan_index]

        #nilai beta
        """
            Bt = Et / (1-Et)
        """
        beta = error_pilihan / (1-error_pilihan)

        #nilai alpha => nilai bobot feature
        feature_weight = np.log(1/beta)
        feature_pilihan.weight = feature_weight
        classifiers.append(feature_pilihan)

        #perbarui bobot images
        """
        Wt+1,i = Wt,i B^1-ei
        
        dimana ei = 0 untuk klasifikasi benar dan ei = 1 untuk salah
        
        maka bobot image[idx] = bobot[idx] jika klasifikasi salah dan bobot[idx] * beta jika benar
        """
        bobot = np.array(list(map(lambda img_idx: bobot[img_idx] if labels[img_idx] != votes[img_idx,feature_pilihan_index] else bobot[img_idx] * beta, range(jumlah_image))))

        #buang feature pilihan dari list features
        banned_index.append(feature_pilihan_index)

    return classifiers, banned_index


def vote(feature, image):
    return feature.get_vote(image)


def ensemble_vote(int_img, classifiers):

    return 1 if sum([c.get_vote(int_img) * c.get_weight() for c in classifiers]) >= 0.5 * sum([clas.get_weight() for clas in classifiers]) else 0


def ensemble_vote_all(int_imgs, classifiers):

    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial,int_imgs))

def ensemble_vote_with_scalar(int_img, classifiers, top_left, scale):
    return 1 if sum([c.get_vote_with_scale(int_img, top_left, scale) * c.get_weight() for c in classifiers]) >= 0.5 * sum([clas.get_weight() for clas in classifiers]) else 0