import numpy as np
from tqdm import tqdm

def learn(faces_ii_data, non_faces_ii_data, features, jumlah_classifier):

    jumlah_positif = len(faces_ii_data)
    jumlah_negatif = len(non_faces_ii_data)
    jumlah_image = jumlah_positif + jumlah_negatif

    images = faces_ii_data + non_faces_ii_data
    labels = np.hstack((np.ones(jumlah_positif),np.zeros(jumlah_negatif)))

    #inisialisasi bobot
    bobot_positif = (np.ones(jumlah_positif) * 1) / (2 * jumlah_positif)
    bobot_negatif = (np.ones(jumlah_negatif) * 1) / (2 * jumlah_negatif)
    bobot = np.hstack((bobot_positif, bobot_negatif))

    jumlah_fitur = len(features)

    classifiers = []
    alpha = []

    for _ in range(jumlah_classifier):

        #inisial error
        classification_errors = np.zeros(len(features))

        #normalisasi bobot
        bobot = bobot / np.sum(bobot)

        #cari error setiap feature
        for idx, feature in enumerate(tqdm(features,total=len(features))):
            error = sum(map(lambda img_idx: bobot[img_idx] if labels[img_idx] != vote(feature, images[img_idx]) else 0, range(jumlah_image)))
            classification_errors[idx] = error

        #pilih feature dengan error terkecil
        index_feature_pilihan = np.argmin(classification_errors)
        error_pilihan = classification_errors[index_feature_pilihan]
        feature_pilihan = features[index_feature_pilihan]

        #tambah weak classifier
        classifiers.append(feature_pilihan)

        #nilai beta
        """
            Bt = Et / (1-Et)
        """
        beta = error_pilihan / (1-error_pilihan)

        #nilai alpha => nilai bobot feature
        alpha.append(np.log(1/beta))

        #perbarui bobot images
        """
        Wt+1,i = Wt,i B^1-ei
        
        dimana ei = 0 untuk klasifikasi benar dan ei = 1 untuk lainnya
        
        maka bobot image[idx] = bobot[idx] jika klasifikasi salah dan bobot[idx] * beta jika benar
        """
        bobot = np.array(list(map(lambda img_idx: bobot[img_idx] if labels[img_idx] != vote(feature_pilihan,images[img_idx]) else bobot[img_idx] * beta, range(jumlah_image))))

        #buang feature pilihan dari list features
        del features[index_feature_pilihan]

    return classifiers, alpha

def vote(feature, image):
    return feature.get_vote(image)

def ensemble_vote(int_img, classifiers, alpha):

    return 1 if sum([alpha[i] * c.get_vote(int_img) for i,c in enumerate(classifiers)]) >= 0.5 * sum(alpha) else 0