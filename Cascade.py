import numpy as np


def cascade_latih(faces_ii_data,non_faces_ii_data,features):
    cascade = []
    alpha = []

    images = faces_ii_data + non_faces_ii_data
    labels = np.hstack((np.ones(len(faces_ii_data)),np.zeros(len(non_faces_ii_data))))

    print(str(labels))


    return cascade, alpha