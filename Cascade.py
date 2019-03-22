import numpy as np
import AdaBoost as ab

def cascade_latih(faces_ii_data,non_faces_ii_data,features,level_cascade):
    cascade = []
    alphas = []

    images = faces_ii_data + non_faces_ii_data
    labels = np.hstack((np.ones(len(faces_ii_data)),np.zeros(len(non_faces_ii_data))))

    print('Mulai pelatihan attentional cascade ...')

    #pilih cascade
    for idx, classifier in enumerate(level_cascade):
        classifiers, alpha = ab.learn(faces_ii_data,non_faces_ii_data,features,classifier)
        cascade.append(classifiers)
        alphas.append(alpha)
        count = 0
        for img in enumerate(faces_ii_data):
            if ab.ensemble_vote(img,classifiers,alpha) == 1:
                count += 1

        print(str(count)+'/'+str(len(faces_ii_data)))

    return cascade, alphas






    return cascade, alpha