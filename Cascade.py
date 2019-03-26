import numpy as np
import AdaBoost as ab
import progressbar
import Utils as ul

def cascade_latih(faces_ii_data,non_faces_ii_data,features,level_cascade):
    cascade = []
    banned_index = []

    images = faces_ii_data + non_faces_ii_data

    votes = calc_votes(features, images)
    correct_faces = 0
    correct_non_faces = 0

    print('Mulai pelatihan attentional cascade ...')

    #pilih cascade
    for idx, classifier in enumerate(level_cascade):
        print('Begin Training '+str(idx)+' layer :...')
        classifiers, banned_index = ab.learn(faces_ii_data, non_faces_ii_data, features, classifier, votes, banned_index)
        cascade.append(classifiers)
        #test classifiers
        correct_faces = sum(ab.ensemble_vote_all(faces_ii_data, classifiers))
        correct_non_faces = len(non_faces_ii_data) - sum(ab.ensemble_vote_all(non_faces_ii_data, classifiers))
        print('Result after ' + str(idx + 1) + ' layer(s):\n     Faces: ' + str(correct_faces) + '/' + str(
            len(faces_ii_data))
              + '  (' + str((float(correct_faces) / len(faces_ii_data)) * 100) + '%)\n  non-Faces: '
              + str(correct_non_faces) + '/' + str(len(non_faces_ii_data)) + '  ('
              + str((float(correct_non_faces) / len(non_faces_ii_data)) * 100) + '%)')

    return cascade


def calc_votes(features, images):
    jumlah_image = len(images)
    jumlah_feature = len(features)

    print(str(jumlah_image)+', '+str(jumlah_feature))
    votes = np.zeros((jumlah_image,jumlah_feature))

    bar = progressbar.ProgressBar()
    print('Mulai kalkulasi vote setiap image dengan seluruh features...')
    for i in bar(range(jumlah_image)):
        for j in range(jumlah_feature):
            votes[i,j] = ab.vote(image=images[i],feature=features[j])

    return votes


def cascade_load(features):
    level_cascade = [2,10,20,20,30]
    cascade = []

    for idx, num_classifier in enumerate(level_cascade):
        classifier = ul.load_features(features,num_classifier,idx+1)
        cascade.append(classifier)

    return cascade