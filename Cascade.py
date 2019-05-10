import numpy as np
import AdaBoost as ab
import Utils as ul
from tqdm import tqdm
import json


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


def cascade_latih(faces_ii_data,non_faces_ii_data,features,level_cascade):
    cascade = []
    start_stage = 35
    banned = True
    path_banned = 'bannen_index_stage34.json'
    if banned:
        features_stg = []
        banned_index = ul.load_banned_index(path_banned)
        for i in range(0,start_stage-1):
            features_stage = ul.load_database('database_stage' + str(i + 1) + '.json')
            for fitur in features_stage:
                features_stg.append(fitur)
    else:
        banned_index = []

    images = faces_ii_data + non_faces_ii_data

    votes = calc_votes(features, images)

    print('Mulai pelatihan attentional cascade ...')
    #pilih cascade
    for idx, classifier in enumerate(level_cascade):
        if start_stage == idx+1:
            print('Begin Training ' + str(idx + 1) + ' layer :...')
            classifiers, banned_index = ab.learn(faces_ii_data, non_faces_ii_data, features, classifier, votes, banned_index)
        else:
            classifiers = ul.load_features(features_stg,classifier,idx+1)
        cascade.append(classifiers)
        #test classifiers
        # correct_faces = sum(ab.ensemble_vote_all(faces_ii_data, classifiers))
        # correct_non_faces = len(non_faces_ii_data) - sum(ab.ensemble_vote_all(non_faces_ii_data, classifiers))
        # print('Result after ' + str(idx + 1) + ' layer(s):\n     Faces: ' + str(correct_faces) + '/' + str(
        #     len(faces_ii_data))
        #       + '  (' + str((float(correct_faces) / len(faces_ii_data)) * 100) + '%)\n  non-Faces: '
        #       + str(correct_non_faces) + '/' + str(len(non_faces_ii_data)) + '  ('
        #       + str((float(correct_non_faces) / len(non_faces_ii_data)) * 100) + '%)')

        database_stage = []
        for clas in classifiers:
            database_stage.append(clas)

        with open('database_stage'+str(idx+1)+'.json', 'w') as f:
            json.dump(database_stage, f, default=dumper, indent=4)

        with open('bannen_index_stage'+str(idx+1)+'.json','w') as b:
            json.dump(banned_index, b,default=dumper, indent=4)

    return cascade


def calc_votes(features, images):
    jumlah_image = len(images)
    jumlah_feature = len(features)

    votes = np.zeros((jumlah_image,jumlah_feature))

    for i in tqdm(range(jumlah_image)):
        for j in range(jumlah_feature):
            votes[i,j] = ab.vote(image=images[i],feature=features[j])

    return votes


def cascade_stage(faces_ii_data, non_faces_ii_data, features, banned_index, votes, stage):
    level_cascade = [2, 10, 20, 20, 30,
                     30, 50, 50, 50, 50,
                     60, 60, 80, 100, 100,
                     100, 100, 100, 100,
                     100, 100, 100, 100,
                     100, 100, 100, 100]
    database_stage = []
    jum_fitur = level_cascade[stage-1]

    classifiers, banned = ab.learn(faces_ii_data, non_faces_ii_data, features, jum_fitur, votes, banned_index)
    for clas in classifiers:
        database_stage.append(clas)
        print(str(clas))

    with open('database_stage' + str(stage) + '.json', 'w') as f:
        json.dump(database_stage, f, default=dumper, indent=4)

    with open('banned_index_stage' + str(stage) + '.json', 'w') as b:
        json.dump(banned, b, default=dumper, indent=4)

    return 1


def cascade_load(features):
    level_cascade = [2,10,20,20,30,30,50,50,50,50,60,60,80,100,100]
    cascade = []

    for idx, num_classifier in enumerate(level_cascade):
        classifier = ul.load_features(features,num_classifier,idx+1)
        cascade.append(classifier)

    return cascade