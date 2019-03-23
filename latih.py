import Cascade as cas
import AdaBoost as ab
import IntegralImage as ii
import Utils as ul


#path data training
positif_data_training = 'trainset/faces'
negatif_data_training = 'trainset/non-faces'

#path data testing
positif_data_testing = 'testset/faces'
negatif_data_testing = 'testset/non-faces'

#define level cascade on list
level_cascade = [2,10,20,20,50]

#load data training
print('Load data training positif...')
faces_data = ul.load_images(positif_data_training)
faces_ii_data = list(map(ii.to_integral_image,faces_data))
print(str(len(faces_ii_data))+' Has been loaded.\nLoad data training negatif...')
non_faces_data = ul.load_images(negatif_data_training)
non_faces_ii_data = list(map(ii.to_integral_image,non_faces_data))
print(str(len(non_faces_ii_data))+' Has been Loaded.')

img_height, img_width = faces_ii_data[0].shape

#create features
features = ab.create_features(19,19,min_feature_height=8,max_feature_height=10,min_feature_width=8,max_feature_width=10)

#cascade => stage of bunch classifiers, alpha => weights every classifier
cascade = cas.cascade_latih(faces_ii_data,non_faces_ii_data,features, level_cascade)


print('Load data testing positif...')
faces_data_testing = ul.load_images(positif_data_testing)
faces_ii_data_testing = list(map(ii.to_integral_image,faces_data_testing))
print(str(len(faces_ii_data_testing))+' Has been loaded.\nLoad data training negatif...')
non_faces_data_testing = ul.load_images(negatif_data_testing)
non_faces_ii_data_testing = list(map(ii.to_integral_image,non_faces_data_testing))
print(str(len(non_faces_ii_data_testing))+' Has been Loaded.')


correct_faces = 0
correct_non_faces = 0

for idx,classifier in enumerate(cascade):
    correct_faces = sum(ab.ensemble_vote_all(faces_ii_data_testing, classifier))
    correct_non_faces = len(non_faces_data_testing) - sum(ab.ensemble_vote_all(non_faces_ii_data_testing, classifier))

    print('Result after ' + str(idx+1) + ' layer(s):\n     Faces: ' + str(correct_faces) + '/' + str(len(faces_data_testing))
        + '  (' + str((float(correct_faces) / len(faces_data_testing)) * 100) + '%)\n  non-Faces: '
        + str(correct_non_faces) + '/' + str(len(non_faces_data_testing)) + '  ('
        + str((float(correct_non_faces) / len(non_faces_data_testing)) * 100) + '%)')