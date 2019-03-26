import Utils as ul
import IntegralImage as ii
import Cascade as cas
import AdaBoost as ab

pathDatabase = 'database.json'
positif_data_testing = 'trainset/faces'
negatif_data_testing = 'trainset/non-faces'

print('load faces data...')
faces_data = ul.load_images(positif_data_testing)
faces_ii_data = list(map(ii.to_integral_image,faces_data))
print(str(len(faces_ii_data))+' Has been loaded.\nload non faces data...')
non_faces_data = ul.load_images(negatif_data_testing)
non_faces_ii_data = list(map(ii.to_integral_image,non_faces_data))
print(str(len(non_faces_ii_data))+' Has been loaded.')

print('loaded database')
features = ul.load_database(pathDatabase)

print('create cascade stage')
cascade = cas.cascade_load(features)

correct_faces = 0
correct_non_faces = 0

for idx, classifiers in enumerate(cascade):
    correct_faces = sum(ab.ensemble_vote_all(faces_ii_data,classifiers))
    correct_non_faces = len(non_faces_data) - sum(ab.ensemble_vote_all(non_faces_ii_data,classifiers))

    print('Result after ' + str(idx + 1) + ' layer(s):\n     Faces: ' + str(correct_faces) + '/' + str(
        len(faces_data))
          + '  (' + str((float(correct_faces) / len(faces_data)) * 100) + '%)\n  non-Faces: '
          + str(correct_non_faces) + '/' + str(len(non_faces_data)) + '  ('
          + str((float(correct_non_faces) / len(non_faces_data)) * 100) + '%)')