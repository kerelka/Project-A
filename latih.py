import Cascade as cas
import AdaBoost as ab
import IntegralImage as ii
import Utils as ul
import json


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__

#path data training
positif_data_training = 'train/faces'
negatif_data_training = 'train/non-faces'

#path data testing
positif_data_testing = 'testset/faces'
negatif_data_testing = 'testset/non-faces'

#define level cascade on list
level_cascade = [2,10,20,20,30,30,50,50,50,50,60,60,80,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]

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
features = ab.create_features(24,24,min_feature_height=4,max_feature_height=10,min_feature_width=4,max_feature_width=10)

#cascade => stage of bunch classifiers, alpha => weights every classifier
cascade = cas.cascade_latih(faces_ii_data,non_faces_ii_data,features, level_cascade)

database = []

for casc in cascade:
    for item in casc:
        database.append(item)

with open('database.json','w') as f:
    json.dump(database,f,default=dumper,indent=4)