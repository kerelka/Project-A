import cv2 as cv
import numpy as np
from PIL import Image
import Utils as ul
import Deteksi as dt
import Cascade as cas
from Segmentasi import Segmentasi
from mergerect import mergeRects

#pathGambar
path = 'image_0045.jpg'

#load feature and create cascade stage
features = ul.load_database('database.json')
print(str(len(features)))
cascade = cas.cascade_load(features)
#imread
image = cv.imread(path)
height = image.shape[0]
width = image.shape[1]

print(str(height)+','+str(width))

print('Segmentasi gambar...')
gambar = Segmentasi(image, height, width)
Seg = gambar.segmentasi_warna()

#convert YCbCr to BGR
print('convert ke gray...')
normal = cv.cvtColor(Seg, cv.COLOR_YCrCb2BGR)
gray = cv.cvtColor(normal, cv.COLOR_BGR2GRAY)

faces = dt.detect(gray,cascade)

# faces = mergeRects(faces,min_overlap_cnt=2,overlap_rate=0.3)
faces = mergeRects(faces)
for x,y,w,h in faces:
    cv.rectangle(gray,(x,y),(x+w,y+h),(255,255,0),2)

cv.imshow('gambar',gray)
cv.waitKey(0)



