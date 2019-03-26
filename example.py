import cv2 as cv
import numpy as np
from PIL import Image
import Utils as ul
import Deteksi as dt
import Cascade as cas
from Segmentasi import Segmentasi
from mergerect import mergeRects

#pathGambar
path = 'gambar.jpg'

#load feature and create cascade stage
features = ul.load_database('database.json')
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

# faces = mergeRects(faces)

for x,y,w,h in faces:
    cv.rectangle(image,(y,x),(y+h,x+w),(255,255,0),2)

cv.imshow('gambar',image)
cv.waitKey(0)



