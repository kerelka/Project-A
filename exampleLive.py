import cv2 as cv
import Utils as ul
import Cascade as cas
import Deteksi as dt
from mergerect import mergeRects
from Segmentasi import Segmentasi
import numpy as np

cap = cv.VideoCapture(0)
features = ul.load_database('database.json')
cascade = cas.cascade_load(features)

while True:
    _, frame = cap.read()
    height, width = frame.shape[0], frame.shape[1]
    gambar = Segmentasi(frame, height, width)
    Seg = gambar.segmentasi_warna()

    # convert YCbCr to BGR
    print('convert ke gray...')
    normal = cv.cvtColor(Seg, cv.COLOR_YCrCb2BGR)
    gray = cv.cvtColor(normal, cv.COLOR_BGR2GRAY)

    faces = dt.detect(gray,cascade)


    # faces = mergeRects(faces)
    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xff == ord('q'):
        break

cv.destroyAllWindows()