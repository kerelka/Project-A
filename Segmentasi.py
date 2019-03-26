import cython
import numpy as np
import cv2 as cv
from PIL import ImageShow,Image
import multiprocessing

class Segmentasi(object):

    def __init__(self,image, height, width):
        self.image = image
        self.height = height
        self.width = width
        self.YCbCr = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

    def get_original(self):
        return self.image

    def get_ycbcr(self):
        return self.YCbCr

    def get_scale(self):
        return self.height,self.width

    def show_ycbcr(self):
        cv.imshow('image',self.YCbCr)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def segmentasi_warna(self):
        #inisial threshold
        nilai_min = np.array([0,131,80], dtype='uint8')
        nilai_max = np.array([255,185,135], dtype='uint8')

        # skinMask = self.color_threshold(nilai_min,nilai_max)
        # skinMask = np.array(skinMask)
        skinMask = cv.inRange(self.YCbCr,nilai_min,nilai_max)
        skinMask = cv.GaussianBlur(skinMask,(3,3),0)
        skin =cv.bitwise_and(self.YCbCr,self.YCbCr, mask=skinMask)
        
        return skin
        
    def color_threshold(self,nilai_min,nilai_max):
        #YCBCR = 3 channel Y, Cr, Cb
        #for every channel in every pixel
        for x in range(0,self.height):
            for y in range(0,self.width):
                #for Y component
                if self.YCbCr[x,y,0] >= nilai_min[0] and self.YCbCr[x,y,0] <= nilai_max[0]:
                    self.YCbCr[x,y,0] = 235
                else:
                    self.YCbCr[x,y,0] = 16
                #for Cr Component
                if self.YCbCr[x,y,1] >= nilai_min[1] and self.YCbCr[x,y,1] <= nilai_max[1]:
                    self.YCbCr[x,y,1] = 128
                else:
                    self.YCbCr[x,y,1] = 128
                #for Cb component
                if self.YCbCr[x,y,2] >= nilai_min[2] and self.YCbCr[x,y,2] <= nilai_max[2]:
                    self.YCbCr[x,y,2] = 128
                else:
                    self.YCbCr[x,y,2] = 128

        return self.YCbCr