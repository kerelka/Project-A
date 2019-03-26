import IntegralImage as ii
import numpy as np

def enum(**enums):
    return type('Enum',(),enums)

#define features
FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_VERTICAL=(1, 3), THREE_HORIZONTAL=(3, 1),
                   FOUR=(2, 2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL,
                FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]
"""
HaarLikeFeature:

TWO_VERTICAL      TWO_HORIZONTAL
_____             __________
|  1 |            |  1 |  2 |    
|____|            |____|____|
|  2 |
|____|

THREE_VERTICAL    THREE_HORIZONTAL
_____             _______________
|  1 |            |  1 |  2 |  3 |
|____|            |____|____|____|
|  2 |
|____|
|  3 |
|____|

FOUR
__________
|  1 |  2 |
|____|____|
|  3 |  4 |
|____|____|

"""

class HaarLikeFeature(object):

    def __init__(self, type, posisi, width, height, threshold, polarity, weight=1):
        self.type = type
        self.top_left = posisi
        self.bottom_right = (posisi[0]+width,posisi[1]+height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity
        self.weight = weight

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def get_vote(self,int_img):
        score = self.get_score(int_img)
        """ 1, jika PjFj(x) < Pj0j(x) else 0"""
        return 1 if self.polarity * score < self.polarity * self.threshold else 0

    def get_vote_with_scale(self,int_img, sub_window_top_left, scalar):
        score = self.get_score_with_scale(int_img,sub_window_top_left, scalar)
        return 1 if self.polarity * score < self.polarity * self.threshold else 0

    def get_weight(self):
        return self.weight

    def get_score(self,int_img):
        score = 0
        if self.type == FeatureType.TWO_VERTICAL:
            first = ii.sum_region(int_img, self.top_left,
                                  (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
            second = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                                   self.bottom_right)
            score = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = ii.sum_region(int_img, self.top_left,
                                  (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))
            second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                                   self.bottom_right)
            score = first - second
        elif self.type == FeatureType.THREE_VERTICAL:
            first = ii.sum_region(int_img, self.top_left,
                                  (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            second = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                                   (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            third = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)),
                                  self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = ii.sum_region(int_img, self.top_left,
                                  (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
            second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 3), self.top_left[1]),
                                   (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
            third = ii.sum_region(int_img, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]),
                                  self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.FOUR:
            # top left area
            first = ii.sum_region(int_img, self.top_left,
                                  (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
            # top right area
            second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                                   (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
            # bottom left area
            third = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                                  (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
            # bottom right area
            fourth = ii.sum_region(int_img,
                                   (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)),
                                   self.bottom_right)
            score = first - second - third + fourth

        return score

    def get_score_with_scale(self,int_img, sub_window_top_left, scalar):

        score = 0

        tl = (np.int(np.ceil(self.top_left[0] * scalar) + sub_window_top_left[0]),
              np.int(np.ceil(self.top_left[1] * scalar) + sub_window_top_left[1]))
        br = (np.int(np.ceil(self.bottom_right[0] * scalar) + sub_window_top_left[0]),
              np.int(np.ceil(self.bottom_right[1] * scalar) + sub_window_top_left[1]))
        w = np.int(np.ceil(self.width * scalar))
        w2 = np.int(np.ceil(self.width * scalar / 2))
        w3 = np.int(np.ceil(self.width * scalar / 3))
        h = np.int(np.ceil(self.height * scalar))
        h2 = np.int(np.ceil(self.height * scalar / 2))
        h3 = np.int(np.ceil(self.height * scalar / 3))

        if self.type == FeatureType.TWO_VERTICAL:
            '''
                -----------
                |         |
                -----------
                |#########|
                -----------
            '''

            first = ii.sum_region(int_img, tl, (tl[0] + w, tl[1] + h2))
            second = ii.sum_region(int_img, (tl[0], tl[1] + h2), br)
            score = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            '''
                  -------------
                  |     |#####|
                  |     |#####|
                  |     |#####|
                  -------------
            '''
            first = ii.sum_region(int_img, tl, (tl[0] + w2, tl[1] + h))
            second = ii.sum_region(int_img, (tl[0] + w2, tl[1]), br)
            score = first-second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            '''
                  -------------------
                  |     |#####|     |
                  |     |#####|     |
                  |     |#####|     |
                  -------------------
            '''
            first = ii.sum_region(int_img, tl, (tl[0] + w3, tl[1] + h))
            second = ii.sum_region(int_img, (tl[0] + w3, tl[1]), (tl[0] + 2 * w3, tl[1] + h))
            third = ii.sum_region(int_img, (tl[0] + 2 * w3, tl[1]), br)
            score = first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            '''
                -----------
                |         |
                -----------
                |#########|
                -----------
                |         |
                -----------
            '''
            first = ii.sum_region(int_img, tl, (br[0], tl[1] + h3))
            second = ii.sum_region(int_img, (tl[0], tl[1] + h3), (br[0], tl[1] + 2 * h3))
            third = ii.sum_region(int_img, (tl[0], tl[1] + 2 * h3), br)
            score = first - second + third
        elif self.type == FeatureType.FOUR:
            '''
                ----------------------
                |         |##########|
                |         |##########|
                ----------------------
                |#########|          |
                |#########|          |
                ----------------------
            '''
            # top left area
            first = ii.sum_region(int_img, tl, (tl[0] + w2, tl[1] + h2))
            # top right area
            second = ii.sum_region(int_img, (tl[0] + w2, tl[1]), (br[0], tl[1] + h2))
            # bottom left area
            '''
            print('self.bottomR : {0}'.format(self.bottom_right))
            print('sub_window_top_left : {0}'.format(sub_window_top_left))
            print('br : {0}'.format(br))
            print('scalar : {0}'.format(scalar))
            print((tl[0], tl[1] + h2), (tl[0] + w2, br[1]))
            '''
            third = ii.sum_region(int_img, (tl[0], tl[1] + h2), (tl[0] + w2, br[1]))
            # bottom right area
            fourth = ii.sum_region(int_img, (tl[0] + w2, tl[1] + h2), br)
            score = first - second - third + fourth

        return score