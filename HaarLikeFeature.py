import IntegralImage as ii

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

    def __init__(self, type, posisi, width, height, threshold, polarity):
        self.type = type
        self.top_left = posisi
        self.bottom_right = (posisi[0]+width,posisi[1]+height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def get_vote(self,int_img):
        score = self.get_score(int_img)
        return (1 if self.polarity * score < self.polarity * self.threshold else 0)

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