

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

    def __init__(self, type, posisi, height, width, threshold, polarity):
        self.type = type
        self.top_left = posisi
        self.bottom_right = (posisi[0]+height,posisi[1]+width)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)