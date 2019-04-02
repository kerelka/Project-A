import IntegralImage as ii
import AdaBoost as ab
import numpy as np

def detect(image,cascade):

    img_height, img_width = image.shape
    int_img = ii.to_integral_image(image)

    step = 2
    scale = 1.5
    height = 48
    width = 48
    curScale = 2

    faces = []

    count = 0
    correct_face = 0
    while width <= img_width and height <= img_height :
        for x in range(0, np.int(img_width-width), np.int(np.ceil(step*scale))):
            for y in range(0, np.int(img_height-height), np.int(np.ceil(step*scale))):
                for classifiers in cascade:
                    correct_face = ab.ensemble_vote_with_scalar(int_img, classifiers, (x, y), curScale)
                    if correct_face == 0:
                        break

                if correct_face == 1:
                    faces.append([x, y, np.int(np.ceil(width)), np.int(np.ceil(height))])
                    count += 1
        width *= scale
        height *= scale
        curScale *= scale


    print(str(count)+' Faces Detected')

    return faces