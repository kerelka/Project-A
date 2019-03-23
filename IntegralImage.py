import numpy as np


def to_integral_image(img_arr):
    integral_array = np.zeros((img_arr.shape[0]+1,img_arr.shape[1]+1))

    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            integral_array[x+1,y+1] = img_arr[x,y] + integral_array[x+1,y] + integral_array[x,y+1] - integral_array[x,y]

    return integral_array


def sum_region(integral_img_arr, top_left, bottom_right):

    """
    integral_img_arr(x,y) = row and col
    top_left (y,x) = col and row
    bottom_right (y,x) = col and row
    :param integral_img_arr:
    :param top_left:
    :param bottom_right:
    :return:
    """
    top_left = (top_left[1],top_left[0])
    bottom_right = (bottom_right[1],bottom_right[0])
    if top_left == bottom_right:
        return integral_img_arr[top_left]
        # calculate the top bottom corner and the right left corner
    top_right = (top_left[0], bottom_right[1])
    bottom_left = (bottom_right[0], top_left[1])
    return integral_img_arr[bottom_right] - integral_img_arr[top_right] - integral_img_arr[bottom_left] + \
           integral_img_arr[top_left]