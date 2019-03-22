import numpy as np

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