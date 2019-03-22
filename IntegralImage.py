import numpy as np

def sum_region(integral_img_arr, top_left, bottom_right):

    if top_left == bottom_right:
        return integral_img_arr[top_left]

    top_right = (top_left[0], bottom_right[1])
    bottom_left = (bottom_right[0], top_left[1])

    return integral_img_arr[top_left] + integral_img_arr[bottom_right] - integral_img_arr[bottom_left] - integral_img_arr[top_right]