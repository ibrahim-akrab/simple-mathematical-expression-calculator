# -*- coding: utf-8 -*-
import numpy as np

def extractROI(img, min_width=0, min_height=0):
    """
    expects a grayscale image with only one layer
    returns cropped version of image
    """
    # sum black pixels in each row and each column
    col_sum = np.sum(img, axis=0)
    row_sum = np.sum(img, axis=1)
    
    # chooose first/last rows/columns that contain black pixels
    non_zero_cols = np.nonzero(col_sum)
    left_index = non_zero_cols[0][0]
    right_index = non_zero_cols[0][-1]
    non_zero_rows = np.nonzero(row_sum)
    top_index = non_zero_rows[0][0]
    bottom_index = non_zero_rows[0][-1]
    if min_width != 0 and right_index - left_index < min_width:
        # make the symbol in the center of an array of zeros
        temp = np.zeros((img.shape[0], min_width))
        diff = min_width - (right_index - left_index)
        temp[:, diff//2:diff//2 + (right_index - left_index) + 1] += img
        img = temp
        left_index, right_index = 0, min_width - 1

    if min_height != 0 and bottom_index - top_index < min_height:
        # make the symbol in the center of an array of zeros
        temp = np.zeros((min_height, img.shape[1]))
        diff = min_height - (bottom_index - top_index)
        temp[diff//2:diff//2 + (bottom_index - top_index) + 1, :] += img
        img = temp
        top_index, bottom_index = 0, min_height - 1
    
    # crop the image to keep only the area with the black pixels
    return img[top_index: bottom_index + 1, left_index: right_index + 1]
    
    
    
def split(img):
    """
    splits the digits or symbols in image
    returs list of smaller images
    """
    
    # count number of black pixels in each column
    col_sum = np.sum(img, axis=0)
    
    # find first nummber after some consecutive zeros
    end_of_zeros = np.where(((col_sum[:-1] == 0) & (col_sum[1:] > 0)))[0]
    # find first zero-sum row after some non-zero rows
    start_of_zeros = np.where(((col_sum[:-1] > 0) & (col_sum[1:] == 0)))[0]
    
    # start of each symbol is the end of each zeros sequence (and the beggining of the img)
    start_of_symbol = np.insert(end_of_zeros + 1, obj=0, values=0)
    # end of each symbol is the start of zeros sequence (and the end of image)
    end_of_symbol = np.append(start_of_zeros + 1, values=img.shape[-1])
    
    # define minimum width and minimum height to avoid having ones and (-) signs stretched out
    return [ img[:, start_of_symbol[i] : end_of_symbol[i]] for i in range(len(start_of_symbol))]
    