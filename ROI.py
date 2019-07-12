# -*- coding: utf-8 -*-
debug = False
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
	
	
	
from imutils import grab_contours, contours
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
def split(img):
	"""
	splits the digits or symbols in image
	returs list of smaller images
	"""
	# get the contours of each symbol
	cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = grab_contours(cnts)
	# sort contours from left to right (the correct order of symbols)
	cnts, boxes = contours.sort_contours(cnts, method='left-to-right')

	if debug:
		_, ax = plt.subplots(1)
		ax.imshow(img, cmap=cm.gray)
		from matplotlib.patches import Polygon, Rectangle
		for i in range(len(cnts)):
			poly = Polygon(cnts[i][:,0,:], edgecolor='r', linewidth=2, facecolor='none')
			ax.add_patch(poly)
			(x,y,w,h) = boxes[i]
			rect = Rectangle((x,y), width=w, height=h, linewidth=3, edgecolor='g', facecolor='none')
			ax.add_patch(rect)
			
		plt.show()
	
	symbols = []
	for c in cnts:
		mask = np.zeros_like(img)	# the mast that will act as alpha channel
		cv2.drawContours(mask, [c], 0, 255, -1)	# draw filled contour to the mask
		symbol = np.zeros_like(img)
		symbol[mask == 255] = img[mask == 255]	# transfer the symbol using the mask
		symbols.append(symbol)
		if debug:
			plt.imshow(mask, cmap=cm.gray)
			plt.show()
	return symbols

	