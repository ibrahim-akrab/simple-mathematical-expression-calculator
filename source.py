# -*- coding: utf-8 -*-
debug = True
# import necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

# disable warnings resulting from keras internal implementation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# change dpi of inline plots
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 150

# read image as grayscale for preprocessing
img = cv2.imread('expressions/87+95_overlapping.jpg', 0)

# binarize image
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3,3), dtype='uint8'))
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3,3), dtype='uint8'))
if debug:
	plt.imshow(img, cmap = cm.gray)

# extract region of interest (ROI)
from ROI import extractROI, split

# extract symbols from image
symbols = split(img)

if debug:
	for symbol in symbols:
		plt.imshow(symbol, cmap = cm.gray)
		plt.show()

# find ROI of each symbol
refined_symbols = []
for symbol in symbols:
	refined_symbols.append(extractROI(symbol, min_width=60, min_height=50))

if debug:
	for symbol in refined_symbols:
		plt.imshow(symbol, cmap=cm.gray)
		plt.show()

# resize each symbol to be exactly 16*16 as the dataset
resized_symbols = np.empty(shape=(len(refined_symbols),16*16), dtype='float')
for i, symbol in enumerate(refined_symbols):
	resized_symbol = cv2.resize(symbol, (16,16), interpolation=cv2.INTER_AREA)
	# keep the image binarized after rescaling
	resized_symbol[resized_symbol > 0] = 1
	resized_symbols[i,:] = resized_symbol.astype('float').flatten()

if debug:
	for symbol in resized_symbols:
		plt.imshow(symbol.reshape((16,16)), cmap = cm.gray)
		plt.show()

# load the classifier saved model
from keras.models import load_model

model = load_model('model.h5')

symbols_pred = model.predict(resized_symbols)
symbols_pred = (symbols_pred > 0.5).astype('uint8').argmax(axis=1)

# evaluate the result of the expression
from evaluate import evaluate

result = evaluate(symbols_pred)
print("result of expression =", result)