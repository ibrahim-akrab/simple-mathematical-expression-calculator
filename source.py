# -*- coding: utf-8 -*-

# import necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# read image as grayscale for preprocessing
img = cv2.imread('expressions/87+95.jpg', 0)

# binarize image
mask = img > 100
img[mask == False] = 255
img[mask == True] = 0


# extract region of interest (ROI)
from ROI import extractROI, split
refined_img = extractROI(img)

# extract symbols from image
symbols = split(refined_img)

# find ROI of each symbol
refined_symbols = []
for symbol in symbols:
    refined_symbols.append(extractROI(symbol, min_width=60, min_height=50))

# resize each symbol to be exactly 16*16 as the dataset
resized_symbols = np.empty(shape=(len(refined_symbols),16*16), dtype='float')
for i, symbol in enumerate(refined_symbols):
    resized_symbol = cv2.resize(symbol, (16,16), interpolation=cv2.INTER_AREA)
    # keep the image binarized after rescaling
    resized_symbol[resized_symbol > 0] = 1
    resized_symbols[i,:] = resized_symbol.astype('float').flatten()

for symbol in resized_symbols:
    plt.imshow(symbol.reshape((16,16)))
    plt.show()

# load the classifier saved model
from keras.models import load_model

model = load_model('model.h5')

symbols_pred = model.predict(resized_symbols)
symbols_pred = (symbols_pred > 0.5).astype('uint8').argmax(axis=1)

# evaluate the result of the expression
from evaluate import *

result = evaluate(symbols_pred)