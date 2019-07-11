# simple-mathematical-expression-calculator

This is a simple system to take an image of a mathematical expression as input and evaluate it.

## supported operations
Only addition and deletion are supported at the time due to the limitedness of the training set.

### notes
Due to broken links, I couldn't get the raw images of the dataset. So, I had to convert the dataset from matlab arrays to numpy arrays. Luckily enough, they are saved as `*.npy` and are loaded faster than raw data would have loaded.

## How it works
* The image is read as grayscale and is then converted to binary using thresholding.
![alt text](images/binary_image.png "binary image")
* The image is cropped to the region of interest(ROI) discarding all empty rows and columns so that it becomes much easier to work with.
![alt text](images/refined_image.png "ROI image")
* The image is split horizontally into symbols. Each symbol is identified by the empty column of pixels before and after it. 

![alt text](images/8_symbol.png "8 symbol")
![alt text](images/7_symbol.png "7 symbol")
![alt text](images/+_symbol.png "+ symbol")
![alt text](images/9_symbol.png "9 symbol")
![alt text](images/5_symbol.png "5 symbol")
* The ROI of each symbol is further cleaned by removing the empty rows and columns surrounding it. It doesn't make much of a difference but to be consistent with the training data

![alt text](images/8_symbol_roi.png "ROI of 8 symbol")
![alt text](images/7_symbol_roi.png "ROI of 7 symbol")
![alt text](images/+_symbol_roi.png "ROI of + symbol")
![alt text](images/9_symbol_roi.png "ROI of 9 symbol")
![alt text](images/5_symbol_roi.png "ROI of 5 symbol")
>There is an exception to ones (1) and the subtraction sign(-) because they would fill out the whole ROI hence they wouldn't be identified correctly. That's why there is a minimum height and width of each symbol. Whenever the symbol dimentions are less than the specified number, the symbol is padded with empty rows/columns.
* Each symbol is then resized to the same size of the training set (16*16).

![alt text](images/8_resized.png "resized 8 symbol")
![alt text](images/7_resized.png "resized 7 symbol")
![alt text](images/+_resized.png "resized + symbol")
![alt text](images/9_resized.png "resized 9 symbol")
![alt text](images/5_resized.png "resized 5 symbol")
* The resized symbols are fed to a pretrained neural network to classify to which symbol they belong.
* The classification results are then joined as operands and operations to be calculated at the end.
