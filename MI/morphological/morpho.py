import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# reading an image
os.chdir(r"C:\medical imaging\AY 2022-23\practical")
image = cv2.imread('RBC.tif',0)
print(image.shape)
#binarize the image
th = 0
max_val = 255
ret, binary = cv2.threshold(image, th, max_val,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )
print("The threshold value is {}".format(ret))
plt.imshow(binary, cmap='gray')
plt.title('Thresholded Image')
plt.show()
#k = np.ones((5, 5), np.uint8)
#k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))  #(col,row)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7))
#k = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))
print(k)

# erosion operation
erosion = cv2.erode(binary, k, iterations = 1)
plt.imshow(erosion, cmap='gray')
plt.title('Eroded Image')
plt.show()

dilation = cv2.dilate(binary, k , iterations = 1)
plt.imshow(dilation, cmap='gray')
plt.title('Dilated Image')
plt.show()

# erosion followed by dilation
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.show()

# Dilation followed by Erosion
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
plt.imshow(closing, cmap='gray')
plt.title('Closing')
plt.show()