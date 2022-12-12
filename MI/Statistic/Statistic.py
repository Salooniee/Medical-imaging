import os
import cv2
import numpy as np
os.chdir(r"C:\Users\salon\OneDrive\Desktop\MI")
# reading an image
image = cv2.imread('Saloni.png')
cv2.imshow('Color Image',image)
#waits for user to press any key. This is necessary to avoid Python kernel form crashing
cv2.waitKey(0)
#closing all open windows
cv2.destroyAllWindows()
# checking image properties
print(image.shape, image.dtype, type(image))
# converting color image to gray scale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray-scale Image',gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(gray_image.shape, gray_image.dtype, type(gray_image))
# checking properties of an image
print(np.mean(gray_image))
print(np.sum(gray_image))
print(np.std(gray_image))
print(np.mean(gray_image)/np.std(gray_image))
print(np.max(gray_image))
print(np.min(gray_image))
# Basic image processing techniques
# changing brightness of an image
BrightImg = gray_image+25
ConcatImg = cv2.hconcat([gray_image,BrightImg ])
cv2.imshow('Change in brightness of Image',ConcatImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
# image negative
NegativeLena=255-gray_image
cv2.imshow('Image Negative',NegativeLena)
cv2.waitKey(0)
cv2.destroyAllWindows()