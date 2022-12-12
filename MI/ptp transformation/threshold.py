import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# reading an image
os.chdir(r"E:\DS\python")
image = cv2.imread('yeast cell.jpg',0)
print(image.shape)
plt.hist(image.ravel(),256,[0,255])
plt.title('Histogram of original image')
plt.show()

#global thresholding
th = 50 #try 127
th_mean =np.mean(image)
max_val = 255
ret, o1 = cv2.threshold(image, th_mean, max_val, cv2.THRESH_BINARY)
plt.imshow(o1, cmap='gray')
plt.title('Thresholded image using global thresholding')
plt.show()
plt.hist(o1.ravel(),256,[0,255])
plt.title('Histogram of binary image')
plt.show()

#Ostu's method, cv.THRESH_OTSU is passed as an extra flag
#The threshold value can be chosen arbitrary
th_ostu = 127
max_val = 255
ret, o2 = cv2.threshold(image, th_ostu, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
plt.imshow(o2, cmap='gray')
plt.title('Thresholded image using ostu''s method')
plt.show()