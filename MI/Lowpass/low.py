import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#reading axial CT scan
#check for noisyimage
os.chdir(r"C:\medical imaging\AY 2022-23\practical")
image = cv2.imread('CT chest.jpg',0)
print(image, image.dtype, image.shape)
kernel = np.ones((5,5))/25
ConvImage = cv2.filter2D(image,-1,kernel)   #-1 ensures input = output dimension
blur = cv2.blur(image,(3,3))
gaussian = cv2.GaussianBlur(image, (37, 37), 0)   # 0 indicates std dev is calculated from kernel size
median = cv2.medianBlur(image,5)
#display images
titles = ['Original Image', 'Convolved image', 'Blur', 'Gaussian Blur','Median blur']
outputs = [image,ConvImage, blur, gaussian, median]
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(outputs[i], cmap='gray')
    plt.title(titles[i])
plt.show()