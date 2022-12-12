import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# reading an image
os.chdir(r"C:\medical imaging\AY 2022-23\practical")
image = cv2.imread('ChestXRD.png',0)
row,col = image.shape
# image rotation
#cv2.getRotationMatrix2D(center, angle, scale)
R = cv2.getRotationMatrix2D((0, 0), 50, 1)
print(R)
rotated_image = cv2.warpAffine(image, R, (image.shape[1], image.shape[0]))
# Display image
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(rotated_image, cmap="gray")
plt.title('Rotated Image')
plt.waitforbuttonpress()
#image scaling
scaled_image= cv2.resize(image,(2*row,3*col))
# Display image
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(scaled_image, cmap="gray")
plt.title('Scaled Image')
plt.waitforbuttonpress()
#image translation
T = np.float32([[1, 0, 150], [0, 1, 50]])
print(T)
shifted_image = cv2.warpAffine(image, T, (image.shape[1], image.shape[0]))
# Display image
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(shifted_image, cmap="gray")
plt.title('Shifted Image')
plt.waitforbuttonpress()
#image crop
crop = image[50:180, 100:300]
# Display image
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(crop, cmap="gray")
plt.title('Crop Image')
plt.waitforbuttonpress()