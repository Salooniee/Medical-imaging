import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Creating a matrix
K= np.array([[4,4,4,4,4],[3,4,5,4,3],[3,5,5,5,3],[3,4,5,4,3],[4,4,4,4,4]], dtype='uint8')
print(K, K.dtype, K.shape)
# reading an image
os.chdir(r"C:\medical imaging\AY 2022-23\practical")
image = cv2.imread('ChestXRD.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image, image.dtype, image.shape)
# Display image
plt.subplot(2,1,2)
plt.imshow(image, cmap='gray')
plt.subplot(2,1,1)
plt.imshow(K, cmap="gray", vmin=0, vmax=7)
plt.waitforbuttonpress()
# Display histogram of images
plt.subplot(2,1,1)
plt.hist(K.ravel(),8,[0,7])
plt.subplot(2,1,2)
plt.hist(image.ravel(),256,[0,255])
plt.waitforbuttonpress()

# perform HE
#Global HE
G_K = cv2.equalizeHist(K)
Scaled_G_K = (G_K/255)*7
G_image = cv2.equalizeHist(image)
plt.subplot(1,2,1)
plt.imshow(Scaled_G_K, cmap="gray", vmin=0, vmax=7)
plt.title('HE image')
plt.subplot(1,2,2)
plt.imshow(K, cmap="gray", vmin=0, vmax=7)
plt.title('image without HE')
plt.waitforbuttonpress()

plt.subplot(1,2,1)
plt.imshow(G_image, cmap="gray", vmin=0, vmax=255)
plt.title('HE image')
plt.subplot(1,2,2)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title('image without HE')
plt.waitforbuttonpress()

#compare histogram of processed images
plt.subplot(1,2,1)
plt.hist(Scaled_G_K.ravel(),8,[0,7])
plt.title('Histogram of HE image')
plt.subplot(1,2,2)
plt.hist(K.ravel(),8,[0,7])
plt.title('Histogram of original image')
plt.waitforbuttonpress()

plt.subplot(1,2,1)
plt.hist(G_image.ravel(),256,[0,255])
plt.title('Histogram of HE image')
plt.subplot(1,2,2)
plt.hist(image.ravel(),256,[0,255])
plt.title('Histogram of original image')
plt.waitforbuttonpress()

#case on non-uniform illuminated image
os.chdir(r"C:\medical imaging\AY 2022-23\practical")
NUIll = cv2.imread('NUIimage.png', 0)
print(NUIll.shape)
G_NUIll = cv2.equalizeHist(NUIll)
#implementing CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_NUIll = clahe.apply(NUIll)
# Display images
plt.subplot(1,3,1)
plt.imshow(NUIll, cmap="gray", vmin=0, vmax=255)
plt.title('Original image')
plt.subplot(1,3,2)
plt.imshow(G_NUIll, cmap="gray", vmin=0, vmax=255)
plt.title('Global HE image')
plt.subplot(1,3,3)
plt.imshow(clahe_NUIll, cmap="gray", vmin=0, vmax=255)
plt.title('CLAHE image')
plt.waitforbuttonpress()

