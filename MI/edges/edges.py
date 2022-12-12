import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#reading axial CT scan
#check for noisyimage
os.chdir(r"C:\medical imaging\AY 2022-23\practical")
image = cv2.imread('CT chest.jpg',0)
print(image, image.dtype, image.shape)
Gxkernel = np.array([[-1, -1, -1],[2, 2, 2],[-1, -1, -1]])
Gx_image = cv2.filter2D(image,-1,Gxkernel)

Gykernel = np.array([[-1, 2, -1],[-1, 2, -1],[-1, 2, -1]])
Gy_image = cv2.filter2D(image,-1,Gykernel)

G_image = Gx_image+Gy_image
#L2gradient: sqrt(gradient_x_square + gradient_y_square)
#L1 gradient: abs(gradient_x) + abs(gradient_y)
Canny_image1 = cv2.Canny(image, 50, 100, apertureSize= 5, L2gradient = True)
Canny_image2 = cv2.Canny(image, 50, 100, apertureSize= 5, L2gradient = False)
#display images
titles = ['Original Image', 'Gradient image along x-axis', 'Gradient image along y-axis', 'Gradient image both axis','Canny image L2', 'Canny image L1']
outputs = [image,Gx_image, Gy_image, G_image, Canny_image1, Canny_image2 ]
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(outputs[i], cmap='gray')
    plt.title(titles[i])
plt.show()