import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

img = cv2.imread("p.png")

def bgtogray(imgs):

    R = np.array(imgs[:,:,2])
    G = np.array(imgs[:,:,1])
    B = np.array(imgs[:,:,0])
    
    R = (R *.3333)
    G = (G *.3333)
    B = (B *.3333)
    
    Avg = (R+G+B)
    
    grayImage = copy.copy(imgs)
    
    for i in range(3):
        grayImage[:,:,i] = Avg
    
    return grayImage
    
bgtogray(img)
gray = bgtogray(img)

print (gray.shape)
print ("Nilai piksel B,G,R pada Citra bewarna pada baris 1000 sampai 105 dan kolom 100 sampai 105 :\n")
print ("B: \n", img[100:105, 100:105,0], "\n\nG: \n", img[100:105, 100:105,1], "\n\nR: \n", img[100:105, 100:105,2])
print ("\n Nilai piksel B,G,R pada Citra bewarna pada baris 1000 sampai 105 dan kolom 100 sampai 105 :\n")
print ("B: \n", gray[100:105, 100:105,0], "\n\nG: \n", gray[100:105, 100:105,1], "\n\nR: \n", gray[100:105, 100:105,2])

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Citra Asli")
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.imshow(gray)
plt.title("Grayscale")
plt.xticks([]), plt.yticks([])

plt.show()