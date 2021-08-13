import scipy
import scipy.io as sio
from keras.models import model_from_json

json_file = open("LAST.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("LAST.h5")
print("Loaded model from disk")

import numpy as np
import cv2
import os
import time

# Load an color image in grayscale
img1 = cv2.imread('mf_dataset_clean/mis/mis_8/lab1.bmp', 0)
img2 = cv2.imread('mf_dataset_clean/mis/mis_8/lab2.bmp', 0)
start = time.time()
height = np.size(img1, 0)
width = np.size(img1, 1)
I = np.zeros((height, width, 3), np.uint8)
I[:, :, 0] = img1
I[:, :, 1] = img2
I[:, :, 2] = 0
I = I / 255
classes = np.zeros((height, width), np.uint8)
binary = np.zeros((height, width, 3), np.uint8)
fused = np.zeros((height, width, 3), np.uint8)
probs = []
cv2.imshow('image', img1)
cv2.waitKey(5000)
cv2.destroyAllWindows()
sheight = 7
swidth = 7
counter = 0
for i in range(0, height - sheight, 1):
    for j in range(0, width - swidth, 1):
        Icrp = I[i:i + sheight, j:j + swidth]

        prediction_model = loaded_model.predict(Icrp.reshape(1, sheight, swidth, 3))
        prediction_classes = loaded_model.predict_classes(Icrp.reshape(1, sheight, swidth, 3))
        probs.extend(prediction_model)
        print(prediction_model)

        classes[i:i + sheight, j:j + swidth] = classes[i:i + sheight, j:j + swidth] + prediction_classes
        if np.nanmean(classes[i:i + sheight, j:j + swidth]) < 23:
            binary[i:i + sheight, j:j + swidth, :] = 0
            # binary[i:i+sheight,j:j+swidth,:].reshape(1,sheight,swidth,3)
            fused[i:i + sheight, j:j + swidth, 0] = img1[i:i + sheight, j:j + swidth]
            fused[i:i + sheight, j:j + swidth, 1] = img1[i:i + sheight, j:j + swidth]
            fused[i:i + sheight, j:j + swidth, 2] = img1[i:i + sheight, j:j + swidth]
        else:
            binary[i:i + sheight, j:j + swidth, :] = 255
            # binary[i:i+sheight,j:j+swidth,:].reshape(1,sheight,swidth,3)
            fused[i:i + sheight, j:j + swidth, 0] = img2[i:i + sheight, j:j + swidth]
            fused[i:i + sheight, j:j + swidth, 1] = img2[i:i + sheight, j:j + swidth]
            fused[i:i + sheight, j:j + swidth, 2] = img2[i:i + sheight, j:j + swidth]

path1 = 'all_binary2'
path2 = 'all_fused2'
cv2.imwrite(os.path.join(path1, 'mis8.png'), binary)
cv2.imwrite(os.path.join(path2, 'mis8.png'), fused)
end = time.time()
print('TIME = ', (end - start))
import matplotlib.pyplot as plt

plotbinary = plt.imshow(binary)
plotfused = plt.imshow(fused)
probas = np.array(probs)
probas = np.reshape(probs, (height - sheight, width - swidth, 2))

scipy.io.savemat('probas2', {"probas": probas})

pred_img_c1 = np.zeros((height, width), np.uint8)
pred_img_c2 = np.zeros((height, width), np.uint8)
k = 0
for x in range(0, height - sheight, 1):
    for y in range(0, width - swidth, 1):
        pred_img_c1[x, y] = probs[k][0] * 255
        pred_img_c2[x, y] = probs[k][1] * 255
        k = k + 1
cv2.imwrite(os.path.join(path1, '58_probas1.png'), pred_img_c1)
cv2.imwrite(os.path.join(path2, '58_probas2.png'), pred_img_c2)

plotpred_img_c1 = plt.imshow(pred_img_c1)
plotpred_img_c2 = plt.imshow(pred_img_c2)
