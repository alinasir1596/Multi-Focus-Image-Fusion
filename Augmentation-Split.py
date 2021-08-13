from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import random
import glob
import keras

X = []
y = []
image_list1 = []
image_list2 = []
image_list3 = []
for filename1 in glob.glob("1/*.png"):
    img1_list = cv2.imread(filename1, 0)
    image_list1.append(img1_list)
for filename2 in glob.glob("2/*.png"):
    img2_list = cv2.imread(filename2, 0)
    image_list2.append(img2_list)
for filename3 in glob.glob("labels/*.png"):
    bmap_list = cv2.imread(filename3, 0)
    image_list3.append(bmap_list)

for k in range(0, np.size(image_list1, 0), 1):

    img1 = image_list1[k]
    img2 = image_list2[k]
    bmap = image_list3[k]

    for i in range(0, np.size(bmap, 0), 1):
        for j in range(0, np.size(bmap, 1), 1):
            if bmap[i, j] == 0:
                bmap[i, j] = 0
            else:
                bmap[i, j] = 255

    height = np.size(img1, 0)
    width = np.size(img1, 1)
    I1 = np.zeros((height, width, 3), np.uint8)
    I1[:, :, 0] = img1
    I1[:, :, 1] = img2
    I1[:, :, 2] = 0
    I2 = np.zeros((height, width, 3), np.uint8)
    I2[:, :, 0] = img2
    I2[:, :, 1] = img1
    I2[:, :, 2] = 0

    sheight = 7
    swidth = 7
    counter1 = 0
    counter2 = 0
    samples = 300
    while counter1 < samples or counter2 < samples:
        i = random.randint(1, np.size(img1, 0) - 1)
        j = random.randint(1, np.size(img1, 1) - 1)
        Icrp1 = I1[i:i + sheight, j:j + swidth]
        Icrp2 = I2[i:i + sheight, j:j + swidth]
        bmapcrp = bmap[i:i + sheight, j:j + swidth]

        if np.size(Icrp1, 0) == sheight and np.size(Icrp1, 1) == swidth and np.var(Icrp1) >= 49:
            if bmapcrp[np.size(bmapcrp, 0) // 2, np.size(bmapcrp, 1) // 2] == 0 and counter1 < samples:
                y.append(0)
                X.extend(Icrp1)
                counter1 = counter1 + 1
            if bmapcrp[np.size(bmapcrp, 0) // 2, np.size(bmapcrp, 1) // 2] == 255 and counter2 < samples:
                y.append(1)
                X.extend(Icrp1)
                counter2 = counter2 + 1
X = np.asarray(X)
X = X.reshape(24000, 7, 7, 3)
y = np.asarray(y)
y = y.reshape(24000, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)

X_train = np.asarray(X_train)
X_train = X_train.reshape(15360, 7, 7, 3)
X_test = np.asarray(X_test)
X_test = X_test.reshape(4800, 7, 7, 3)
X_val = np.asarray(X_val)
X_val = X_val.reshape(3840, 7, 7, 3)

for s in X_train:
    img = s
    aug1 = img[:, :, 0]
    aug2 = img[:, :, 1]
    aug3 = img[:, :, 2]
    img[:, :, 0] = aug2
    img[:, :, 1] = aug1
    img[:, :, 2] = aug3
    X_train = np.append(X_train, img)
print(len(img))
print(len(X_train))

for s in X_test:
    img = s
    aug1 = img[:, :, 0]
    aug2 = img[:, :, 1]
    aug3 = img[:, :, 2]
    img[:, :, 0] = aug2
    img[:, :, 1] = aug1
    img[:, :, 2] = aug3
    X_test = np.append(X_test, img)
print(len(X_test))

for s in X_val:
    img = s
    aug1 = img[:, :, 0]
    aug2 = img[:, :, 1]
    aug3 = img[:, :, 2]
    img[:, :, 0] = aug2
    img[:, :, 1] = aug1
    img[:, :, 2] = aug3
    X_val = np.append(X_val, img)
print(len(X_val))

for i in range(15360):
    if y_train[i] == 0:
        y_train = np.append(y_train, 1)
    else:
        y_train = np.append(y_train, 0)

for i in range(4800):
    if y_test[i] == 0:
        y_test = np.append(y_test, 1)
    else:
        y_test = np.append(y_test, 0)

for i in range(3840):
    if y_val[i] == 0:
        y_val = np.append(y_val, 1)
    else:
        y_val = np.append(y_val, 0)
print(len(img))
print(len(X_train) / 147)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
X_val = np.asarray(X_val)
y_val = np.asarray(y_val)

X_train = X_train.reshape(30720, 7, 7, 3)
y_train = y_train.reshape(30720, 1)
X_test = X_test.reshape(9600, 7, 7, 3)
y_test = y_test.reshape(9600, 1)
X_val = X_val.reshape(7680, 7, 7, 3)
y_val = y_val.reshape(7680, 1)

y_val = keras.utils.to_categorical(y_val, 2)
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

X_val = X_val.astype('float32')
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_val /= 255
X_train /= 255
X_test /= 255

np.save('X_train', X_train)
np.save('X_test', X_test)
np.save('X_val', X_val)
np.save('y_train', y_train)
np.save('y_test', y_test)
np.save('y_val', y_val)
