import numpy as np

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
y_val = np.load('y_val.npy')

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
import matplotlib.pyplot as plt

model = VGG16(include_top=False, weights='imagenet')

weights2 = model.layers[1].get_weights()

classifier = Sequential()
conv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', input_shape=(7, 7, 3))
classifier.add(conv1)
classifier.add(Activation('relu'))

conv2 = Conv2D(128, (3, 3), strides=(1, 1), padding='valid', input_shape=(5, 5, 3))
classifier.add(conv2)
classifier.add(Activation('relu'))
classifier.add(Flatten())
classifier.add(Dense(units=1152))
classifier.add(Dense(units=2))
classifier.add(Activation('softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = classifier.fit(X_train, y_train,
                         batch_size=32,
                         epochs=30,
                         validation_data=(X_val, y_val),
                         shuffle=True)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['val_categorical_accuracy'])
plt.plot(history.history['categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
cvscores = []
scores = classifier.evaluate(X_val, y_val)
print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1] * 100))
cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# SAVING
model_json = classifier.to_json()
with open("LAST.json", "w") as json_file:
    json_file.write(model_json)

from keras.utils import plot_model

plot_model(classifier, show_shapes=True, to_file='LAST.png')
# serialize weights to HDF5
classifier.save_weights("LAST.h5");
