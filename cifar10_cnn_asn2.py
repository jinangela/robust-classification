'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''
# Based on: https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

batch_size = 32
nb_classes = 10
nb_epoch = 1 # Please don't modify this to 50. This code manually runs 50 epochs in the for loop at the end.
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])

adg = Adagrad(lr = 0.01, epsilon = 1e-07) # I still use adagrad here as baseline.
model.compile(loss='categorical_crossentropy',
              optimizer=adg,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('Not using real-time data augmentation.')

# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

import matplotlib.pyplot as plt, numpy as np, random
def viz_losses(losses, scores):
    plt.plot(np.log(losses), label='Train loss')
    plt.plot(np.log(scores), label='Test loss')
    plt.legend()
    plt.show()

def viz_acc(accuracies, validations):
    plt.plot(np.log(accuracies), label='Train accuracy')
    plt.plot(np.log(validations), label='Test accuracy')
    plt.legend()
    plt.show()

obj_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
def test_prediction(im=None, y=None):
    t_img = random.randint(0, len(X_train) - 1)
    if im is None: im, y = X_train[t_img], Y_train[t_img]
    plt.imshow(im.T)
    plt.show()
    pred = model.predict_proba(np.expand_dims(im, 0))
    cls = np.argmax(y)
    print("Actual: %s(%d)" % (obj_classes[cls], cls))
    for cls in list(reversed(np.argsort(pred)[0]))[:5]:
        conf = float(pred[0, cls])/pred.sum()
        print("    predicted: %010s(%d), confidence=%0.2f [%-10s]" % (obj_classes[cls], cls, conf, "*" * int(10*conf)))
    return pred    
    
losses, accuracies, scores, validations = [], [], [], []
for i in range(50):
    print("Epoch:", i)
    # fit the model on the batches generated by datagen.flow()
    mod = model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))
    losses.append(mod.history['loss']), scores.append(mod.history['val_loss'])
    accuracies.append(mod.history['acc']), validations.append(mod.history['val_acc'])
    viz_losses(losses, scores) # plot loss curve
    viz_acc(accuracies, validations) # plot accuracy curve(so I don't save the mod.history to json, we can just save console to pdf)
    t_ind = random.randint(0, len(X_train) - 1)
    im = np.swapaxes(np.flipud(np.rot90(np.swapaxes(X_train[t_ind], 0, 2))), 0, 2)
    test_prediction(im, Y_train[t_ind])
    model.save_weights("0d5.hdf5", overwrite = True)

import time,json
tm = time.time()
with open("results-%d.json" % tm, "w") as f:
    json.dump(mod.history,f)