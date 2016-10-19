""" Northwestern MSiA 490-30 Assignment #2 starter code, Spring 2016
    This code demonstrates the use of transfer learning to speed up
    the training process for a convolutional neural network.
"""    
import os, numpy as np, random, time
from keras.optimizers import SGD, RMSprop, Adagrad, Adam

# Base settings
zoom     = 1            # Change this to enlarge images
# basepath = "assignment2-data-x%d" % zoom
fraction = 1            # Fraction of data; smaller runs faster but is more noisy

batch_size, nb_epoch = 32, 50

# User tweakable settings
vggxfer       = False    # Enables VGG weight transfer
vgglayers     = 2        # Number of VGG layers to create, 0-5 layers
fclayersize   = 512      # Size of fully connected layers
fclayers      = 1        # Number of fully connected layers
fcdropout     = 0.5      # Dropout factor for fully connected layers
normalize     = True     # Normalize input to zero mean
batchnorm     = True     # Batch normalization
saveloadmodel = True     # Save/load models to reduce training time

optimizer = Adagrad(lr = 0.01, epsilon = 1e-07) 
# Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# Adagrad(lr = 0.01, epsilon = 1e-07) 
# SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#%% Create data folders, path is assignment2-data/<classname>/<imgno>.png
def makedata(basepath, zoom=1):
    """ Saves CIFAR images to data folder. zoom specifies zoom factor"""
    print("Making data folder... please wait. This could take ~10 minutes.")
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()    
    obj_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
    
    for c in obj_classes:
        clsdatapath = os.path.join(basepath, c)
        if not os.path.exists(clsdatapath):
            os.makedirs(clsdatapath)

    X_data = np.concatenate((X_train, X_test), axis=0)
    y_data = np.concatenate((y_train, y_test), axis=0)
    import scipy.ndimage, scipy.misc
    for i, (im, lbl) in enumerate(zip(X_data, y_data)):
        cls = obj_classes[int(lbl)]
        pngname = os.path.join(basepath, cls, "%d.png" % i)
        if int(time.time()*1000) % 100 == 0:
            print("Progress:", i, "/", len(X_data), ":", pngname)
        # For some reason, the CIFAR data is rotated, fix this...
        imdata = np.rot90(np.swapaxes(X_data[i], 0, 2), 3)
        imdata = scipy.ndimage.zoom(imdata, (zoom, zoom, 1))
        scipy.misc.imsave(pngname, imdata)
    print("Done making data folder.")

# Note: delete the data folder if you are changing zoom levels or datasets
# if not os.path.exists(basepath): makedata(basepath, zoom)
    
#%% Load train data
def load_data(basepath, fraction=0.25):
    """Loads image data using folder names as class names
       Beware: make sure all images are the same size, or resize them manually"""
    import scipy.misc
    obj_classes = sorted(os.listdir(basepath))
    xdata, ydata = [], []
    for root, dirs, files in os.walk(basepath):
        random.shuffle(dirs)        
        for i, f in enumerate(files):
            if random.randint(0, 100) > int(100*fraction): continue
            im = scipy.misc.imread(os.path.join(root, f))
            xdata.append(np.swapaxes(im, 0, 2))
            cls = os.path.split(root)[-1]
            clsid = obj_classes.index(cls)
            if int(time.time()*1000) % 1000 == 0:
                print("Progress:", len(xdata), os.path.join(root, f), "shape=", im.shape)
            ydata.append(clsid)
    print("Loaded %d samples" % len(xdata))
    shuffle_ind = list(range(len(xdata)))
    random.shuffle(shuffle_ind)
    return np.array(xdata, dtype='float32')[shuffle_ind] / 255.0, np.array(ydata, dtype='float32')[shuffle_ind], obj_classes

# Load test data
from keras.datasets import cifar100
from keras.utils import np_utils
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
# X_train, y_train, obj_classes = load_data(basepath, fraction=1.0)
if normalize: X_train = X_train - X_train.mean()
img_channels, img_rows, img_cols = X_train.shape[1:]
# split_index = int(0.8*len(X_train))
# X_train, y_train, X_test, y_test = X_train[:split_index], y_train[:split_index], X_train[split_index:], y_train[split_index:]
Y_train = np_utils.to_categorical(y_train, 100)
Y_test  = np_utils.to_categorical(y_test, 100)

#%% Define a VGG-compatible model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K

model = Sequential()

# This layer is used for visualizing filters. Don't remove it.
model.add(ZeroPadding2D((1, 1), batch_input_shape=(1, 3, img_rows, img_cols)))
first_layer = model.layers[-1]
input_img = first_layer.input

# VGG net definition starts here. Change the vgglayers to set how many layers to transfer
if vgglayers >= 1:
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', input_shape=(img_channels, img_rows, img_cols)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))

if vgglayers >= 2:
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))

if vgglayers >=3:
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))

if vgglayers >= 4:
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))

if vgglayers >= 5:
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# VGG net definition ends here
# All layers past this convert the VGG convolutional "code" into a classification

# Flatten and normalize data here, we don't know the data distribution of
# the preloaded training weights, so batch normalization helps fix slowdowns
model.add(Flatten())
from keras.layers.normalization import BatchNormalization
if batchnorm: model.add(BatchNormalization())

for l in range(1, fclayers + 1):
    model.add(Dense(fclayersize, name='fc%d' % l))
    model.add(Activation('relu'))
    model.add(Dropout(fcdropout)) # Modify dropout as necessary

if batchnorm: model.add(BatchNormalization())
model.add(Dense(100))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#%% Load existing VGG weights
if vggxfer:
    weights_file = "vgg16_weights.h5" # Pretrained VGG weights
    if os.path.exists(weights_file):
        print("Found existing weights file, loading data...")
        import h5py
        f = h5py.File(weights_file)
        if 'layer_names' in f.attrs.keys(): print("Weights file has:", f.attrs['layer_names'])
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers) or 'conv' not in model.layers[k].name:
                #print("Skipping layer:", k, model.layers[k].name if k < len(model.layers) else "<none>")
                # Commented layers are skipped                  
                continue
            print("Transferring layer:", k, model.layers[k].name, model.layers[k].output_shape)        
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()

# Print model summary
model.summary()
print("Trainable layers:", model.trainable_weights)

#%% Visualization code
import matplotlib.pyplot as plt
layer_dict = dict([(layer.name, layer) for layer in model.layers])

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x = x*0.1 + 0.5
    x = np.clip(x, 0, 1) * 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def viz_filter_max(layer_name, filter_index=0, max_steps=150):
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, filter_index, :, :])
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img], [loss, grads])
    step = 1e-0
    input_img_data = np.random.random((1, 3, img_rows, img_cols)) * 20 + 128.
    tm = time.time()
    for i in range(max_steps):
        if (time.time() - tm > 5) and (i % 10 == 0): print(i, '/', max_steps)
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        if time.time() - tm > 1:
            plt.text(0.1, 0.1, "Filter viz timeout", color='red')
            break
    img = input_img_data[0]
    img = deprocess_image(img)
    fig = plt.imshow(img)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    return layer_output

def viz_losses(losses, scores):
    plt.plot(np.log(losses), label='Train logloss')
    plt.plot(np.log(scores), label='Test logloss')
    plt.legend()
    plt.show()

def viz_acc(accuracies, validations):
    plt.plot(accuracies, label='Train accuracy')
    plt.plot(validations, label='Test accuracy')
    plt.legend()
    plt.show()

def viz_filters(nbfilters=3):
    for layer_name in sorted(layer_dict.keys()):
        if not hasattr(layer_dict[layer_name], 'nb_filter'): continue
        nfilters = layer_dict[layer_name].nb_filter
        print("Layer", layer_name, "has", nfilters, "filters")
        plt.subplots(1, nbfilters)
        for j in range(nbfilters):
            plt.subplot(1, nbfilters, j + 1)
            viz_filter_max(layer_name, random.randint(0, nfilters-1))
        plt.show()

def test_prediction(im=None, y=None):
    t_img = random.randint(0, len(X_train) - 1)
    if im is None: im, y = X_train[t_img], Y_train[t_img]
    plt.imshow(im.T - im.min())
    plt.show()
    pred = model.predict_proba(np.expand_dims(im, 0))
    cls = np.argmax(y)
#==============================================================================
#     print("Actual: %s(%d)" % (obj_classes[cls], cls))
#     for cls in list(reversed(np.argsort(pred)[0]))[:5]:
#         conf = float(pred[0, cls])/pred.sum()
#         print("    predicted: %010s(%d), confidence=%0.2f [%-10s]" % (obj_classes[cls], cls, conf, "*" * int(10*conf)))
#==============================================================================
    return pred
    
#%% Image data augmentation 
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,               # set input mean to 0 over the dataset
    samplewise_center=False,                # set each sample mean to 0
    featurewise_std_normalization=False,    # divide inputs by std of the dataset
    samplewise_std_normalization=False,     # divide each input by its std
    zca_whitening=False,                    # apply ZCA whitening
    rotation_range=0,                       # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,                    # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,                   # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,                  # randomly flip images
    vertical_flip=False)                    # randomly flip images
datagen.fit(X_train)

#%% Training code
if saveloadmodel and os.path.exists("baseline.h5"): model.load_weights("baseline.h5")
#from keras.utils import generic_utils
losses, accuracies, scores, validations = [], [], [], []
for e in range(nb_epoch):
    print('---- Epoch', e, ' ----')
    # Load data subset if needed
#==============================================================================
#     if fraction < 1.0:
#         X_train, y_train, obj_classes = load_data(basepath, fraction)
#         Y_train = np_utils.to_categorical(y_train, len(obj_classes))
#         if normalize: X_train = X_train - X_train.mean()
#==============================================================================
    print('Training...')      
    loss = model.fit_generator(datagen.flow(X_train, Y_train, shuffle=True,
                    batch_size=batch_size),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=1,
                    validation_data=(X_test, Y_test))               
    if saveloadmodel: model.save_weights("baseline.h5", overwrite=True)

    print(loss.history)    
    
    losses.append(loss.history['loss']), scores.append(loss.history['val_loss'])
    accuracies.append(loss.history['acc']), validations.append(loss.history['val_acc'])

    viz_losses(losses, scores)
    viz_acc(accuracies, validations)

    t_ind = random.randint(0, len(X_train) - 1)
    test_prediction(X_train[t_ind], Y_train[t_ind])
    if e % 20 == 0:
        try:
            print("Visualizing filters, press CTRL-C to stop...")
            viz_filters()
        except KeyboardInterrupt:
            pass