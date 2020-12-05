import numpy as np
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, BatchNormalization, Activation, Input, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tqdm import tqdm
from glob import glob
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 28 # because there is not enough RAM
epochs = 40
target_size = (200, 200)

# Data import
train_datagen = ImageDataGenerator(rotation_range=25,
                                   rescale=1./255,
                                   zoom_range=0.2,        
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   vertical_flip=True,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',  # this is the target directory
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    'data/validation',  # this is the target directory
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical')

# Model definition
X_input = Input(train_generator.image_shape)
X = ZeroPadding2D((3, 3))(X_input)

X = Conv2D(filters = 16, kernel_size = (7, 7), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides=(2, 2))(X)

X = Conv2D(filters = 16, kernel_size = (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)
X = Conv2D(filters = 32, kernel_size = (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)
X = MaxPooling2D()(X)

X = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)
X = MaxPooling2D()(X)

X = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)
X = MaxPooling2D()(X)

X = Conv2D(filters = 256, kernel_size = (3, 3), strides = (2,2), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)
X = AveragePooling2D((2, 2))(X)

X = Flatten()(X)
X = Dense(256, kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)
X = Dense(12, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)

model = Model(inputs = X_input, outputs = X, name='ConvNet')

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Training
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.hdf5',
                               verbose=1, save_best_only=True)

history = model.fit_generator(train_generator,
        steps_per_epoch=3839 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=911 // 4,
        verbose=1,
        callbacks=[checkpointer])

model.load_weights('saved_models/weights.best.hdf5')

seedlings_names = [item[11:] for item in sorted(glob("data/train/*"))]

f = open('results.csv', 'w')
f.write('file,species\n')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=target_size)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

test_files = glob("data/test/*")
test = paths_to_tensor(test_files).astype('float32')/255

output = model.predict(test)
for [o, name] in zip(output, test_files):
    f.write(name[10:] + ',')
    seedling = seedlings_names[np.argmax(o)]
    f.write(seedling)
    f.write('\n')
f.close()
