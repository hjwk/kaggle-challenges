import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Activation, BatchNormalization, Input
from keras.layers import add, average
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

# Load data
train_data = pd.read_json("data/train.json")
test_data = pd.read_json("data/test.json")
train_data.inc_angle = train_data.inc_angle.replace('na', 0)
train_data.inc_angle = train_data.inc_angle.astype(float).fillna(0.0)

# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_data["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_data["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis],
                          x_band2[:, :, :, np.newaxis],
                         ((x_band1 + x_band1) / 2)[:, :, :, np.newaxis]],
                          axis=-1)
X_angle_train = np.array(train_data.inc_angle)
y_train = np.array(train_data["is_iceberg"])

# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_data["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_data["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis],
                         x_band2[:, :, :, np.newaxis],
                        ((x_band1 + x_band1) / 2)[:, :, :, np.newaxis]],
                        axis=-1)
X_angle_test = np.array(test_data.inc_angle)

X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train
                    , X_angle_train, y_train, random_state=123, train_size=0.75)

datagen = ImageDataGenerator(rotation_range=10,
        rescale=1./30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

X_valid = X_valid.astype('float32')/30

# Model definition
inp = Input(X_train.shape[1:])

num_ens = 2
outs = []
for i in range(0, num_ens):
    conv1 = Conv2D(32, (3,3),
                   kernel_initializer='he_uniform',
                   activation='relu')(inp)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.3)(conv1)

    conv3 = Conv2D(32, (3,3),
                   kernel_initializer='he_uniform',
                   activation='relu')(conv1)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling2D()(conv3)
    conv3 = Dropout(0.3)(conv3)

    conv4 = Conv2D(64, (3,3),
                   kernel_initializer='he_uniform',
                   activation='relu')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = MaxPooling2D()(conv4)
    conv4 = Dropout(0.3)(conv4)

    conv5 = Conv2D(64, (3,3),
                   kernel_initializer='he_uniform',
                   activation='relu')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = MaxPooling2D()(conv5)
    conv5 = Dropout(0.3)(conv5)

    conv6 = Conv2D(128, (3,3),
                   kernel_initializer='he_uniform',
                   activation='relu')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Flatten()(conv6)
    conv6 = Dropout(0.3)(conv6)

    '''conv7 = Conv2D(256, (3,3), kernel_initializer='he_uniform')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Flatten()(conv7)
    conv7 = Dropout(0.3)(conv7)'''

    dense1 = Dense(256, kernel_initializer='he_uniform', activation='relu')(conv6)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(128, kernel_initializer='he_uniform', activation='relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    out = Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid')(dense1)
    outs.append(out)

out = average(outs)

model = Model(outputs=out, inputs=inp)

model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["accuracy"])
model.summary()

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.hdf5', 
                               verbose=1, save_best_only=True)

model.fit_generator(datagen.flow(X_train, y_train),
                    steps_per_epoch = len(X_train) / 12,
                    validation_data=(X_valid, y_valid),
                    epochs=70,
                    callbacks=[checkpointer],
                    verbose=1)

model.load_weights('saved_models/weights.best.hdf5')

X_test = X_test.astype('float32')/30
labels = model.predict(X_test)

submission = pd.DataFrame({'id': test_data["id"], 'is_iceberg': labels.reshape((labels.shape[0]))})
submission.head(10)
submission.to_csv("./results.csv", index=False)
'''
X_pl = np.append(X_train, X_test[460:860], axis=0)
y_pl = np.append(y_train, labels[460:860,0], axis=0)

checkpointer_pl = ModelCheckpoint(filepath='saved_models/weights_pl.best.hdf5', 
                                  verbose=1, save_best_only=True)

model.fit_generator(datagen.flow(X_pl, y_pl),
          steps_per_epoch= len(X_pl) / 16,
          validation_data=(X_valid, y_valid),
          epochs=40,
          callbacks=[checkpointer_pl],
          verbose=1)

model.load_weights('saved_models/weights_pl.best.hdf5')

predictions = model.predict(X_test)

submission = pd.DataFrame({'id': test_data["id"], 'is_iceberg': predictions.reshape((predictions.shape[0]))})
submission.head(10)
submission.to_csv("./results_pl.csv", index=False)
'''