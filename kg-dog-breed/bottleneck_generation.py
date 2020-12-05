import numpy as np
from sklearn.datasets import load_files
import os
from keras import applications
from keras.preprocessing import image     
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def generate_bottleneck_features(input_dir):
    #datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input)

    # build the network
    model = applications.ResNet50(include_top=False, weights='imagenet')

    files = load_files(input_dir)
    tensors = paths_to_tensor(files['filenames'])#.astype('float32')/255
    data = applications.resnet50.preprocess_input(tensors)
        
    bottleneck_features = model.predict(
        data, batch_size=16)

    filename = 'bottleneck_features/' + input_dir + '.npy'
    np.save(filename, bottleneck_features)

def generate_bottleneck_features_files(input_files):
    #datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input)

    # build the network
    model = applications.ResNet50(include_top=False, weights='imagenet')

    tensors = paths_to_tensor(input_files)#.astype('float32')/255
    data = applications.resnet50.preprocess_input(tensors)
        
    bottleneck_features = model.predict(
        data, batch_size=16)

    filename = 'bottleneck_features/' + input_files[0] + '.npy'
    np.save(filename, bottleneck_features)

files = os.listdir('data/test')
generate_bottleneck_features_files(np.vstack(files))