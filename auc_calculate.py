import keras
from sklearn.metrics import roc_auc_score
import numpy as np
from keras.applications import DenseNet121
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.applications.densenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import GlobalAveragePooling2D, Dense
from keras import optimizers
import PIL.Image as pil
import PIL.ImageOps
from sys import argv
import pickle
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/mnt/output/home/data/ai-radiologist/images_all/',
    help='The directory where the input data is stored.')


parser.add_argument(
    '--train_label', type=str, default='/mnt/output/home/data/ai-radiologist',
    help='Path to the training label file')

parser.add_argument(
    '--validation_label', type=str, default='/mnt/output/home/data/ai-radiologist',
    help='Path to the validation label file')

parser.add_argument(
    '--weights_file', type=str, help='Path to the trained weight file')

FLAGS, _ = parser.parse_known_args()

train_label=FLAGS.train_label
validation_label=FLAGS.validation_label

def load_train_valid_labels(train_label,validation_label):

    with open(train_label+'/training_labels_new.pkl', 'rb') as f:
      training_labels = pickle.load(f)
    training_files = np.asarray(list(training_labels.keys()))

    with open(validation_label+'/validation_labels_new.pkl', 'rb') as f:
      validation_labels = pickle.load(f)
    validation_files = np.asarray(list(validation_labels.keys()))
    #labels = dict(training_labels.items() +  validation_labels.items())
    labels = dict(list(training_labels.items()) + list(validation_labels.items()))
    return labels, training_files, validation_files


def load_batch(batch_of_files, labels, is_training=False):
  batch_images = []
  batch_labels = []
  for filename in batch_of_files:
    img = pil.open(os.path.join(FLAGS.data_dir, filename))
    img = img.convert('RGB')
    img = img.resize((256, 256),pil.NEAREST)
    if is_training and np.random.randint(2):
      img = PIL.ImageOps.mirror(img)
    batch_images.append(np.asarray(img))
    batch_labels.append(labels[filename])
  return preprocess_input(np.float32(np.asarray(batch_images))), np.asarray(batch_labels)


def val_generator(num_of_steps, validation_files, labels):
  while True:
    np.random.shuffle(validation_files)
    batch_size = 128
    for i in range(num_of_steps):
      batch_of_files = validation_files[i*batch_size: i*batch_size + batch_size]
      batch_images, batch_labels = load_batch(batch_of_files, labels, True)
      yield batch_images, batch_labels

val_steps = 8653 // 128

labels, training_files, validation_files = load_train_valid_labels(train_label,validation_label)


base_model = DenseNet121(include_top=False,
                   weights=None,
                   input_shape=(256, 256, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(14, activation='sigmoid', bias_initializer='ones')(x)
model = Model(inputs=base_model.input, outputs=predictions)


print("Loading checkpoint: ", FLAGS.weights_file)

model.load_weights(FLAGS.weights_file)


auc_labels = []
probs = []
for i in range(val_steps):
  batch_of_files = validation_files[i*128: i*128+128]
  batch_images, batch_labels = load_batch(batch_of_files, labels)
  probs.extend(model.predict_on_batch(batch_images))
  auc_labels.extend(batch_labels)

for i in range(14):
  print(roc_auc_score(np.asarray(auc_labels)[:,i], np.asarray(probs)[:,i]))



