from __future__ import print_function


import sys
sys.path.append('/mnt/output/home/ai-radiologist')

print(sys.path)

import os
import argparse
import pickle
import numpy as np
import PIL.Image as pil
import PIL.ImageOps	
import keras
from keras.applications import DenseNet121
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.applications.densenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import GlobalAveragePooling2D, Dense
from keras import optimizers
from auc_callback import AucRoc
import horovod.keras as hvd
import time

# Output produced by the experiment (summaries, checkpoints etc.) has to be placed in this folder.
EXPERIMENT_OUTPUT_PATH = "/mnt/output/experiment"


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/mnt/output/home/data/ai-radiologist/images_all/',
    help='The directory where the input data is stored.')

parser.add_argument(
    '--model_dir', type=str, default=EXPERIMENT_OUTPUT_PATH,
    help='The directory where the model will be stored.')

parser.add_argument(
    '--batch_size', type=int, default='16',
    help='Batch size for SGD')

parser.add_argument(
    '--image_size', type=int, default='256',
    help='Image size')

parser.add_argument(
    '--opt', type=str, default='adam',
    help='Optimizer to use (adam, sgd, rmsprop, adagrad, adadelta, adamax, nadam)')

parser.add_argument(
    '--momentum', type=float, default='0.0',
    help='Momentum rate for SGD optimizer')

parser.add_argument(
    '--nesterov', type=bool, default=False,
    help='Use Nesterov momentum for SGD optimizer')

parser.add_argument(
    '--lr', type=float, default='1e-3',
    help='Learning rate for optimizer')

parser.add_argument(
    '--epochs', type=int, default=15,
    help='Number of epochs to train')

parser.add_argument(
    '--train_label', type=str, default='/mnt/output/home/data/ai-radiologist',
    help='Path to the training label file')

parser.add_argument(
    '--validation_label', type=str, default='/mnt/output/home/data/ai-radiologist',
    help='Path to the validation label file')


def load_train_valid_labels(train_label,validation_label):

    with open(train_label+'/training_labels_new.pkl', 'rb') as f:
      training_labels = pickle.load(f)
    training_files = np.asarray(list(training_labels.keys()))

    with open(FLAGS.validation_label+'/validation_labels_new.pkl', 'rb') as f:
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
    img = img.resize((FLAGS.image_size, FLAGS.image_size),pil.NEAREST)
    if is_training and np.random.randint(2):
      img = PIL.ImageOps.mirror(img)
    batch_images.append(np.asarray(img))
    batch_labels.append(labels[filename])
  return preprocess_input(np.float32(np.asarray(batch_images))), np.asarray(batch_labels)

def train_generator(num_of_steps, training_files, labels):
  while True:
    np.random.shuffle(training_files)
    batch_size = FLAGS.batch_size
    for i in range(num_of_steps):
      batch_of_files = training_files[i*batch_size: i*batch_size + batch_size]
      batch_images, batch_labels = load_batch(batch_of_files, labels, True)
      yield batch_images, batch_labels

def val_generator(num_of_steps, validation_files, labels):
  while True:
    np.random.shuffle(validation_files)
    batch_size = FLAGS.batch_size
    for i in range(num_of_steps):
      batch_of_files = validation_files[i*batch_size: i*batch_size + batch_size]
      batch_images, batch_labels = load_batch(batch_of_files, labels, True)
      yield batch_images, batch_labels

def main():

  train_label=FLAGS.train_label
  validation_label=FLAGS.validation_label
  labels, training_files, validation_files = load_train_valid_labels(train_label,validation_label)

  hvd.init()

  np.random.seed(hvd.rank())

  # Horovod: print logs on the first worker.
  verbose = 2 if hvd.rank() == 0 else 0

  print("Running with the following config:")
  for item in FLAGS.__dict__.items():
    print('%s = %s' %(item[0], str(item[1])))

  base_model = DenseNet121(include_top=False,
                   weights='imagenet',
                   input_shape=(FLAGS.image_size, FLAGS.image_size, 3))
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  predictions = Dense(14, activation='sigmoid', bias_initializer='ones')(x)
  model = Model(inputs=base_model.input, outputs=predictions)

  if FLAGS.opt == 'adam': 
    opt = optimizers.Adam(lr=FLAGS.lr)
  elif FLAGS.opt == 'sgd':
    opt = optimizers.SGD(lr=FLAGS.lr, momentum=FLAGS.momentum, nesterov=FLAGS.nesterov)
  elif FLAGS.opt == 'rmsprop':
    opt = optimizers.RMSProp(lr=FLAGS.lr)
  elif FLAGS.opt == 'adagrad':
    opt = optimizers.Adagrad(lr=FLAGS.lr)
  elif FLAGS.opt == 'adadelta':
    opt = optimizers.Adadelta(lr=FLAGS.lr)
  elif FLAGS.opt == 'adamax':
    opt = optimizers.Adamax(lr=FLAGS.lr)
  elif FLAGS.opt == 'nadam':
    opt = optimizers.Nadam(lr=FLAGS.lr)
  else:
    print("No optimizer selected. Using Adam.")
    opt = optimizers.Adam(lr=FLAGS.lr)
      
  hvd_opt = hvd.DistributedOptimizer(opt)
  
  model.compile(loss='binary_crossentropy',
                         optimizer=hvd_opt,
                         metrics=['accuracy'])
  # Path to weights file
  weights_file=FLAGS.model_dir + '/lr_{:.3f}_bz_{:d}'.format(FLAGS.lr, FLAGS.batch_size) + '_loss_{val_loss:.3f}_epoch_{epoch:02d}.h5'
  
  # Callbacks
  steps_per_epoch = 77871 // FLAGS.batch_size
  val_steps = 8653 // FLAGS.batch_size
  lr_reducer =  ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0.01,
                                      cooldown=0, patience=1, min_lr=1e-15, verbose=2)
  auc = AucRoc(val_generator(val_steps, validation_files, labels), val_steps)
  model_checkpoint= ModelCheckpoint(weights_file, monitor="val_loss",save_best_only=True,
                                    save_weights_only=True, verbose=2)


  callbacks = [ 
                hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                hvd.callbacks.MetricAverageCallback(),
                keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=64),
                lr_reducer
              ]

  if hvd.rank() == 0:
    #callbacks.append(auc)
    callbacks.append(model_checkpoint)

  start_time=time.time()
  # specify training params and start training
  model.fit_generator(
  		train_generator(steps_per_epoch // hvd.size(), training_files, labels),
  		steps_per_epoch=steps_per_epoch // hvd.size(),
        	epochs=FLAGS.epochs,
       		validation_data=val_generator(3 * val_steps // hvd.size(), validation_files, labels),
  		validation_steps=3 * val_steps // hvd.size(),
  	        callbacks=callbacks,
        	verbose=verbose)
  end_time=time.time()
  print("start time: {} , end time: {} , elapsed time: {}".format(start_time,end_time,end_time-start_time))
if __name__ == '__main__':
 FLAGS, _ = parser.parse_known_args()
 main()

