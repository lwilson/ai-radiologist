import keras
from sklearn.metrics import roc_auc_score
import numpy as np

class AucRoc(keras.callbacks.Callback):
  def __init__(self,val_generator, steps):
    self.val_generator = val_generator 
    self.steps = steps

  def on_train_begin(self, logs={}):
    return

  def on_train_end(self, logs={}):
    return

  def on_epoch_begin(self, epoch, logs={}):
    return

  def on_epoch_end(self, epoch, logs={}):        
    labels = []
    probs = []
    for i in range(self.steps):
      batch_images, batch_labels = next(self.val_generator)
      probs.extend(self.model.predict_on_batch(batch_images))
      labels.extend(batch_labels)
    for i in range(14):
      print(roc_auc_score(np.asarray(labels)[:,i], np.asarray(probs)[:,i]))

  def on_batch_begin(self, batch, logs={}):
    return

  def on_batch_end(self, batch, logs={}):
    return   
