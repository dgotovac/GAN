#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from tqdm import tqdm
from numpy.random import randn
 
import pathlib
import random
import matplotlib.pyplot as plt
 
import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
 
from matplotlib.image import imread
from keras.preprocessing import image
 

 
AUTOTUNE = tf.data.experimental.AUTOTUNE
 
data_dir = 'DATA2/'
data_dir = pathlib.Path(data_dir)

data_dir_test="test/"
data_dir_test=pathlib.Path(data_dir_test)

label_names={'smoke': 0, 'nonsmoke': 1}
label_key=['smoke','nonsmoke']

all_images_test=list(data_dir_test.glob('*/*'))
all_images_test = [str(path) for path in all_images_test]
random.shuffle(all_images_test)

all_images = list(data_dir.glob('*/*'))
all_images = [str(path) for path in all_images]
random.shuffle(all_images)

all_labels_test=[label_names[pathlib.Path(path).parent.name] for path in all_images_test]
data_size=len(all_images_test)

all_labels=[label_names[pathlib.Path(path).parent.name] for path in all_images]

data_size=len(all_images)

train_test_split=(int)(data_size*0.2)

x_t=all_images_test
y_t=all_labels_test

x_train=all_images[train_test_split:]
x_test=all_images[:train_test_split]

y_train=all_labels[train_test_split:]
y_test=all_labels[:train_test_split]

IMG_SIZE=224

BATCH_SIZE = 32

def _parse_data(x,y):
   image = tf.io.read_file(x)
   image = tf.image.decode_jpeg(image, channels=3)
   image = tf.cast(image, tf.float32)
   image = (image/127.5) - 1
   image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
   
 
   return image,y

def _input_fn(x,y):
    ds=tf.data.Dataset.from_tensor_slices((x,y))
    ds=ds.map(_parse_data)
    ds=ds.shuffle(buffer_size=data_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
  
    return ds
  
train_ds=_input_fn(x_train,y_train)
validation_ds=_input_fn(x_test,y_test)
test_ds=input_fn(x_t,y_t)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False,weights='imagenet')
VGG16_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(label_names),activation='sigmoid')
model = tf.keras.Sequential([VGG16_MODEL,global_average_layer,prediction_layer])											   
model.compile(optimizer=tf.optimizers.Adam(),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=["acc"])											  
											   
history = model.fit(train_ds,
                    epochs=500, 
                    steps_per_epoch=2,
                    validation_steps=2,
                    validation_data=validation_ds)
tf.saved_model.save(model, "/model/4")
f= plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
f.show()
plt.gcf().savefig('model/acc.jpg')


# summarize history for loss
g=plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.gcf().savefig('model/loss.jpg')


