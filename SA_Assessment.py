#Build models using Keras
import tensorflow.keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

#Build models using Keras
import keras
from keras import models
from keras import layers
from keras.layers.core import Permute
import tensorflow as tf
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
tf.enable_eager_execution() 

from PIL import Image
import glob
import cv2
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import matplotlib.pyplot as plt
%matplotlib inline

#input images and labels (labes were pre-defined with folder name)
direct = r'C:/Users/admin/Desktop/SA/' 
classes = [clas.name for clas in os.scandir(direct) if clas.is_dir()]
X, y = [], []
for clas in classes:
    path = os.path.join(direct, clas)
    for img in os.listdir(path):
        X.append(cv2.imread(os.path.join(path, img)))
        y.append(int(clas))        
print(np.asarray(X).shape)  #identify the total number of instances

#reshape X and y 
X = np.asarray(X)
X = X.reshape(n,3,224,224).transpose(0,2,3,1).astype("uint8") #update the number based on instances
y = np.array(y)

#define function for encoding label y
def one_hot_encode(vec, vals = 3):   #SA has three statuses in each level, therefore the class is 3 here
    #to one-hot encode the 4- possible labesl
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

#match images with the labels
class CifarHelper():
    def __init__(self):
        self.i = 0
        self.images = None
        self.labels = None
        
    def set_up_images(self):
        print("Setting up images and labels")
        self.images = np.vstack([X])
        all_len = len(self.images)
        self.images = self.images.reshape(n, 3, 224, 224).transpose(0,2,3,1)/255  #update like above
        self.labels = one_hot_encode(np.hstack([y]), 3)

#before tensorflow run:
ch = CifarHelper()
ch.set_up_images()

#Encoding data y  
def to_one_hot(y, dimension=3):
    results = np.zeros((len(y), dimension))
    for i, label in enumerate(y):
        results[i, label] = 1.
    return results
  
one_hot_labels = to_one_hot(y)

def load_and_preprocess_from_path_label(X, y):
    X = 2*tf.cast(X, dtype=tf.float32) / 255.-1
    y = tf.cast(y, dtype=tf.int32)
    return X, y
  
#import libraries for model training and testing  
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, Model, Sequential, regularizers
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2, random_state=42, shuffle = True) #can be changed to cross-validation

#define function, here is the architecture for Xception
def entry_flow(inputs) :
    x = Conv2D(32, 3, strides = 2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64,3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    previous_block_activation = x

    for size in [128, 256, 728] :
        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(3, strides=2, padding='same')(x)

        residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)

        x = tensorflow.keras.layers.Add()([x, residual])
        previous_block_activation = x
    return x
 
def middle_flow(x, num_blocks=8) :
    previous_block_activation = x
    for _ in range(num_blocks) :
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = tensorflow.keras.layers.Add()([x, previous_block_activation])
        previous_block_activation = x
    return x
 
def exit_flow(x) :
    previous_block_activation = x
    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x) 
    x = BatchNormalization()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
    x = tensorflow.keras.layers.Add()([x, residual])

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(3, activation='linear')(x)
    return x

#construct the model with the above functions  
inputs = Input(shape=(224,224,3))
outputs = exit_flow(middle_flow(entry_flow(inputs)))
xception = Model(inputs, outputs)

import time
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score 

model = xception
model.summary()

#record the model weights
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_filepath = 'weights.{epoch:02d}-{val_loss:.2f}.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                            monitor = 'val_accuracy',  #record the best performing-model based on validation accuracy
                                                            mode = 'max',
                                                            save_best_only=True)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),  #learning rate has to be evaluated during fine-tunning
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics = ['accuracy'])

#model fitting
history = model.fit(X_train, y_train, epochs = 30,
                   validation_data=(X_validation, y_validation), batch_size = 2, callbacks=[model_checkpoint_callback])

#load the best-performing model and save it
model.load_weights('weights.14-0.39.h5')
model.save('saved_model/xception')

#evaluate the model again with testing set if optional
test = model.predict(X_validation, batch_size = 2)
pred = np.argmax(test, axis=1)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum()

prob = []
for i in range(len(pred)):
    logit = softmax(test[i])
    prob.append(logit[1])
prob

#print the classification report and confusion matrix
print(classification_report(y_validation, pred))
confusion_matrix(y_validation, pred)

from sklearn.metrics import roc_curve, roc_auc_score, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle

#draw the auc curve based on the classes
y_val = label_binarize(y_validation, classes=[0, 1, 2])

fpr = [0]*3
tpr = [0]*3
thresholds = [0]*3
auc_score = [0]*3
n_classes = 3

for i in range(n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_val[:, i], 
                                             test[:, i])
    auc_score[i] = auc(fpr[i], tpr[i])
auc_score

fig, ax = plt.subplots(figsize = (10, 10))
target_names = ["Level 1", "Level 2", "Level 3"]
class_id = [0, 1, 2]
colors = cycle(["#508ca4", "#b7d1da", "#68b0ab"])

for class_id, color in zip(range(n_classes), colors):
    RocCurveDisplay.from_predictions(
        y_val[:, class_id],
    test[:, class_id],
    name=f"ROC curve for {target_names[class_id]}",
    color = color, ax=ax)
    
# plt.savefig('Level_1.png', dpi=300)





