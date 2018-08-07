from keras.applications.resnet50 import ResNet50
from keras import callbacks
import numpy as np
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import random

random.seed(2018)
np.random.seed(2018)

# Params
batch_size = 16
epochs = 50
classes = 6
save_file_name = 'resnet50_2.h5'

# load data
file = np.load('224_npz/split_data_shuffle1.npz')
[x_train, y_train, x_val, y_val, x_test, y_test] = [file['x_train'], file['y_train'] - 1,
                                                    file['x_val'], file['y_val'] - 1,
                                                    file['x_test'], file['y_test'] - 1]
# ## computer class_weight
# class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
# class_weight_dict = dict(enumerate(class_weight))
# print(class_weight_dict)

# data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

x_test = x_test.astype('float32') / 255.

val_x = x_val.astype('float32') / 255.
val_y = y_val

train_steps = int(len(x_train) / batch_size)
val_steps = int(len(x_val) / batch_size)

y_train = to_categorical(y_train, num_classes=classes)
y_val = to_categorical(y_val, num_classes=classes)

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

# callbacks
save_root = 'models'
save = callbacks.ModelCheckpoint(os.path.join(save_root, save_file_name),
                                 save_best_only=True,
                                 verbose=1)
early_stop = callbacks.EarlyStopping(patience=5)

tensorboard = callbacks.TensorBoard(log_dir='./tmp/2')

# model
base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)

predictions = Dense(classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary()
# train the model on the new data for a few epochs
model.fit_generator(train_generator,
                    steps_per_epoch=train_steps,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=val_steps,
                    use_multiprocessing=True,
                    callbacks=[save, early_stop, tensorboard])


model.load_weights(os.path.join(save_root, save_file_name))
# Val acc, confusion matrix
y_pred = model.predict(val_x)
y_pred = np.argmax(y_pred, axis=-1)
val_acc = accuracy_score(val_y, y_pred)
val_cm = confusion_matrix(val_y, y_pred)
print('Val acc:', val_acc)
print('Val confusion matrix:\n', val_cm)
# Test acc, confusion matrix
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=-1)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print('Test acc:', acc)
print('Test confusion matrix:\n', cm)
