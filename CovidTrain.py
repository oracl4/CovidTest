import os
import splitfolders
import datetime

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# Global Variable
train_ratio = 0.6
tests_ratio = 0.4
input_size = (224, 224)
channel = (3, )
input_shape = input_size + channel
batch_size = 16
epoch = 15

splitfolders.ratio("Dataset_Confirm",
                   output="Dataset_Final",
                   seed=1337,
                   ratio=(train_ratio, tests_ratio),
                   group_prefix=None)

# Create Train and Validation Path
dataset_dir = 'Dataset_Final'
train_dir = os.path.join(dataset_dir, 'train')
tests_dir = os.path.join(dataset_dir, 'val')

# Image Augmentation (Pre-process)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                  rescale=1./255,
                  rotation_range=20,
                  horizontal_flip=True,
                  shear_range = 0.2,
                  fill_mode = 'nearest',
                  )
  
tests_datagen = ImageDataGenerator(
                  rescale=1./255,
                  )

# Data Generator
train_generator = train_datagen.flow_from_directory(
                        train_dir,
                        target_size=input_size,
                        batch_size=batch_size,
                        class_mode='categorical')

tests_generator = tests_datagen.flow_from_directory(
                        tests_dir,
                        target_size=input_size,
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=False)

n_class = tests_generator.num_classes
labels = list(tests_generator.class_indices.keys())
print(labels)

# Creating tf.data Format
def create_tfData(generator, input_shape):
    num_class = generator.num_classes
    tfData = tf.data.Dataset.from_generator(
        lambda: generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None
                        , input_shape[0]
                        , input_shape[1]
                        , input_shape[2]]
                       ,[None, num_class])
    )
    return tfData

train_data = create_tfData(train_generator, input_shape)
tests_data = create_tfData(tests_generator, input_shape)

from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Creating CNN Model
baseModel = ResNet152V2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# Compile the Model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# Train the Model
time_start = datetime.datetime.now()

learning_history = model.fit(x=train_data,
                             steps_per_epoch=len(train_generator),
                             epochs=epoch,
                             batch_size=batch_size,
                             validation_data=tests_data,
                             validation_steps=len(tests_generator),
                             shuffle=True
                             )

time_end = datetime.datetime.now()
time_elapsed = time_end-time_start
print("Training Time : ", time_elapsed)

# Evaluating the Model
# train_loss, train_acc = model.evaluate(train_data,
#                                        steps=len(train_generator),
#                                        verbose=0)

# tests_loss, tests_acc = model.evaluate(tests_data,
#                                        steps=len(tests_generator),
#                                        verbose=0)

# print('Training Accuracy : {:.4f} \nTraining Loss : {:.4f}'.format(train_acc, train_loss),'\n')
# print('Testing Accuracy  : {:.4f} \nTesting Loss: {:.4f}'.format(tests_acc, tests_loss),'\n')

# Save the Model Training Result
model.save("Resnet_Covid19Model.h5")

# """**Model Training Result**"""

# # Visualize training history
# fig = plt.figure(facecolor="w")

# # Accuracy History
# plt.plot(learning_history.history['accuracy'])
# plt.plot(learning_history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# fig = plt.figure(facecolor="w")
# # Loss History
# plt.plot(learning_history.history['loss'])
# plt.plot(learning_history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# from sklearn.metrics import classification_report, confusion_matrix

# #Confution Matrix and Classification Report
# Y_pred = model.predict_generator(tests_generator, steps=len(tests_generator.filenames) // batch_size+1)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(tests_generator.classes, y_pred))
# print('Classification Report')
# target_names = ['covid', 'normal', 'viral_pnemonia']
# print(classification_report(tests_generator.classes, y_pred, target_names=target_names))

# # # Commented out IPython magic to ensure Python compatibility.
# # # Predicting From Uploaded Files
# # import numpy as np
# # from google.colab import files
# # from keras.preprocessing import image
# # import matplotlib.pyplot as plt
# # import matplotlib.image as mpimg
# # # %matplotlib inline

# # uploaded = files.upload()

# # for filename in uploaded.keys():
# #   img = image.load_img(filename, target_size=input_size)
# #   imgplot = plt.imshow(img)
# #   x = image.img_to_array(img)/255
# #   x = np.expand_dims(x, axis=0)
 
# #   images = np.vstack([x])

# #   # Predict the Image Classes
# #   class_predict = model.predict(images)
# #   class_index = np.argmax(class_predict)

# #   # Remove File
# #   os.remove(filename)

# #   # Print the Result
# #   print("Filename : ", filename, "\tPredicted Class : ", labels[class_index], "\tProbability : ", np.max(class_predict))