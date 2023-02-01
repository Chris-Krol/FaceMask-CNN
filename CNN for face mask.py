import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

tensorboard = TensorBoard(log_dir='logs'.format("FaceMaskCnn"), profile_batch=0)
img = image.load_img("Dataset/test_pred/1.png")
print(img)
plt.imshow(img)


cv2.imread("Dataset/without_mask/2.png")
train = ImageDataGenerator(rescale= 1/255)


train_dataset = train.flow_from_directory("Dataset/train", target_size= (128, 128), batch_size = 64, class_mode = 'binary')
validation_dataset = train.flow_from_directory("Dataset/validation", target_size= (128, 128), batch_size = 64, class_mode = 'binary')
test_dataset = train.flow_from_directory("Dataset/test_eval", target_size= (128, 128), batch_size = 64, class_mode = 'binary')

#creating the model for learning
model = tf.keras.models.Sequential([ 
    tf.keras.layers.Conv2D(16,(3,3),activation = 'relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, 3,activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, 3,activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation= 'relu'),
    tf.keras.layers.Dense(3)
    ])
print(model.summary())

#loss func
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#gradient descent, testing for accuracy
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"] 

model.compile(loss= loss, optimizer = optim, metrics = metrics)
history = model.fit(train_dataset, steps_per_epoch =  5, epochs = 40, validation_data = validation_dataset, callbacks=[tensorboard])

print(history.history)

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,41)

plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1,41)

plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.evaluate(test_dataset, batch_size=64)
probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax() #Use softmax layer to get probabilities
])
dir_path = "Dataset/test_pred"

#iteratively predict each image in the test dataset
for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+i, target_size=(128,128,3))
    plt.imshow(img)
    plt.show()
    
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis =0)
    images = np.vstack([X])
    curr_pred = probability_model.predict(images)[0]
    label = None
    if np.argmax(curr_pred) == curr_pred[0]:
        label = "Correctly wearing mask"
    elif np.argmax(curr_pred) == curr_pred[1]:
        label = "Incorrectly wearing mask"
    else:
        label = "Not wearing a mask"
    print(label)