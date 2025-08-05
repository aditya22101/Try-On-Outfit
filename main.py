import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
# Convert the SavedModel to TensorFlow Lite format
img=cv2.imread("C:\\Users\\hp\\Desktop\\135.jpg")
print(img.shape)
img=cv2.resize(img,(224,224))
img=img.reshape(1,224,224,3)
model = keras.models.load_model("C:\\Users\\hp\\Desktop\\New folder (2)\\fashion_Class")

pre=model.predict(img)
print(pre)
# Save the TFLite model to a file


