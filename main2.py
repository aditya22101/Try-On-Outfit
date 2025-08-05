import tensorflow as tf

from tensorflow import keras
from keras.applications.resnet50 import ResNet50,preprocess_input
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from keras.layers import Dense,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
file_path = 'your_file.csv'
tu=[]
dt=[]

with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    
    # Iterate over rows and print each row
    i=1
    for row in csv_reader:
        tu=[]
        for item in row:
            tu.append(int(item))
        dt.append(tu)
        print(i)
        i+=1
df=pd.DataFrame(dt)
df=df.iloc[1:,:]
print("Shuffling....")
df=shuffle(df)
X=df.iloc[:,1:]
Y=df.iloc[:,0]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=10)
def preprocess(row):
    row=row.reshape(224,224,3)
    row=np.expand_dims(row,axis=0)
    row=preprocess_input(row)
    row.reshape(224,224,3)
    return row
X_train=np.array(X_train)
X_test=np.array(X_test)
Y_train=np.array(Y_train)
Y_test=np.array(Y_test)
tup=[]
for i in X_train:
    ele=preprocess(i)
    tup.append(ele)
X_train=np.array(tup)
tup=[]
print(X_test.shape)
for j in X_test:
    print(j)
    ele=preprocess(j)
    tup.append(ele)
X_test=np.array(tup)
X_train=np.squeeze(X_train,axis=1)
X_test=np.squeeze(X_test,axis=1)
print(X_train.shape,X_test.shape)
print("PreProcess completed")
print("Training Started....")
train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test,Y_test))
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
resnet_model=Sequential()
pretrained=ResNet50(
    include_top=False,
    weights='imagenet',
    classes=7,
    input_shape=(224,224,3),
    pooling='avg'
)
for layer in pretrained.layers:
    layer.trainable=False
resnet_model.add(pretrained)
resnet_model.add(Flatten())
resnet_model.add(Dense(512,activation='relu'))
resnet_model.add(Dense(7,activation='softmax'))

print(resnet_model.summary)
resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
epochs=10

history=resnet_model.fit(train_dataset,epochs=10,validation_data=test_dataset)
print("Training Completed")
# resnet_model.save("C:\\Users\\hp\\Desktop\\New folder (2)\\fashion_Class")
fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

