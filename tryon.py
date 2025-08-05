import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import cv2
from numpy.linalg import norm
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16,preprocess_input as ps
from keras.layers import GlobalMaxPooling2D
from keras.models import Model

# Load VGG16 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='min')
model.trainable = False
last = model.layers[-2].output
model = Model(inputs=model.input, outputs=last)

model2=VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3),pooling='max')
model = keras.Sequential([
    model,
    GlobalMaxPooling2D()
    ])
model2 = keras.Sequential([
    model2,
    ])

# Cap_200 0.57  
#Cap_187 0.45
#Cap_190 0.46
#Cap_180 0.50
#Cap_186 0.50
#Shirt_4 0.57 Shirt_42 0.64 Shirt_


# Cap 0.45 : 0.57
im1 = cv2.imread("C:\\Users\\hp\\Desktop\\pink-cap-for-girls.jpeg")
im1 = cv2.resize(im1, (224, 224))
im2 = cv2.imread("C:\\Users\\hp\\Desktop\\download.jpeg")
im2 = cv2.resize(im2, (224, 224))
def preprocess(img):
    img = img.reshape(1, 224, 224, 3)
    img1= model.predict(img)
    img2=model2.predict(img)

    img=0.5*img1+0.5*img1
    img=img/norm(img)
    return img

y=preprocess(im2)
tp={'0':[0.45,0.65],'1':[0.48,0.67],'2':[0.37,0.57],'3':[0.44,0.58],'4':[0.38,0.57],'5':[0.42,0.60],'6':[0.46,0.7]}
def eval(imgt):
    md= keras.models.load_model("C:\\Users\\hp\\Desktop\\New folder (2)\\fashion_Class")
    imgt=imgt.reshape(1,224,224,3)
    ind=np.argmax(md.predict(imgt))
    x=np.array(preprocess(imgt))
    score=np.dot(y,x.T)
    rank=np.linspace(tp[str(ind)][0],tp[str(ind)][1],num=6)
    for i in range(len(rank)):
        if i==0 and score<rank[i]:
            return 0
        elif score==rank[i] and i==0:
            return i+1
        
        elif score<=rank[i]:
            return i
    else:
        return 5    
rk=eval(im1)
print(rk,rk*'*')
      
        









# arr=[]
# mint=-1
# maxt=0
# for i in os.listdir("C:\\Users\\hp\\Desktop\\New folder (2)\\images"):
#     mint=-1
#     maxt=0
#     for j in os.listdir("C:\\Users\\hp\\Desktop\\New folder (2)\\images\\"+i):
#         im = cv2.imread("C:\\Users\\hp\\Desktop\\New folder (2)\\images\\"+i+"\\"+j)
#         if im is None:
#             continue
#         im = cv2.resize(im, (224, 224))
#         x=preprocess(im)
#         z=np.dot(x,np.array(y).T)[0][0]
#         if (mint==-1) or z<mint:
#             mint=z
#         if z>maxt:
#             maxt=z
#     arr.append((mint,maxt,i))
# print(arr)
# [(0.450236344, 0.65202796, 'Cap'), (0.460236344, 0.698668, 'EyeWear'), (0.37804973, 0.57229686, 'Jeans'), 
# (0.42565817, 0.60312426, 'Jewellery'), (0.4887212, 0.67350423, 'Shirt'), 
# (0.38934934, 0.5756805, 'Shoe'), (0.44331813, 0.58563864, 'Watch')]





