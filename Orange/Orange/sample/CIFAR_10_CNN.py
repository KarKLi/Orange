"""
CIFAR_10_CNN.py, a sample code based on OrangeLinear moudle in Orange library.
written by Kark Li. All rights reserved.
"""
from Orange import OrangeLinear
import tensorflow as tf
import numpy as np
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

model=OrangeLinear("CNN",x_train.shape[1:])

model.Convolution(32,(3,3),padding='same')
model.Convolution(32,(3,3))
model.pooling((2,2))
model.dropout(0.25)

model.Convolution(64,(3,3),padding='same')
model.Convolution(64,(3,3))
model.pooling((2,2))
model.dropout(0.25)

model.flatten()
model.relu(512)
model.dropout(0.5)
model.softmax(10)


model.ModelCompile('RMSProp,0.0001,1e-6','Sof-CE',y_labels=y_train)
model.Displaymodel()
# uncomment the below line to train your model.
#model.ModelTrain(x_train,y_train,32,60,(x_test,y_test),1,True,'CSVLogger','result.csv')
model.CSVDisplay('result.csv')