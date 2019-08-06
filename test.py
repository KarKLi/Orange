from Orange import OrangeLinear
import tensorflow as tf
model=OrangeLinear("VGG-16",(32,32))
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

# Build VGG-16
model.Convolution(64,(3,3),1,padding='same')
model.Convolution(64,(3,3),1,padding='same')
model.pooling((2,2),2,pool_type='maxpooling')

model.Convolution(128,(3,3),1,padding='same')
model.Convolution(128,(3,3),1,padding='same')
model.pooling((2,2),2,pool_type='maxpooling')

model.Convolution(256,(3,3),1,padding='same')
model.Convolution(256,(3,3),1,padding='same')
model.Convolution(256,(3,3),1,padding='same')
model.pooling((2,2),2,pool_type='maxpooling')

model.Convolution(512,(3,3),1,padding='same')
model.Convolution(512,(3,3),1,padding='same')
model.Convolution(512,(3,3),1,padding='same')
model.pooling((2,2),2,pool_type='maxpooling')

model.flatten()
model.relu(4096)
model.relu(4096)
model.dropout(0.5)
model.softmax(10)

model.ModelCompile('SGD,0.01,0.99,0.0','Sof-CE',y_labels=y_train)
model.Displaymodel()