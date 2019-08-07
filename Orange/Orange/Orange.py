"""
Orange -- A library based on tensorflow.keras, using for simpilfying computer vision code based on deep learning.
In Orange, you can only focus on your model's structure and don't need to focus on the code's structure.
Orange gives a library that much close to model's structure.
Orange can construct two types of network. First is linear stacked neural network such as VGGNet, AlexNet, etc.
Second is non-linear stacked neural network (may have shortcut connection like ResNet or inception-block like GoogleNet).
Orange is a abstract library based on tensorflow.keras, so it doesn't provide the basic operation of math such as gradient computation, etc.
Orange doesn't provide the interface of RNN and mbedding layer. If you want to use it in the network built by Orange, use AddLayer interface.
All the input data should follow the format of  (batch, height, width, channels)
Followed by MIT license.
"""
import os,sys
# import TensorFlow
try:
    import tensorflow as tf
except ImportError:
    raise ImportError('Please install TensorFlow first.')

class VersionError(BaseException):
    """Just an error, never mind.
"""
    pass
class ExecutionError(BaseException):
    """Just an error, never mind.
"""
    pass

class OrangeLinear(object):
    __version__='1.0.0'

    def __init__(self,model_name,input_shape):
        """__init__ accept two parameters, model_name and input_shape,
    @ param:
    model_name : Model's name, such as ResNet-18, ResNet-34. This model name is a symbol to recognize the model.
    input_shape : Regulate the input vector's shape.
    model : The tf.keras.Sequential() model, which is a stack of layers.
    """
        version=list(tf.__version__.split('.'))
        if int(sys.version[0]) < 3:
            raise VersionError('The python version must 3, please check your execute enviornment.')
        if version[0]!='1' or int(version[1]) < 7:
            raise VersionError('TensorFlow imported, but the version need to update. Please update your TensorFlow to 1.7.0 or higher.')
        if tf.test.is_built_with_cuda() is not True:
            raise VersionError('Orange only support tensorflow-gpu library, please check your library.')
        if tf.test.is_gpu_available() is not True:
            raise VersionError('Orange need to run on the machine with NVIDIA GPU, please check your device.')
        self.__model_name=model_name
        self.__shape=input_shape
        self.__model=tf.keras.Sequential()
        self.__y_label=None
        print('Orange initialization complete.')
        self.__modelhasbuilt=False

    def GetVersion(self):
        """GetVersion returns the Orange library's version.
    """
        return __version__

    def Getmodelname(self):
        """Getmodelname returns the name of model.
    """
        return self.__model_name

    def GetInputShape(self):
        """Return the self.__shape
    """
        return self.__shape

    def Displaymodel(self):
        """Print the model'summary by using the bulit-in function tf.keras.Sequential.summary()
    """
        if self.__modelhasbuilt is True:
            self.__model.summary()
        else:
            raise ExecutionError('Please compile your model first.')

    """
    First part of the Orange library : Regularizers, optimizers, loss and noise.
    """
    def __L1Regularizer(self,init_param):
        """L1 Regularizer, put a linear regularizer into Dense/Conv2D layer. This function doesn't use it separately. So just set it private.
    The regularizer can only be used in FC layer and Convolution layer. There are three key argument avaliable in using regularizer :
    kernel_regularizer, bias_regularizer and activity_regularizer
    """
        return tf.keras.regularizers.l1(init_param)

    def __L2Regularizer(self,init_param):
        """L2 Regularizer, put a square regularizer into Dense/Conv2D layer. This function doesn't use it separately. So just set it private.
    The regularizer can only be used in FC layer and Convolution layer. There are three key argument avaliable in using regularizer :
    kernel_regularizer, bias_regularizer and activity_regularizer
    """
        return tf.keras.regularizers.l2(init_param)

    def __L1L2Regularizer(self,init_param):
        """L1-L2 Regularizer, put a square regularizer and add a linear regularizer into Dense/Conv2D layer simultaneously. 
    This function doesn't use it separately. So just set it private.
    The regularizer can only be used in FC layer and Convolution layer. There are three key argument avaliable in using regularizer :
    kernel_regularizer, bias_regularizer and activity_regularizer
    """
        return tf.keras.regularizers.l1_l2(init_param)

    def __SGD(self,lr=0.01,momentum=0.0,decay=0.0,nesterov=False):
        """SGD optimizers, used in model compilcation.
    @ param:
    lr -- learning rate
    momentum -- A parameter that acclerates the SGD go forward in a direction, and it can also suppress the concussion on the area of the saddle point.
    decay -- the decay ratio of learning rate.
    nesterov -- Using nesterov momentum or not. Default parameter is False.
    """
        return tf.keras.optimizers.SGD(lr,momentum,decay,nesterov)

    def __RMSProp(self,lr=0.001,rho=0.9,epsilon=None,decay=0.0):
        """RMSProp optimizers, used in model compilcation.
    @ param:
    lr -- learning rate
    rho -- a postive decay ratio of the movement average of the RMSProp's gradient's square.
    epsilon -- Set None if you don't know what value should be passed. The default value is K.epsilon()
    decay -- the decay ratio of learning rate.
    For more information, see the paper below:
    rmsprop: Divide the gradient by a running average of its recent magnitude
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
        return tf.keras.optimizers.RMSprop(lr,rho,epsilon,decay)

    def __Adam(self,lr=0.001,beta_1=0.9,beta_2=0.99,epslion=None,decay=0.0,amsgrad=False):
        """Adam optimizers, used in model complication.
    @ param:
    lr -- learning rate
    beta_1 -- 0 < beta < 1, always close to 1, don't change the default value in general.
    beta_2 -- 0 < beta < 1, always close to 1, don't change the default value in general.
    epsilon -- fusion factor, be set as K.epsilon() if you not change the default value.
    decay -- the decay ratio of learning rate.
    amsgrad -- Use the algorithm of AMSGrad, for more information, see the paper below:
    On the Convergence of Adam and Beyond
    https://openreview.net/forum?id=ryQu7f-RZ
    For more information of Adam optimizer, see the paper below:
    Adam - A Method for Stochastic Optimization
    http://arxiv.org/abs/1412.6980v8
    """
        return tf.keras.optimizers.Adam(lr,beta_1,beta_2,epslion,decay,amsgrad)

    def __MSE():
        """Mean squared error (MSE) is a loss function used in model complication.
    @ formula:
    MSE(bar(theta))=E(bar(theta)-theta)^2
    After equal transformation, the MSE is equal to D(theta)+[E(theta)-theta)]^2

    MSE has two parts, first is the theta's deviation, and the second is the distance between exception value and the truth value.
    It is L2 loss.
    If the estimation of theta is unbiased estimate, then the MSE(theta) is equal to its deviation.
    """
        return tf.keras.losses.mean_squared_error

    def __MAE():
        """Mean absolute error (MAE) is a loss function used in model complication.
    @ formula:
    for a set of data {x_1,x_2,...,x_n}, the MAE is defined as:
    MAE=1/n*(\sigma_{i=1}^n{x_i-m(x)}) which the m(x) is the truth value.

    MAE is L1 loss, it treat all deviation as equal and doesn't take harsh punishment to the data which has large deviation.
    """
        return tf.keras.losses.mean_absolute_error
    
    def __MAPE():
        """Mean absolute percentage error (MAPE) is a loss function used in model complication.
    @ formula:
    MAPE=100%/n*(\sum_{i=1}^n{(bar(y_i)-y_i)/y_i}

    MAPE is a loss ratio of all data. If MAPE=0%, the model is a perfect model. If MAPE >= 1, the model is an awful model.
    """
        return tf.keras.losses.mean_absolute_percentage_error
    
    def __hinge(squared=False,categorical=False):
        """Hinge loss is a loss function used in model complication.
    @ formula:
    min_{W,b}\sum_{i=1}^N\sum_{j not equal to y_i}max(0,s_j-s_{y_i}+1)+\lambda \sum_k \sum_n W_{k,n}^2

    @ param:
    squared -- Use squared_hinge loss.
    categorical -- Use categorical_hinge loss.

    Hinge loss always used in SVM task. If you want to use squared hinge loss, set the sqaured parameter as True.
    If you want to use categorical hinge loss, set the categorical parameter as True.
    DON'T set the parameters True at the same time!
    """
        if squared is False and categorical is False:
            return tf.keras.losses.hinge
        elif squared is True and categorical is False:
            return tf.keras.losses.squared_hinge
        elif squared is False and categorical is True:
            return tf.keras.losses.categorical_hinge
        else:
            raise ValueError('Not sure what hinge loss you need to use ! (Squared hinge or Categorical hinge?)')

    def __logcosh():
        """Log-cosh loss function is a loss function used in model complication.
    @ formula:
    L(y,y^p)=\sum_{i=1}^n log(cosh(y_i^p-y_i))

    It has this mathematical nature:
    log(cosh(x)) \t equals to x^2/2         \t when x is small
                 \t equals to abs(x)-log(2) \t when x is large
    logcosh is approximately equal to MSE and decay the crazy prediction's effect.
    """
        return tf.keras.losses.logcosh

    def __crossentropy(self,Sigmoid=False,Softmax=False,Sparse=False,y_labels=None):
        """Cross entropy loss function is a loss function used in model complication.
    @ param:
    Sigmoid -- Will use binary_crossentropy if the activate function is Sigmoid
    Softmax -- Will use categorical_crossentropy if the activate function is Softmax
    Sparase -- Will use sparse_categorical_crossentropy if the dataset has many zero data.

    You don't need to do preprocess of the label. At categorical_crossentropy and sparse_categorical_crossentropy, the crossentropy function
    will do the preprocessing. But you need to pass the y_labels to process. If you don't want Orange to process your label, pass the y_labels None.
    """
        if Sigmoid is True and Sparse is False:
            return tf.keras.losses.binary_crossentropy
        elif Softmax is True and Sparse is False and y_labels is not None:
            from keras.utils.np_utils import to_categorical
            self.__y_label=to_categorical(y_labels)
            return tf.keras.losses.categorical_crossentropy
        elif Softmax is True and Sparse is True and y_labels is not None:
            import numpy as np
            self.__y_label=np.expand_dims(y_labels,-1)
            return tf.keras.losses.sparse_categorical_crossentropy
        elif Softmax is True and Sparse is False and y_labels is None:
            return tf.keras.losses.categorical_crossentropy
        elif Softmax is True and Sparse is True and y_labels is None:
            return tf.keras.losses.sparse_categorical_crossentropy
        else:
            raise ValueError('Pass invalid parameters to __crossentropy!')

    def GaussianNoise(self,stddev,input_shape=None):
        """Gaussian Noise layer. Add a noise which its distribution follow the Gaussian distribution.
    @ formula:
    f(x)=1/(sqrt(2*pi)*mu)*exp(-(x-mu)^2/(2*sigma^2))
    @ param:
    stddev -- the standard deviation of the noise, equals to the sigma in Gaussian distribution.
    input_shape -- None if you didn't set the noise layer as the first layer. Otherwise you should set a tuple of the data.
    """
        if input_shape is None:
            try:
                self.__model.get_layer(index=0)
            except ValueEroor:
                raise ValueError('The model doesn\'t have any layer. Set the input_shape if you want to add this layer as the first layer of the network.')
        else:
            self.__model.add(tf.keras.layers.GaussianNoise(stddev,input_shape=input_shape))

    def GaussianDropout(self,rate,input_shape=None):
        """Gaussian Dropout layer. A dropout layer which dropout probability follow the Gaussian distribution.
    @ formula:
    f(x)=1/(sqrt(2*pi)*mu)*exp(-(x-mu)^2/(2*sigma^2))
    @ param:
    rate -- the drop rate.
    input_shape -- None if you didn't set the noise layer as the first layer. Otherwise you should set a tuple of the data.
    The standard deviation (sigma) of this layer equals to sqrt(rate/(1-rate))
    For more information of dropout, see the paper below :
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014
    http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    """
        if input_shape is None:
            try:
                self.__model.get_layer(index=0)
            except ValueEroor:
                if self.__shape is None:
                    raise ValueError('The model doesn\'t have any layer. Set the input_shape if you want to add this layer as the first layer of the network.')
                else:
                    self.__model.add(tf.keras.layers.GaussianDropout(rate,input_shape=self.__shape))
        else:
            self.__model.add(tf.keras.layers.GaussianDropout(rate,input_shape=input_shape))

    """
    Second part of the Orange library : common layers in neural network.
    Include fully-connected layer, flatten layer (preprocess layer), batch normalization, pooling and convolution layer.
    At most of fully-connected layer, the activate function is:relu,leaky relu, tanh, softmax and sigmoid.
    The Orange only provide convolution operation which the kernel is two-dimensional.
    """
    def AddLayer(self,layer):
        """AddLayer provides the way to add your own tf.keras.layers that the Orange doesn't provide.
    @ param:
    layer : A tf.keras.layers object.
    """
        if isinstance(layer,type(tf.keras.layers)) is not True:
            print('The layer passed in is not tf.keras.layers, check your object\'s type.')
        else:
            self.__model.add(layer)

    def flatten(self,input_shape=None):
        """Flatten accept most 1 parameter, the flatten layer will convert a matrix into a one-dimensional vector.
    @ param:
    input_shape : The input data's shape (None if mentioned on __init__)
    """
        if input_shape is not None:
            self.__model.add(tf.keras.layers.Flatten(input_shape=input_shape))
            self.__shape=input_shape
        else:
            self.__model.add(tf.keras.layers.Flatten(input_shape=self.__shape))

    def dropout(self,ratio):
        """Dropout accept 1 parameter. The function of dropout is kill some neurals randomly to avoid over-fitting.
    @ param :
    ratio -- The drop ratio of neural units.
    """
        self.__model.add(tf.keras.layers.Dropout(ratio))
    
    def relu(self,neural_numbers):
        """relu accept 1 parameter, the relu layer will activate the neural unit.
    @ formula:
    f_{relu}(x) = max(0,x)
    @ param:
    neural_numbers : The number of neural units you want to activate.
    """
        firstlayer=False
        try:
            self.__model.get_layer(index=0)
        except ValueError:
            firstlayer=True
        if firstlayer is True:
            self.__model.add(tf.keras.layers.Dense(neural_numbers,activation=tf.nn.relu,input_shape=self.__shape))
        else:
            self.__model.add(tf.keras.layers.Dense(neural_numbers,activation=tf.nn.relu))

    def leaky_relu(self,neural_numbers):
        """
    leaky_relu accept 1 parameter, the leaky relu layer will activate the neural unit.
    @ formula:
    f_{leaky_relu}(x) = x when x > 0
                        \lambda x when x < 0 or = 0
    @ param:
    neural_numbers : The number of neural units you want to activate.
    """
        firstlayer=False
        try:
            self.__model.get_layer(index=0)
        except ValueError:
            firstlayer=True
        if firstlayer is True:
            self.__model.add(tf.keras.layers.Dense(neural_numbers,activation=tf.nn.leaky_relu,input_shape=self.__shape))
        else:
            self.__model.add(tf.keras.layers.Dense(neural_numbers,activation=tf.nn.leaky_relu))

    def tanh(self,neural_numbers):
        """tanh accept 1 parameter, the tanh layer will activate the neural unit.
    @ formula:
    f_{tanh}(x) = tanh(x) = (exp(x)-exp(-x)) / (exp(x)+exp(-x))
    @ param:
    neural_numbers : The number of neural units you want to activate.
    """
        firstlayer=False
        try:
            self.__model.get_layer(index=0)
        except ValueError:
            firstlayer=True
        if firstlayer is True:
            self.__model.add(tf.keras.layers.Dense(neural_numbers,activation=tf.nn.tanh,input_shape=self.__shape))
        else:
            self.__model.add(tf.keras.layers.Dense(neural_numbers,activation=tf.nn.tanh))

    def sigmoid(self,neural_numbers):
        """sigmoid accept 1 parameter, the sigmoid layer will activate the neural unit like relu.
    You can also use sigmoid on the lasted layer and set the neural number by 2 to output the result if your task is binary-
    classfication task.
    @ formula:
    f_{sigmoid}(x) = 1 / (1+exp(-x))
    @ param:
    neural_numbers : The number of neural units you want to activate.
    """
        firstlayer=False
        try:
            self.__model.get_layer(index=0)
        except ValueError:
            firstlayer=True
        if firstlayer is True:
            self.__model.add(tf.keras.layers.Dense(neural_numbers,activation=tf.nn.sigmoid,input_shape=self.__shape))
        else:
            self.__model.add(tf.keras.layers.Dense(neural_numbers,activation=tf.nn.sigmoid))

    def softmax(self,neural_numbers):
        """softmax accept 1 parameter, softmax layer is always at the final layer in classfication task 
    to output the possibilities of each classes.
    @ formula:
    \sigma(z)_j = exp(z_j) / (_sum_{k=1}^K exp(z_k))
    @ param:
    neural_numbers : The number of neural units you want to activate. In classfication task, it equals to the number of 
    classes generally.
    """
        firstlayer=False
        try:
            self.__model.get_layer(index=0)
        except ValueError:
            firstlayer=True
        if firstlayer is True:
            self.__model.add(tf.keras.layers.Dense(neural_numbers,activation=tf.nn.softmax,input_shape=self.__shape))
        else:
            self.__model.add(tf.keras.layers.Dense(neural_numbers,activation=tf.nn.softmax))
    
    def bn(self,axis=1,momentum=0.99,epslion=1e-3):
        """bn accept three parameters, which all have default values. This three parameters are the most common parameter in
    batch normalization opertaion.
    For more details about batch normalization, see : 
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    https://arxiv.org/abs/1502.03167
    @ param :
    axis : The normalization direction, after convolutional operation, it always equals to 1.
    momentum : The momentum of batch normalization, always close to 1.
    epslion : A small value that avoid division by zero if the data is zero.
    """
        self.__model.add(tf.keras.layers.BatchNormalization(axis=axis,momentum=momentum,epslion=epslion))
    
    def Convolution(self,output_depth,kernel_size,x_stride=1,y_stride=None,activation='relu',padding='valid'):
        """Convolution is one of the most important operations in neural network. After convolutional operation the image can
    get the new matrix set which called 'feature map'.
    @ param :
    output_depth : The depth of output feature map
    kernel_size : The convolutional kernel size. Generally, it's a tuple with two integers (x,y). If you want to
    have the same size on x-axis and y-axis, just pass one integer.
    stride : The stride of kernel move. Passing one integer in general. If you want to have different stride on different
    direction, pass a tuple with two integers (x,y).
    activation : The activation function, default value is 'relu',if you don't want to use activation function, set activation as None manully.
    padding : Padding zero before convolution operation or not. The default value is 'valid', which means not padding zero.
    Pass 'same' if you want to padding zero.
    """
        firstlayer=False
        try:
            self.__model.get_layer()
        except ValueError:
            firstlayer=True

        if firstlayer is False:
            if activation is not None:
                if y_stride is not None:
                    self.__model.add(tf.keras.layers.Conv2D(filters=output_depth,kernel_size=kernel_size,
                                           strides= (x_stride,y_stride),activation='relu',
                                           padding=padding
                                           ))
                else:
                    self.__model.add(tf.keras.layers.Conv2D(filters=output_depth,kernel_size=kernel_size,
                                           strides=x_stride,activation='relu',
                                           padding=padding
                                           ))
            else:
                if y_stride is not None:
                    self.__model.add(tf.keras.layers.Conv2D(filters=output_depth,kernel_size=kernel_size,
                                           strides= (x_stride,y_stride),
                                           padding=padding
                                           ))
                else:
                    self.__model.add(tf.keras.layers.Conv2D(filters=output_depth,kernel_size=kernel_size,
                                           strides=x_stride,
                                           padding=padding
                                           ))
        else:
            if activation is not None:
                if y_stride is not None:
                    self.__model.add(tf.keras.layers.Conv2D(filters=output_depth,kernel_size=kernel_size,
                                           strides= (x_stride,y_stride),activation='relu',
                                           padding=padding,input_shape=(self.__shape[0],self.__shape[1],3)
                                           ))
                else:
                    self.__model.add(tf.keras.layers.Conv2D(filters=output_depth,kernel_size=kernel_size,
                                           strides=x_stride,activation='relu',
                                           padding=padding,input_shape=(self.__shape[0],self.__shape[1],3)
                                           ))
            else:
                if y_stride is not None:
                    self.__model.add(tf.keras.layers.Conv2D(filters=output_depth,kernel_size=kernel_size,
                                           strides= (x_stride,y_stride),
                                           padding=padding,input_shape=(self.__shape[0],self.__shape[1],3)
                                           ))
                else:
                    self.__model.add(tf.keras.layers.Conv2D(filters=output_depth,kernel_size=kernel_size,
                                           strides=x_stride,
                                           padding=padding,input_shape=(self.__shape[0],self.__shape[1],3)
                                           ))

    def pooling(self,pool_size,strides=None,padding='valid',pool_type='maxpooling'):
        """Pooling layer is a layer to highlight the feature of feature map.
    @ param :
    pool_size : The size of pooling kernel, which is same as convolution operation. Pass a tuple if you want to pooling
    differently at different direction.
    stride : The stride of kernel move. Passing one integer in general. If you want to have different stride on different
    direction, pass a tuple with two integers (x,y).
    padding : Padding zero before convolution operation or not. The default value is 'valid', which means not padding zero.
    Pass 'same' if you want to padding zero.
    pool_type : pass 'maxpooling' if you want to execute MaxPooling.
                pass 'averagepooling' if you want to execute AveragePooling.
                pass 'globalpooing' if you want to execute GlobalPooling.
    if you pass 'globalpooing', all other parameters will be treated as invaild.
    """
        if pool_type == 'maxpooling':
            self.__model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size,strides=strides,padding=padding))
        elif pool_type == 'averagepooling':
            self.__model.add(tf.keras.layers.AveragePooing2D(pool_size=pool_size,strides=strides,padding=padding))
        elif pool_type == 'globalpooing':
            self.__model.add(tf.keras.layers.GlobalPooing2D())
        else:
            print('No such argument called : '% pool_type)

    """
    Third part of the Orange library : Data augmentation and preprocessing,callback function and model complication.
    """
    def DataAugmentation(x_train=None,
                         FeatureAverageSetZero=False,
                         SampleAverageSetZero=False,
                         FeatureDivideStddev=False,
                         SampleDivideStddev=False,
                         RotationRange=0,
                         WidthShiftRange=0.0,
                         HeightShiftRange=0.0,
                         ZoomRange=0.0,
                         HorizontalFilp=False,
                         VerticalFilp=False,
                         ReScale=None,
                         ValidationSplit=0,
        ):
        """Before other preprocessing function, you should always call DataAugmentation() first.
        DataAugmentation() doesn't enlarge the data immediately. After you call other function based on the object of ImageDataGenerator class,
        the change will be applied. Returns a tf.keras.preprocessing.image.ImageDataGenerator class object.
    @ param:
    x_train -- The input data should be passed when FeatureAverageSetZero/SampleAverageSetZero/FeatureDivideStddev/SampleDivideStddev is True.
    FeatureAverageSetZero -- Set each input 's average to 0, execute by each feature.
    SampleAverageSetZero -- Set each sample 's average to 0.
    FeatureDivideStddev -- Dividing each input by its standard deviation, execute by each feature.
    SampleDivideStddev -- Dividing each sample by its standard deviation.
    RotationRange -- An integer of the rotation angle range. The range is equal to [0,RotationRange]
    WidthShiftRange -- You can pass a float/a 1-d array/an integer for image width shifting.
                       \t Pass a float : Divide the width of image if pass a float < 1. Divide the pixel value if pass a float >=1
                       \t Pass a 1-d array : Divide random element of the array.
                       \t Pass an integer : The integer value may be chosen randomly at the range of (-WidthShiftRange,WidthShiftRange).
    HeightShiftRnage -- The possible values are same as WidthShiftRange. This parameter is used for image height shifting.
    ZoomRange -- A float or a list which shows the zoom range. If pass a float, the range is equal to [1-ZoomRange,1+ZoomRange].
    HorizontalFilp -- Horizontal filp randomly.
    VerticalFilp -- Vertical filp randomly.
    ReScale -- The rescale factor. Image will have no change if pass None or 0. Otherwise the data will mutiple the rescale factor before any other convertion.
    ValidationSplit -- The split ration of validation image. Must a float between 0 and 1.
    """
        datagen=tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=FeatureAverageSetZero,
                                                                samplewise_center=SampleAverageSetZero,
                                                                featurewise_std_normalization=FeatureDivideStddev,
                                                                samplewise_std_normalization=SampleDivideStddev,
                                                                rotation_range=RotationRange,
                                                                width_shift_range=WidthShiftRange,
                                                                height_shift_range=HeightShiftRange,
                                                                zoom_range=ZoomRange,
                                                                horizontal_filp=HorizontalFilp,
                                                                vertical_filp=VerticalFilp,
                                                                rescale=ReScale,
                                                                validation_spilt=ValidationSplit
            )
        if FeatureAverageSetZero is True or SampleAverageSetZero is True or FeatureDivideStddev is True or SampleDivideStddev is True:
            datagen.fit(x_train)
        return datagen

    def TransformForSingleImage(img,generator=None):
        """TransfromForSingleImage, get a generator set by DataAugmentation() and apply it on single image. Returns an image with the same size of input.
    @ param:
    img -- The image need to be transformed.
    generator -- The regulation of transformation, generated by DataAugmentation() or tf.keras.preprocessing.image.ImageDataGenerator.
    """
        if generator is None:
            ExecutionError('The generator is invalid!')
        if isinstance(generator,tf.keras.preprocessing.image.ImageDataGenerator):
            img_after_transform=generator.apply_transform(img)
        else:
            ExecutionError('Please call DataAugmentation() first!')

        return img_after_transform

    def __Callback(self,param):
        """Callback function, a function which used in model execution.
    The param is a list which can have this values:
    @ param:
    param -- a list contains the callback functions.
    'NaN', TerminateOnNaN, if the Loss is too big, stop the training process.
    'SaveCheckpoint', ModelCheckpoint, save the model after a epoch. If you pass the SaveCheckPoint and other callbacks, the next element of the param must be a dict which set 
    the parameters of ModelCheckpoint or callbacks.
    'EarlyStopping', EarlyStopping, stops the training process when the monitored index doesn't increase/decrease.
    'LearningRateChange', LearningRateScheduler, changes the lr by your own regulation (accomplished by customized function).
    'TensorBoard'. TensorBoard, uses the visualization tools provided by TensorFlow.
    'CSVLogger', CSVLogger, saves the result to the CSV file.
    'SaveLoss' saves the loss during the training process.
    'OwnCallback' use your own callback function, after this string you should pass your own function directly.
    """
        i=0
        if len(param) == 0:
            return param
        params=[]
        while i<len(param):
            if param[i] == 'NaN':
                params.append(tf.keras.callbacks.TerminateOnNaN())
                i += 1
            elif param[i] ==  'SaveCheckpoint':
                TP=[]
                if param[i+1].get('filepath') is None:
                    raise ValueError('You doesn\'t pass the save file path!')
                else:
                    TP.append(param[i+1].get('filepath'))
                if param[i+1].get('monitor') is None:
                    TP.append('val_loss')
                else:
                    TP.append(param[i+1].get('monitor'))
                if param[i+1].get('verbose') is None:
                    TP.append(0)
                else:
                    TP.append(param[i+1].get('verbose'))
                if param[i+1].get('save_best_only') is None:
                    TP.append(False)
                else:
                    TP.append(param[i+1].get('save_best_only'))
                if param[i+1].get('save_weights_only') is None:
                    TP.append(False)
                else:
                    TP.append(param[i+1].get('save_weights_only'))
                if param[i+1].get('mode') is None:
                    TP.append('auto')
                else:
                    TP.append(param[i+1].get('mode'))
                if param[i+1].get('period') is None:
                    TP.append(1)
                else:
                    TP.append(param[i+1].get('period'))
                params.append(tf.keras.callbacks.ModelCheckpoint(TP[0],TP[1],TP[2],TP[3],TP[4],TP[5],TP[6],TP[7]))
                i+=2
            elif param[i] == 'EarlyStopping':
                TP=[]
                if param[i+1].get('monitor') is None:
                    TP.append('val_loss')
                else:
                    TP.append(param[i+1].get('monitor'))
                if param[i+1].get('min_delta') is None:
                    TP.append(0)
                else:
                    TP.append(param[i+1].get('min_delta'))
                if param[i+1].get('patience') is None:
                    TP.append(0)
                else:
                    TP.append(param[i+1].get('patience'))
                if param[i+1].get('verbose') is None:
                    TP.append(0)
                else:
                    TP.append(param[i+1].get('verbose'))
                if param[i+1].get('mode') is None:
                    TP.append('auto')
                else:
                    TP.append(param[i+1].get('mode'))
                if param[i+1].get('baseline') is None:
                    TP.append(None)
                else:
                    TP.append(param[i+1].get('baseline'))
                if param[i+1].get('restored_best_weights') is None:
                    TP.append(False)
                else:
                    TP.append(param[i+1].get('restored_best_weights'))
                params.append(tf.keras.callbacks.EarlyStopping(TP[1],TP[2],TP[3],TP[4],TP[5],TP[6],TP[7]))
                i+=2
            elif param[i] == 'LearningRateChange':
                TP=[]
                if param[i+1].get('schedule') is None:
                    raise ValueError('Please pass a changing function when you call LearningRateChange')
                else:
                    TP.append(param[i+1].get('schedule'))
                if param[i+1].get('verbose') is None:
                    TP.append(0)
                else:
                    TP.append(param[i+1].get('verbose'))
                params.append(tf.keras.callbacks.LearningRateScheduler(TP[0],TP[1]))
                i+=2
            elif param[i] == 'TensorBoard':
                TP=[]
                if param[i+1].get('log_dir') is None:
                    TP.append('./logs')
                else:
                    TP.append(param[i+1].get('log_dir'))
                if param[i+1].get('histogram_freq') is None:
                    TP.append(0)
                else:
                    TP.append(param[i+1].get('histogram_freq'))
                if param[i+1].get('batch_size') is None:
                    TP.append(32)
                else:
                    TP.append(param[i+1].get('batchsize'))
                if param[i+1].get('write_graph') is None:
                    TP.append(True)
                else:
                    TP.append(param[i+1].get('write_graph'))
                if param[i+1].get('write_grads') is None:
                    TP.append(False)
                else:
                    TP.append(param[i+1].get('write_grads'))
                if param[i+1].get('write_images') is None:
                    TP.append(False)
                else:
                    TP.append(param[i+1].get('write_images'))
                params.append(tf.keras.callbacks.TensorBoard(TP[0],TP[1],TP[2],TP[3],TP[4],TP[5],TP[6]))
                i+=2
            elif param[i] == 'CSVLogger':
                params.append(tf.keras.callbacks.CSVLogger(param[i+1]))
                i+=2
            elif param[i] == 'SaveLoss':
                class LossHistory(keras.callbacks.Callback):
                    def on_train_begin(self, logs={}):
                        self.losses = []

                    def on_batch_end(self, batch, logs={}):
                        self.losses.append(logs.get('loss'))

                params.append(LossHistory())
                i+=1
            elif param[i] == 'OwnCallback':
                params.append(param[i+1])
        return params

    def ModelCompile(self,optimizer,loss,metrics=None,y_labels=None):
        """Compile the model with designated optimizer and loss.
    @ param:
    optimizer -- There are three options of optimizer, Adam, SGD and RMSProp, call these optimizers followed by the rule below:
    \t 'Adam,lr,decay' which the lr and decay are truth value.
    \t 'SGD,lr,momentum,decay' which the lr, momemtum and decay are truth value.
    \t 'nes-SGD,lr,momentum,decay' if you want to use nesterov momemtum.
    \t 'RMSProp,lr,decay' which the lr and decay are truth value.
    loss -- There are six options of optimizer, MAE, MAPE, MSE, Crossentropy, Hinge and logcosh
    possible value: 'MAE', 'MAPE', 'MSE', 'Sig-CE', 'Sof-CE', 'SP-Sof-CE', 'hinge', 'S-hinge', 'C-hinge', 'logcosh',
    which is Mean Average Error, Mean Average Percentage Error, Mean Squared Error, Binary Crossentropy, Categorical Crossentropy,
    Sparse Categorical Crossentropy, Hinge, Squared Hinge, Categorical Hinge, logcosh function.
    If the loss is 'SP-Sof-CE', you should also pass the y_labels into this function for label preprocessing.
    """
        opt=None
        if 'Adam' in optimizer:
            param=list(optimizer.split(','))
            param=[param[0]]+[float(i) for i in param[1:]]
            opt=self.__Adam(lr=param[1],decay=param[2])
        elif 'SGD' in optimizer and 'nes-SGD' not in optimizer:
            param=list(optimizer.split(','))
            param=[param[0]]+[float(i) for i in list(param[1:])]
            opt=self.__SGD(lr=param[1],momentum=param[2],decay=param[3])
        elif 'nes-SGD' in optimizer:
            param=list(optimizer.split(','))
            param=[param[0]]+[float(i) for i in param[1:]]
            opt=self.__SGD(param[1],param[2],param[3],True)
        elif 'RMSProp' in optimizer:
            param=list(optimizer.split(','))
            param=[param[0]]+[float(i) for i in param[1:]]
            opt=self.__RMSProp(lr=param[1],decay=param[2])

        lossfunc=''
        if loss == 'MAE':
            lossfunc=self.__MAE()
        elif loss == 'MAPE':
            lossfunc=self.__MAPE()
        elif loss == 'MSE':
            lossfunc=self.__MSE()
        elif loss == 'Sig-CE':
            lossfunc=self.__crossentropy(Sigmoid=True)
        elif loss == 'Sof-CE':
            lossfunc=self.__crossentropy(Softmax=True,y_labels=y_labels)
        elif loss == 'SP-Sof-CE':
            lossfunc=self.__crossentropy(Softmax=True,Sparse=True,y_labels=y_labels)
        elif loss == 'hinge':
            lossfunc=self.__hinge()
        elif loss == 'S-hinge':
            lossfunc=self.__hinge(squared=True)
        elif loss == 'C-hinge':
            lossfunc=self.__hinge(categorical=True)
        elif loss == 'logcosh':
            lossfunc=self.__logcosh()

        if metrics is None:
            self.__model.compile(optimizer=opt,loss=lossfunc,metrics=['accuracy'])
            self.__modelhasbuilt=True
        else:
            self.__model.compile(optimizer=opt,loss=lossfunc,metrics=[metrics])
            self.__modelhasbuilt=True
    
    def ModelTrain(self,data,label,batch_size=None,epochs=1,validation_data=None,verbose=1,callbacks=False,*args):
        """ Train the model with the specify data.
    @ param :
    data -- The input data
    label -- The input data's label.
    batch_size -- A batch's size.
    epochs -- The epoch of training process.
    verbose -- The verbose mode, 0 -- quiet mode, 1 -- progress bar, 2 -- a line per epoch
    validation_data -- The validation data after a epoch's training.
    callbacks -- The callback function you want to call.
    *args -- The function name you want to call and the parameter you want to pass. (For 'OwnCallback', the next parameter is the function directly.)

    'NaN', TerminateOnNaN, if the Loss is too big, stop the training process.
    'SaveCheckpoint', ModelCheckpoint, save the model after a epoch. If you pass the SaveCheckPoint and other callbacks, the next element of the param must be a dict which set 
    the parameters of ModelCheckpoint or callbacks.
    'EarlyStopping', EarlyStopping, stops the training process when the monitored index doesn't increase/decrease.
    'LearningRateChange', LearningRateScheduler, changes the lr by your own regulation (accomplished by customized function).
    'TensorBoard'. TensorBoard, uses the visualization tools provided by TensorFlow.
    'CSVLogger', CSVLogger, saves the result to the CSV file.
    'SaveLoss' saves the loss during the training process.
    'OwnCallback' use your own callback function, after this string you should pass your own function directly.

    Sample : model.ModelTrain(x,y,32,20,(x_test,y_test),True,'EarlyStopping',
                              {'monitor':'val_loss','verbose':1})
    """
        if self.__y_label is None:
            cb=list(args)
            self.__model.fit(data,label,batch_size,epochs,verbose,self.__Callback(cb),validation_data=validation_data)
        else:
            from keras.utils.np_utils import to_categorical
            y_test=to_categorical(validation_data[1])
            cb=list(args)
            self.__model.fit(data,self.__y_label,batch_size,epochs,verbose,self.__Callback(cb),validation_data=(validation_data[0],y_test))
        
    def ModelEvaluate(self,data,label,batch_size=None,verbose=0,steps=None):
        """ModelEvaluate, after the model's training, you can use the evaluate to get the test loss and accuracy with the specify data.
    Returns a list contains by loss and accuracy.
    @ param :
    data -- The input data.
    label -- The input data's label.
    batch_size -- THe batch size.
    verbose -- 0, quiet mode. 1, progress bar.
    steps -- An integer if you want to declare the total steps before the evaluation stops. None for default.
    """
        return self.__model.evaluate(data,label,batch_size,verbose,None,steps)

    """
    Fourth part of the Orange library : Visualization, image and video process.
    """
    def CSVDisplay(self,filepath):
        """ CSVDisplay can draw a plot for the CSV file generated by the CSVLogger function.
    @ param:
    filepath -- The CSV file path.
    """
        try:
            import pandas as pd
            from matplotlib import pyplot as plt
        except ImportError:
            raise ImportError('Please install the pandas and matplotlib library first!')
        csv=pd.read_csv(filepath)
        epoch=csv['epoch'].to_numpy()
        acc=csv['acc'].to_numpy()
        loss=csv['loss'].to_numpy()
        val_acc=csv['val_acc'].to_numpy()
        val_loss=csv['val_loss'].to_numpy()
        fig=plt.figure(num='CSV')
        plt.subplot(121)
        plt.plot(epoch,loss)
        plt.plot(epoch,acc)
        plt.legend(('Train loss','Train accuracy'))
        plt.subplot(122)
        plt.plot(epoch,val_loss)
        plt.plot(epoch,val_acc)
        plt.legend(('Validate loss','Validate accuracy'))
        plt.show()

    """
    Fifth part of the Orange library : Common datasets.
    """
    def CIFAR_10(path=None):
        """CIFAR-10 dataset is an image dataset which contains 50000 colorful image with the size of 32*32 on the train set.
    And 10000 colorful image avaliable with the same size of train set on the test set.
    10 classes totally.
    If the download source of CIFAR-10 has blocked, Orange provides a way to load the downloaded dataset for the
    researchers in this region.If you want to load it by your own path, set the path parameter.
    Returns two tuples like (x_train,y_train),(x_test,y_test) which means y_train equals to x_train_labels and the y_test equals to x_test_labels.
    @ param:
    path -- the path for the CIFAR-10 dataset.
    """
        from urllib import error
        try:
            (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
        except error.HTTPError:
            # Copy by the keras source code.
            cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
            datadir_base = os.path.expanduser(cache_dir)
            if not os.access(datadir_base, os.W_OK):
                datadir_base = os.path.join('/tmp', '.keras')
                cache_subdir='datasets'
                datadir = os.path.join(datadir_base, cache_subdir)
            if not os.path.exists(datadir):
                os.makedirs(datadir)
            fname='cifar-10-batches-py.tar.gz'
            fpath = os.path.join(datadir, fname)
            from shutil import copyfile
            copyfile(path,datadir)
            (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
            return (x_train,y_train),(x_test,y_test)
           
    def CIFAR_100(path=None,label_mode='fine'):
        """CIFAR-100 dataset is an image dataset which contains 50000 colorful image with the size of 32*32 on the train set.
    And 10000 colorful image avaliable with the same size of train set on the test set.
    100 classes totally.
    If the download source of CIFAR-100 has blocked, Orange provides a way to load the downloaded dataset for the
    researchers in this region.If you want to load it by your own path, set the path parameter.
    Returns two tuples like (x_train,y_train),(x_test,y_test) which means y_train equals to x_train_labels and the y_test equals to x_test_labels.
    @ param:
    path -- the path for the CIFAR-10 dataset.
    label_model -- 'fine' or 'coarse', means the label's accuracy (maybe?)
    """
        from urllib import error
        try:
            (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar100.load_data(label_mode=label_mode)
        except error.HTTPError:
            # Copy by the keras source code.
            cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
            datadir_base = os.path.expanduser(cache_dir)
            if not os.access(datadir_base, os.W_OK):
                datadir_base = os.path.join('/tmp', '.keras')
                cache_subdir='datasets'
                datadir = os.path.join(datadir_base, cache_subdir)
            if not os.path.exists(datadir):
                os.makedirs(datadir)
            fname='cifar-10-batches-py.tar.gz'
            fpath = os.path.join(datadir, fname)
            from shutil import copyfile
            copyfile(path,datadir)
            (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar100.load_data(label_mode=label_mode)
            return (x_train,y_train),(x_test,y_test)

    def IMDB(path=None):
        """IMDB dataset is a dataset from IMDB comment area. It contains 25000 comments and has preprocessed. The label from this dataset only
        have two classes : negative/postive.
        If the download source of IMDB has blocked, Orange provides a way to load the downloaded dataset for the
        researchers in this region.If you want to load it by your own path, set the path parameter.
        Returns two tuples like (x_train,y_train),(x_test,y_test) which means y_train equals to x_train_labels and the y_test equals to x_test_labels.
        @ param:
        path -- the path for the IMDB dataset.
    """
        from urllib import HTTPError
        try:
            (x_train,y_train),(x_test,y_test)=tf.keras.datasets.imdb.load_data()
        except HTTPError:
            pass

    # Put in OrangeLinear



if __name__ == '__main__':
    print('Usage : import Orange')