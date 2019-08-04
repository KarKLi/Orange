"""
Orange -- A library based on tensorflow.keras, using for simpilfying computer vision code based on deep learning.
In Orange, you can only focus on your model's structure and don't need to focus on the code's structure.
Orange gives a library that much close to model's structure.
Orange can construct two types of network. First is linear stacked neural network such as VGGNet, AlexNet, etc.
Second is non-linear stacked neural network (may have shortcut connection like ResNet or inception-block like GoogleNet).
Orange is a abstract library based on tensorflow.keras, so it doesn't provide the basic operation of math such as gradient computation, etc.
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

class __Orange(object):
    def __init__(self,model_name,input_shape):
        """__init__ accept two parameters, model_name and input_shape,
    @ param:
    model_name : Model's name, such as ResNet-18, ResNet-34. This model name is a symbol to recognize the model.
    input_shape : Regulate the input vector's shape.
    model : The tf.keras.Sequential() model, which is a stack of layers.
    """
        version=list(tf.__version__.split('.'))
        if version[0]!='1' or int(version[1]) < 7:
            raise VersionError('TensorFlow imported, but the version need to update. Please update your TensorFlow to 1.7.0 or higher.')
        if tf.test.is_built_with_cuda() is not True:
            raise VersionError('Orange only support tensorflow-gpu library, please check your library.')
        if tf.test.is_gpu_available() is not True:
            raise VersionError('Orange need to run on the machine with NVIDIA GPU, please check your device.')
        self.__model_name=model_name
        self.__shape=input_shape
        self.__model=tf.keras.Sequential()
        print('Orange initialization complete.')
        # self.__modelhasbuilt=False
        # Put it in OrangeLinear


    def Getmodelname(self):
        """Getmodelname returns the name of model.
    """
        return self.__model_name


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
        return tf.keras.optimizers.RMSProp(lr,rho,epsilon,decay)

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
        return tf.keras.optimizers.Adam(lr,beta_1,beta_2,epslion,decay.amsgrad)

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

    def __crossentropy(Sigmoid=False,Softmax=False,Sparse=False,y_labels=None):
        """Cross entropy loss function is a loss function used in model complication.
    @ param:
    Sigmoid -- Will use binary_crossentropy if the activate function is Sigmoid
    Softmax -- Will use categorical_crossentropy if the activate function is Softmax
    Sparase -- Will use sparse_categorical_crossentropy if the dataset has many zero data.

    You don't need to do preprocess of the label. At categorical_crossentropy and sparse_categorical_crossentropy, the crossentropy function
    will do the preprocessing. But you need to pass the y_labels to process.
    """
        if Sigmoid is True:
            return tf.keras.losses.binary_crossentropy
        elif Softmax is True and y_labels is not None:
            from keras.utils.np_utils import to_categorical
            y_labels=to_categorical(y_labels,num_classes=None)
            return tf.keras.losses.categorical_crossentropy
        else:
            import numpy as np
            y_labels=np.expand_dims(y_labels,-1)
            return tf.keras.losses.sparse_categorical_crossentropy

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
        if isinstance(layer,tf.keras.layers) is not True:
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
            self.__model.add(tf.keras.layers.Flatten(input_shape=__shape))
    
    def relu(self,neural_numbers):
        """relu accept 1 parameter, the relu layer will activate the neural unit.
    @ formula:
    f_{relu}(x) = max(0,x)
    @ param:
    neural_numbers : The number of neural units you want to activate.
    """
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
        self.__model.add(tf.keras.layers.Dense(neural_numbers,activation=tf.nn.leaky_relu))

    def tanh(self,neural_numbers):
        """tanh accept 1 parameter, the tanh layer will activate the neural unit.
    @ formula:
    f_{tanh}(x) = tanh(x) = (exp(x)-exp(-x)) / (exp(x)+exp(-x))
    @ param:
    neural_numbers : The number of neural units you want to activate.
    """
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
    
    def Convolution(self,output_depth,kernel_size,x_stride,y_stride=None,padding='valid'):
        """Convolution is one of the most important operations in neural network. After convolutional operation the image can
    get the new matrix set which called 'feature map'.
    @ param :
    output_depth : The depth of output feature map
    kernel_size : The convolutional kernel size. Generally, it's a tuple with two integers (x,y). If you want to
    have the same size on x-axis and y-axis, just pass one integer.
    stride : The stride of kernel move. Passing one integer in general. If you want to have different stride on different
    direction, pass a tuple with two integers (x,y).
    padding : Padding zero before convolution operation or not. The default value is 'valid', which means not padding zero.
    Pass 'same' if you want to padding zero.
    """
        if y_stride is not None:
            self.__model.add(tf.keras.layers.Conv2D(filters=output_depth,kernel_size=kernel_size,
                                           stride= (x_stride,y_stride),
                                           padding=padding
                                           ))
        else:
            self.__model.add(tf.keras.layers.Conv2D(filters=output_depth,kernel_size=kernel_size,
                                           strides=x_stride,
                                           padding=padding
                                           ))


    def pooling(self,pool_size,strides,padding='valid',pool_type='maxpooling'):
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

    def __Callback(param=[]):
        """Callback function, a function which used in model execution.
        The param is a list which can have this values:
    @ param:
    param -- a list contain the callback functions.
    'NaN', TerminateOnNaN, if the Loss is too big, stop the training process.
    'SaveCheckpoint', ModelCheckpoint, save the model after a epoch. If you pass the SaveCheckPoint, the next element of the param must be a dict which set 
    the parameters of ModelCheckpoint.
    'EarlyStopping', EarlyStopping, stop the training process when the monitored index doesn't increase/decrease.
    'LearningRateChange', 
    """
    
    """
    Fourth part of the Orange library : Visualization, image and video process.
    """
    
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
        from urllib import HTTPError
        try:
            (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
        except HTTPError:
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
        from urllib import HTTPError
        try:
            (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar100.load_data(label_mode=label_mode)
        except HTTPError:
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
    #def Displaymodel(self):
    #    if self.__modelhasbuilt is True:
    #        self.__model.summary()
    #    else:
    #        print('Please compile your model first.')
    #"""
    #Print the model'summary by using the bulit-in function tf.keras.Sequential.summary()
    #"""
    # Put in OrangeLinear

if __name__ == '__main__':
    print('Usage : import Orange')
    print('Or from Orange import OrangeLinear')
    print('Or from Orange import OrangeNonLinear')