from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Activation, GaussianNoise, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Maximum, Dot, LeakyReLU, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
#from ClassBlender import ClassBlender
#from DataAugmenter import DataAugmenter
#from Clipper import Clipper
#from Grayscaler import Grayscaler


class Linear(layers.Layer):
    def __init__(self, cm):
        super(Linear, self).__init__()
        self.w = tf.transpose(tf.cast(cm, dtype='float32'))
        #self.w = tf.subtract(tf.multiply(self.w, tf.constant([2.])), tf.constant([1.]))

    def call(self, inputs):
        # inputs_new = tf.subtract(tf.multiply(inputs, tf.constant([2.])), tf.constant([1.]))
        mat1 = tf.matmul(tf.nn.sigmoid(inputs), self.w)
        # mat1 = tf.matmul(inputs, self.w)
        l1 = tf.maximum(mat1, 0)
        # l2 = l1 / K.sum(l1, axis=1, keepdims=True)
        return l1

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'w': self.w
        })
        return config


def decoder(inputs,
            opt='dense',
            drop_prob=0,
            cm=None):
    x = Dropout(drop_prob)(inputs)
    if opt == 'dense':
        outputs = Dense(10, activation='softmax')(x)
    if opt == 'dense_L1':
        outputs = Dense(10, activation='softmax', kernel_regularizer=regularizers.l1(0.01))(x)
    if opt == 'linear':
        linear_layer = Linear(cm)
        outputs = linear_layer(x)
    return outputs


def shared(q):
    shared = Dense(q)
    return shared


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    #conv = Conv2D(
    #    num_filters,
    #    kernel_size=kernel_size,
    #    strides=strides,
    #    padding='same',
    #    kernel_initializer='he_normal',
    #    kernel_regularizer=l2(1e-4))

    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input, depth, num_classes=10, dataset='cifar10', shared_dense=None):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = input
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            #if stack==2 and res_block==num_res_blocks-1:
            #    x = LeakyReLU()(x)
            #else:
            #    x = Activation('relu')(x)
            x = Activation('relu', name='Act'+'_'+str(stack)+'_'+str(res_block))(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    if dataset=='mnist':
        poolsize = 7
    else:
        poolsize = 8
    x = AveragePooling2D(pool_size=poolsize)(x)
    x = Flatten()(x)
    if shared_dense==None:
        outputs = Dense(1)(x)
    else:
        outputs = shared_dense(x)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v1_back(input, stack0, res_block0, depth, num_classes=10, dataset='cifar10'):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = input
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    #if res_block0 == num_res_blocks-1:
    #    stack_init = stack0+1
    #    res_block_init = 0
    #else:
    #    stack_init = stack0
    #    res_block_init = res_block0

    # old setting:
    stack_init = stack0+1
    res_block_init = 0

    for stack in range(stack_init, 3):
        for res_block in range(res_block_init, num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            #if stack==2 and res_block==num_res_blocks-1:
            #    x = LeakyReLU()(x)
            #else:
            #    x = Activation('relu')(x)
            x = Activation('relu', name='Act'+'_'+str(stack)+'_'+str(res_block))(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    if dataset=='mnist':
        poolsize = 7
    else:
        poolsize = 8
    x = AveragePooling2D(pool_size=poolsize)(x)
    x = Flatten()(x)
    model = Model(inputs=inputs, outputs=x)
    return model


def resnet_v1_ind(input, depth, num_classes=10, dataset='cifar10', shared_dense=None):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = input
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            #if stack==2 and res_block==num_res_blocks-1:
            #    x = LeakyReLU()(x)
            #else:
            #    x = Activation('relu')(x)
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    if dataset=='mnist':
        poolsize = 7
    else:
        poolsize = 8
    x = AveragePooling2D(pool_size=poolsize)(x)
    x = Flatten()(x)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=x)
    return model
    
    
    

#------ResNet----------

def conv_front_resnet(x, depth, dataset, layer_name):
    model = resnet_v1(input=x, depth=depth, num_classes=10, dataset=dataset)
    y = model.get_layer(layer_name).output
    return y

def conv_branch_resnet(x, layer_name, stack, res_block, depth, dataset, model):
    index = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            index = idx
            break
    DL_input = Input(model.layers[index+1].input_shape[1:])
    model_back = resnet_v1_back(DL_input, stack, res_block, depth, num_classes=10, dataset=dataset)
    y = model_back(x)
    return y
    

def subnet_resnet(inputs, n, depth, dataset, stack, res_block, shared_dense=None):
    layer_name = 'Act'+'_'+str(stack)+'_'+str(res_block)
    outputs1 = conv_front_resnet(inputs, depth, dataset, layer_name)
    model_out = []
    for i in range(n):
        if dataset == 'cifar10' or dataset == 'cifar100':
           inputshape = (32, 32, 3)
        if dataset == 'mnist':
           inputshape = (28, 28, 1)
        model = resnet_v1(input=Input(shape=inputshape), depth=depth, num_classes=10, dataset=dataset)
        outputs = conv_branch_resnet(outputs1, layer_name, stack, res_block, depth, dataset, model)
        # outputs = Dense(128, activation='relu')(outputs)
        outputs = Dense(32, activation='relu')(outputs)
        if shared_dense == None:
            outputs = Dense(2)(outputs)
        else:
            outputs = shared_dense(outputs)
        model_out.append(outputs)
    model_output = tf.keras.layers.concatenate(model_out)
    model = Model(inputs, model_output)
    return model


def subnet_resnet_noFrontShare(inputs, n, depth, dataset, stack, res_block, shared_dense=None):
    model_out = []
    for i in range(n):
        if dataset == 'cifar10' or dataset == 'cifar100':
           inputshape = (32, 32, 3)
        if dataset == 'mnist':
           inputshape = (28, 28, 1)
        model = resnet_v1_ind(input=inputs, depth=depth, num_classes=10, dataset=dataset)
        outputs = model.output
        # outputs = Dense(256, activation='relu')(outputs)
        outputs = Dense(32, activation='relu')(outputs)
        if shared_dense == None:
            outputs = Dense(2, activation='linear')(outputs)
        else:
            outputs = shared_dense(outputs)
        model_out.append(outputs)
    model_output = tf.keras.layers.concatenate(model_out)
    model = Model(inputs, model_output)
    return model