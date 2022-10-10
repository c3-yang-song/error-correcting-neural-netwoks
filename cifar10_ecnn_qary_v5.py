from __future__ import print_function
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LambdaCallback
from utils_ecnn_qary import *
from model_qary import *
import numpy as np
import os

from scipy.linalg import hadamard

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session

config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)

# Subtracting pixel mean improves accuracy
FLAGS.indv_CE_lamda = 5.0 # loss weight 
FLAGS.augmentation = True
FLAGS.dataset = 'cifar10'
FLAGS.epochs = 200
FLAGS.num_models = 30 # number of binary classifiers
FLAGS.log_det_lamda = 0.5 # regularizer weight
subtract_pixel_mean = True
Dense_code = False
nbits_per_subnet = FLAGS.num_models
loss_type = 'hinge' # ce or hinge
hinge_cfd_level = 1 # loss = max(wrong_logit - correct_logit + hinge_cfd_level, 0)
depth= 20
q = 2
stack, res_block = 2, 1
net_name = str(q)+'ary_ens_16x1_'+str(loss_type)+'_resnet'+str(depth)+'_s'+str(stack)+'r'+str(res_block)+'_Dense32_'+'hinge_cfd_'+str(hinge_cfd_level)+'_set1_nBN'


dir_pwd = '/home/twp/work/songy/codematrix_v5_qary'  #  os.getcwd()
dir_name = FLAGS.dataset + '_Ensemble_saved_models' + str(FLAGS.num_models) + '_indvCElamda' + str(
            FLAGS.indv_CE_lamda) + '_logdetlamda' + str(FLAGS.log_det_lamda) + '_submean_' + str(subtract_pixel_mean) \
           + '_dense_' + str(Dense_code) + '_augment_'+str(FLAGS.augmentation) + net_name
save_dir = os.path.join(dir_pwd, dir_name)
model_name = 'model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


#savepath = os.path.join(save_dir, 'model.051.h5')

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [checkpoint, lr_scheduler]


cm = np.loadtxt('/home/twp/work/songy/all_matrix/100/2/1.txt') # cm is a FLAGS.num_models x num_classes binary codematrix containing {0,1} and you can generate your own cm here

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Input image dimensions.
input_shape = x_train.shape[1:]
# Normalize data.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
# Convert class vectors to binary class matrices.
y_train_1hot = tf.keras.utils.to_categorical(y_train, 10)
y_test_1hot = tf.keras.utils.to_categorical(y_test, 10)

# Augment labels
y_train_code = np.zeros((y_train.shape[0], FLAGS.num_models))
y_test_code = np.zeros((y_test.shape[0], FLAGS.num_models))
for i in range(10):
    idx_train = list(np.where(y_train == i)[0])
    idx_test = list(np.where(y_test == i)[0])
    y_train_code[idx_train, :] = cm[i, :]
    y_test_code[idx_test, :] = cm[i, :]


if __name__ == "__main__":
    with tf.Session() as sess:

        model_input = Input(shape=input_shape)
        model_dic = {}
        model_out = []
        shared_dense = shared(q)
        if FLAGS.num_models//nbits_per_subnet == 1:
            model = subnet_resnet(model_input, nbits_per_subnet, depth, FLAGS.dataset, stack, res_block, shared_dense=shared_dense)
        else:
            for i in range(FLAGS.num_models//nbits_per_subnet):
                model_dic[str(i)] = subnet_resnet(model_input, nbits_per_subnet, depth, FLAGS.dataset, stack, res_block, shared_dense=shared_dense)
                model_out.append(model_dic[str(i)].output)
            model_output = tf.keras.layers.concatenate(model_out)
            model = Model(model_input, model_output)
        tf.keras.utils.plot_model(model, to_file='network_image.png')
        model.summary()
        # model.load_weights(savepath)


        acc_metric = ce_metric(FLAGS.num_models, q, loss_type)
        div_metric = ens_div_metric(FLAGS.num_models, q)  
        total_loss = custom_loss(FLAGS.num_models, q, hinge_cfd_level, loss_type)
        model.compile(
            loss=total_loss,
            optimizer=Adam(lr=lr_schedule(0)),
            metrics=[acc_metric, div_metric])

        # Run training, with or without data augmentation.
        if not FLAGS.augmentation:
            print('Not using data augmentation.')
            model.fit(x_train, y_train_code,
                      batch_size=FLAGS.batch_size,
                      epochs=FLAGS.epochs,
                      verbose=1,
                      shuffle=True,
                      callbacks=callbacks,
                      validation_data=(x_test, y_test_code))
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(
                datagen.flow(x_train, y_train_code, batch_size=FLAGS.batch_size),
                validation_data=(x_test, y_test_code),
                epochs=FLAGS.epochs,
                verbose=1,
                callbacks=callbacks)



