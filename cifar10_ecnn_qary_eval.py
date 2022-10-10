from __future__ import print_function
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, LBFGS, DeepFool, BasicIterativeMethod, \
    CarliniWagnerL2, ProjectedGradientDescent, SaliencyMapMethod
from tensorflow.keras.datasets import cifar10
from cleverhans.utils_tf import model_eval
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from utils_ecnn_qary import *
from model_qary import *
import numpy as np
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session

from scipy.linalg import hadamard

config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)


FLAGS.dataset = 'cifar10'
FLAGS.augmentation = True
subtract_pixel_mean = True
FLAGS.num_models = 30
Dense_code = False


#att_para = (['FGSM', [0.0, 0.04]],
#            ['PGD', [0.04]],
#            ['CWL2', [0]])

att_para = (['FGSM', [0.0, 0.04]],
            ['PGD', [0.04]])

#att_para = (['FGSM', [0.0]],)

#att_para = (['CWL2', [0]], )

# net_name, num of binary nets, best epoch, indv_lamda, div_lamda
# in case _rc*: div_lamda for feature_div; otherwise div_lamda for weight_div

t_cnn3 = (['5ary_ens_16x1_hinge_resnet20_s2r1_multiDense_v5', 20, 160, 5.0, 0.5],
          ['3ary_ens_16x1_hinge_resnet20_s2r1_multiDense_v5', 30, 152, 5.0, 0.5, 3],
          ['4ary_ens_16x1_hinge_resnet20_s2r1_multiDense_v5', 30, 184, 5.0, 0.5, 4],
          ['5ary_ens_16x1_hinge_resnet20_s2r1_multiDense_v5', 30, 136, 5.0, 0.5, 5],
          ['3ary_ens_16x1_hinge_resnet20_s2r1_multiDense_v5_set2', 30, 158, 5.0, 0.5, 3],
          ['4ary_ens_16x1_hinge_resnet20_s2r1_multiDense_v5_set2', 30, 151, 5.0, 0.5, 4],
          ['5ary_ens_16x1_hinge_resnet20_s2r1_multiDense_v5_set2', 30, 124, 5.0, 0.5, 5])

t_cnn4 = (['3ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set1', 30, 121, 5.0, 0.5, 3],
          ['3ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set2', 30, 134, 5.0, 0.5, 3],
          ['3ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set3', 30, 115, 5.0, 0.5, 3],
          ['4ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set1', 30, 155, 5.0, 0.5, 4],
          ['4ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set2', 30, 153, 5.0, 0.5, 4],
          ['4ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set3', 30, 120, 5.0, 0.5, 4],
          ['5ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set1', 30, 148, 5.0, 0.5, 5],
          ['5ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set2', 30, 158, 5.0, 0.5, 5],
          ['5ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set3', 30, 165, 5.0, 0.5, 5],
          ['5ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set4', 30, 159, 5.0, 0.5, 5],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set4', 30, 120, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_50_set4', 30, 161, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_5_set4', 30, 139, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_10_set4', 40, 149, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_5_set4', 40, 153, 5.0, 0.5, 2],
          ['dup_2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set4', 30, 154, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set4_noEndShare', 30, 158, 5.0, 0.5, 2],
          ['dup_2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set6', 30, 116, 5.0, 0.5, 2],
          ['dup_2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set5', 30, 158, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set6_noShareEnd', 30, 148, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set5_noShareEnd', 30, 161, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set4', 30, 162, 5.0, 0.5, 2],
          ['3ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set4', 30, 147, 5.0, 0.5, 3],
          ['4ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set4', 30, 156, 5.0, 0.5, 4],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1_BN', 30, 199, 5.0, 0.5, 2],
          ['3ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1_BN', 30, 180, 5.0, 0.5, 3],
          ['4ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1_BN', 30, 106, 5.0, 0.5, 4],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1', 10, 151, 5.0, 0.5, 2],
          ['2ary_ens_16x1_ce_resnet20_s2r1_Dense32_1_set1', 30, 79, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1_noEndShare', 10, 197, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1', 10, 150, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r2_Dense32_hinge_cfd_1_set1_noEndShare', 10, 147, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r2_Dense32_hinge_cfd_1_set1', 10, 160, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r2_noDense32_hinge_cfd_1_set1_noShareEnd', 10, 158, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set1_noShareEnd', 10, 88, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set1', 10, 90, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set2_noShareEnd', 10, 115, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set2', 10, 118, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set3', 10, 98, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set3_noShareEnd', 10, 92, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set4', 10, 96, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set4', 10, 143, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set4_noShareEnd', 10, 122, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set4_noShareEnd', 10, 115, 5.0, 0.5, 2])

t_cnn5 = (['3ary_ens_16x1_hinge_resnet20_s2r1_Dense32_v5_set4', 30, 162, 5.0, 0.5, 3],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set5_noShareEnd', 10, 159, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set4_BN_noShareEnd', 10, 150, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set4_BN', 10, 177, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set4_BN_noShareEnd', 10, 89, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set4_BN', 10, 83, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set5_BN_noShareEnd', 10, 173, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_noDense32_hinge_cfd_1_set5_BN', 10, 180, 5.0, 0.5, 2],)

t_cnn6 = (['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1', 10, 151, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1', 10, 150, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set2', 10, 161, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set2', 10, 141, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1_BN', 10, 163, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1_BN', 10, 114, 5.0, 0.0, 2])

t_cnn7 = (['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set4', 30, 162, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1', 30, 137, 5.0, 0.0, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1_BN', 30, 199, 5.0, 0.5, 2],
          ['2ary_ens_16x1_hinge_resnet20_s2r1_Dense32_hinge_cfd_1_set1_BN', 30, 143, 5.0, 0.0, 2])

depth= 20
stack, res_block = 2, 1
nbits_per_subnet = 10


x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(tf.float32, shape=(None, 10))

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Input image dimensions.
input_shape = x_train.shape[1:]
# Normalize data.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# If subtract pixel mean is enabled
clip_min = 0.0
clip_max = 1.0
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    clip_min -= x_train_mean
    clip_max -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)




if __name__ == "__main__":
    sess = tf.Session()
    tf.keras.backend.set_session(sess)

    t = t_cnn6
    test_opts = [-2]
    for att_idx in range(len(att_para)):
        att_name = att_para[att_idx][0]
        gammas = att_para[att_idx][1]
        for i in range(len(test_opts)):
            model_idx = test_opts[i]

            net_name = t[model_idx][0]
            FLAGS.num_models = t[model_idx][1]
            nEpoch = t[model_idx][2]
            FLAGS.indv_CE_lamda = t[model_idx][3]
            FLAGS.log_det_lamda = t[model_idx][4]
            q = t[model_idx][5]

            dir_pwd = '/home/twp/work/songy/codematrix_v5_qary'
            dir_name = FLAGS.dataset + '_Ensemble_saved_models' + str(FLAGS.num_models) + '_indvCElamda' + str(
                FLAGS.indv_CE_lamda) + '_logdetlamda' + str(FLAGS.log_det_lamda) + '_submean_' + str(subtract_pixel_mean) \
                       + '_dense_' + str(Dense_code) + '_augment_' + str(FLAGS.augmentation) + net_name
            save_dir = os.path.join(dir_pwd, dir_name)
            model_name = 'model.%s.h5' % str(nEpoch).zfill(3)
            filepath = os.path.join(save_dir, model_name)

            cm0 = np.loadtxt('/home/twp/work/songy/all_matrix/'+'/Q/' + str(FLAGS.num_models) + '/RC/' + str(q) + '/1.txt')
            #cm0[:,15:] = 1-cm0[:,:15]
            cm1 = []
            for n in range(FLAGS.num_models):
                cm1.append(tf.keras.utils.to_categorical(cm0[:,n], q))
            cm = np.concatenate(cm1, axis=-1)
            # cm = 2*cm - 1


            for j in range(len(gammas)):
                model_input = Input(shape=input_shape)
                model_dic = {}
                model_out = []
                shared_dense = shared(q)
                #shared_dense = None
                if FLAGS.num_models//nbits_per_subnet == 1:
                    model = subnet_resnet(model_input, nbits_per_subnet, depth, FLAGS.dataset, stack, res_block, shared_dense=shared_dense)
                    model_output = model.output
                else:
                    for i in range(FLAGS.num_models//nbits_per_subnet):
                        model_dic[str(i)] = subnet_resnet(model_input, nbits_per_subnet, depth, FLAGS.dataset, stack, res_block, shared_dense=shared_dense)
                        model_out.append(model_dic[str(i)].output)
                    model_output = tf.keras.layers.concatenate(model_out)
                    model = Model(model_input, model_output)
                model.load_weights(filepath)
                model_output = decoder(model_output,
                                       opt='linear',
                                       drop_prob=0.0,
                                       cm=cm)
                model_output = Activation('softmax', name='last_softmax')(model_output)
                print(model_output)
                modelfull = Model(model_input, model_output)
                # modelfull.summary()

                wrap = KerasModelWrapper(modelfull)
                if att_name == 'FGSM':
                    attack = FastGradientMethod(wrap, sess=sess)
                    attacker_params = {'eps': gammas[j],
                                       'clip_min': clip_min,
                                       'clip_max': clip_max}
                if att_name == 'BIM':
                    attack = BasicIterativeMethod(wrap, sess=sess)
                    attacker_params = {'eps': gammas[j],
                                       'eps_iter': 0.02,
                                       'nb_iter': 200,
                                       'clip_min': clip_min,
                                       'clip_max': clip_max}
                if att_name == 'PGD':
                    attack = ProjectedGradientDescent(wrap, sess=sess)
                    attacker_params = {'eps': gammas[j],
                                       'eps_iter': 0.02,
                                       'nb_iter': 200,
                                       'clip_min': clip_min,
                                       'clip_max': clip_max}
                if att_name == 'JSMA':
                    attack = SaliencyMapMethod(wrap, sess=sess)
                    attacker_params = {'gamma': gammas[j],
                                       'clip_min': clip_min,
                                       'clip_max': clip_max}
                if att_name == 'CWL2':
                    attack = CarliniWagnerL2(wrap, sess=sess)
                    attacker_params = {'batch_size': 100,
                                       'confidence': gammas[j],
                                       'max_iterations': 2000,
                                       'initial_const': 1e-1,
                                       'learning_rate': 1e-3,
                                       'binary_search_steps': 10,
                                       'clip_min': clip_min,
                                       'clip_max': clip_max
                                       }
                x_adv = attack.generate(x, **attacker_params)
                x_adv = tf.stop_gradient(x_adv)

                eval_par = {'batch_size': 100}
                preds = modelfull(x_adv)
                acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_par)
                print('att_name: %s, att_para= %.2f, net: %s, adv_acc: %.3f, using %d models, detLamd=%.2f, indvLamd=%.2f' %
                      (att_name, gammas[j], net_name, acc, FLAGS.num_models, FLAGS.log_det_lamda, FLAGS.indv_CE_lamda))      
                
                #logits = model(x)
                #z_n = sess.run(logits, feed_dict={x: x_test[:2000]})
                #np.savetxt("z_n.txt", z_n, fmt="%s")
                



