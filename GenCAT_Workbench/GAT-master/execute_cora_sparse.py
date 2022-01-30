import time
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse

from models import GAT
from models import SpGAT
from utils import process

import time

# checkpt_file = 'pre_trained/citeseer/mod_citeseer.ckpt'
# checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

# dataset = 'citeseer'
# dataset = 'cora'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('iter_count', '0', 'The number of iterator')
flags.DEFINE_string('root_path', '/content/drive/My Drive/Colab Notebooks/GenCAT/', 'Root path of the project')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('lr', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10000, 'Number of epochs to train.')
flags.DEFINE_integer('units', 8, 'Number of hidden units.')
# flags.DEFINE_integer('hidden1', 8, 'Number of units in hidden layer 1.')
# flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('l2_coef', 5e-5, 'Weight for L2 loss on embedding matrix.')
# flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
# flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('patience', 100, 'patience')
flags.DEFINE_bool('print_per_epoch', False, 'Whether showing the loss and accuracy for each epoch')
flags.DEFINE_bool('_use_full_gpu', True, 'Do you use GPU?')

# training params
batch_size = 1
# nb_epochs = 100000
# patience = 100
# lr = 0.005  # learning rate
# l2_coef = 0.0005  # weight decay
hid_units = [FLAGS.units] # numbers of hidden units per each attention head in each layer
# n_heads = [16, 1] # additional entry for the output layer
n_heads = [8] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
# model = GAT
model = SpGAT

# print('Dataset: ' + dataset)
# print('----- Opt. hyperparams -----')
# print('lr: ' + str(lr))
# print('l2_coef: ' + str(l2_coef))
# print('----- Archi. hyperparams -----')
# print('nb. layers: ' + str(len(hid_units)))
# print('nb. units per layer: ' + str(hid_units))
# print('nb. attention heads: ' + str(n_heads))
# print('residual: ' + str(residual))
# print('nonlinearity: ' + str(nonlinearity))
# print('model: ' + str(model))

sparse = True

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(FLAGS.dataset,FLAGS.root_path)
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

if sparse:
    biases = process.preprocess_adj_bias(adj)
else:
    adj = adj.todense()
    adj = adj[np.newaxis]
    biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

if FLAGS._use_full_gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

start = time.time()
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        if sparse:
            #bias_idx = tf.placeholder(tf.int64)
            #bias_val = tf.placeholder(tf.float32)
            #bias_shape = tf.placeholder(tf.int64)
            bias_in = tf.sparse_placeholder(dtype=tf.float32)
        else:
            bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, FLAGS.lr, FLAGS.l2_coef)

    # saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        epoch_time = []

        for epoch in range(FLAGS.epochs):

            t = time.time()
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                if sparse:
                    bbias = biases
                else:
                    bbias = biases[tr_step*batch_size:(tr_step+1)*batch_size]

                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: bbias,
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                if sparse:
                    bbias = biases
                else:
                    bbias = biases[vl_step*batch_size:(vl_step+1)*batch_size]
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: bbias,
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            if FLAGS.print_per_epoch is True:
                print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                        (train_loss_avg/tr_step, train_acc_avg/tr_step,
                        val_loss_avg/vl_step, val_acc_avg/vl_step))

            epoch_time.append(time.time() - t)

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    # saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == FLAGS.patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        # saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            if sparse:
                bbias = biases
            else:
                bbias = biases[ts_step*batch_size:(ts_step+1)*batch_size]
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: bbias,
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()
        
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

with open(FLAGS.root_path+"experimental_results/GATsp_"+FLAGS.dataset+"_iter"+str(FLAGS.iter_count), 'w') as f:
    f.write(str(ts_acc/ts_step) + "\n" + str(elapsed_time) + "[sec]\n"+ str(np.average(epoch_time)) + "\n" + str(len(epoch_time))+"\n"+str(train_acc_avg/tr_step))
