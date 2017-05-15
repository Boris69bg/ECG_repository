import numpy as np
import theano
import theano.tensor as T
import lasagne
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
#%matplotlib inline

import gzip
import pickle
import cPickle

from datetime import datetime
import time
import glob
import os

import cv2
from tqdm import tqdm
from tqdm import tqdm_notebook

def gen_minibatches(X, y, batch_size, shuffle=False, seed = 270893):
    assert len(X) == len(y), "Training data sizes don't match"
    if shuffle:
        np.random.seed(seed)
        ids = np.random.permutation(len(X))
    else:
        ids = np.arange(len(X))
    for start_idx in range(0, len(X) - batch_size + 1, batch_size):
        ii = ids[start_idx:start_idx + batch_size]
        yield X[ii].astype('float32'), y[ii].astype('int32')
    if len (X) % batch_size != 0:
        ii = ids[-(len( X ) % batch_size):]
        yield X[ii].astype('float32'), y[ii].astype('int32')

def gen_balanced_minibatches(X, y, batch_size):
    assert len(X) == len(y), "Training data sizes don't match"

    false_num = np.sum(y==0)
    true_num = np.sum(y==1)

    translation = {0 : 1.0 / (2 * false_num), 1 : 1.0 / (2 * true_num)}
    ids = np.arange(len(X))
    ids_p = map(lambda x : translation[x], y)

    for i in range(len(y) / batch_size):
        total_batch = np.random.choice(ids, batch_size, p = ids_p)

#        print X[total_batch].astype("float32")

#        false_batch = np.random.choice(ids[y==0], int(batch_size * 0.5))
#        true_batch = np.random.choice(ids[y==1], batch_size - int(batch_size * 0.5))

#        total_batch = np.concatenate([false_batch, true_batch])

        yield X[total_batch].astype('float32'), y[total_batch].astype('int32')

def train_for_one_epoch(train_fn, X_train, y_train, batch_size):
    train_err = train_batches = 0
    epoch_loss_history = []
    for X_batch, y_batch in gen_balanced_minibatches(X_train, y_train, batch_size):
#    for X_batch, y_batch in tqdm(gen_minibatches(X_train, y_train, batch_size, shuffle = True), leave = True):
        err = train_fn(X_batch, y_batch)
        train_err += err
        train_batches += 1
        epoch_loss_history.append(err)
#        tqdm.write("".join(["Err: ", str(err)]))
    return train_err, train_batches, epoch_loss_history

def validate_model(val_fn, X_val, y_val, batch_size):
    val_err = val_batches = 0
    total_preds = []
    for X_batch, y_batch in gen_minibatches(X_val, y_val, batch_size, shuffle=False):
        err, preds = val_fn(X_batch, y_batch)
        total_preds += list(preds)
        val_err += err
        val_batches += 1
    rocauc = roc_auc_score(y_val, total_preds)
    return val_err, rocauc, val_batches, np.array(total_preds)

def predict(network, input_var, x):
    prediction = lasagne.layers.get_output(network)
    test_prediction = lasagne.layers.get_output(network)
    predict_fn = theano.function([input_var], test_prediction)

    pred = predict_fn(x.astype('float32'))

    return pred

def train(network, input_var,
          X_train, y_train, X_val, y_val,
          learning_rate, grad_clipping=1.5, learning_rate_decay=1.0,
          momentum=1.0, momentum_decay=1.0,
          decay_after_epochs=1,
          regu=0.0,
          batch_size=100, num_epochs=10):

 #   print("Compiling...")
    target_var = T.ivector('target')
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()
    
 #   train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
    learning_rate_var = theano.shared(np.float32(learning_rate))
    params = lasagne.layers.get_all_params(network, trainable=True)
    all_grads = T.grad(loss, params)
    scaled_grads = lasagne.updates.total_norm_constraint(all_grads, grad_clipping)
    updates = lasagne.updates.adam(
        scaled_grads, params, learning_rate=learning_rate_var)
    test_prediction = lasagne.layers.get_output(network)
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
    test_loss = test_loss.sum()
#    test_answers = T.argmax(test_prediction, axis=1)

#    test_prec = T.sum(T.and_(test_answers, target_var)) / T.sum(test_answers)
#    test_rec = T.sum(T.and_(test_answers, target_var)) / T.sum(target_var)
 #   test_acc = T.mean(T.eq(test_answers, target_var), dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
#    train_acc_fn = theano.function([input_var, target_var], train_acc)
    val_fn = theano.function([input_var, target_var], [test_loss, test_prediction])

    predict_fn = theano.function([input_var], test_prediction)

#    print("Training...")

    best_val_rocauc = 0.0
    best_val_preds = None
    loss_history = []
    train_rocauc_history = []
    val_rocauc_history = []

    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()

        # train model for one pass
        train_err, train_batches, epoch_loss_history = train_for_one_epoch(
            train_fn, X_train, y_train, batch_size)
        loss_history.extend(epoch_loss_history)

        # training accuracy
        n_acc = len(y_val)
        trval_err, trval_rocauc, trval_batches, tr_preds = validate_model(
            val_fn, X_train[:n_acc], y_train[:n_acc], batch_size)
        train_rocauc_history.append(trval_rocauc)

        # validation accuracy
        val_err, val_rocauc, val_batches, val_preds = validate_model(val_fn, X_val, y_val, batch_size)
        val_rocauc_history.append(val_rocauc)

        # keep track of the best model based on validation accuracy

        if val_rocauc > best_val_rocauc:
          # make a copy of the model
          best_val_rocauc = val_rocauc
          best_model = lasagne.layers.get_all_param_values(network)
          best_val_preds = val_preds
        print('epoch %d / %d in %.1fs: loss %f, trval_rocauc %3f, val_rocauc %3f, lr %4f'
              % (epoch + 1, num_epochs, time.time() - start_time,
                 train_err / train_batches, trval_rocauc, val_rocauc,
                 learning_rate_var.get_value()))

    return best_model, loss_history, train_rocauc_history, val_rocauc_history, best_val_preds


