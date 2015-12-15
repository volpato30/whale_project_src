#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.layers import GlobalPoolLayer

BATCHSIZE=40

def load_data(version='resize'):
    length=4288
    arr=np.arange(length)
    np.random.shuffle(arr)
    train_i=arr[:int(np.floor(length*0.9))]
    valid_i=arr[int(np.floor(length*0.9)):]

    if version=='resize':
        a=np.load("/scratch/rqiao/full_data/resize_dataOmit5.npz")
        data=a['arr_0']
        target=a['arr_1']
        return data[train_i,:,:,:],target[train_i],data[valid_i,:,:,:],target[valid_i]
    if version=='flipx':
        a=np.load("/scratch/rqiao/full_data/FlipX_dataOmit5.npz")
        data=a['arr_0']
        target=a['arr_1']
        return data[train_i,:,:,:],target[train_i],data[valid_i,:,:,:],target[valid_i]
    if version=='r180':
        a=np.load("/scratch/rqiao/full_data/Rotate180.npz")
        data=a['arr_0']
        target=a['arr_1']
        return data[train_i,:,:,:],target[train_i],data[valid_i,:,:,:],target[valid_i]
    if version=='r90':
        a=np.load("/scratch/rqiao/full_data/Rotate90_dataOmit5.npz")
        data=a['arr_0']
        target=a['arr_1']
        return data[train_i,:,:,:],target[train_i],data[valid_i,:,:,:],target[valid_i]
    if version=='r270':
        a=np.load("/scratch/rqiao/full_data/Rotate270.npz")
        data=a['arr_0']
        target=a['arr_1']
        return data[train_i,:,:,:],target[train_i],data[valid_i,:,:,:],target[valid_i]

Conv2DLayer = lasagne.layers.Conv2DLayer

def inception_module(l_in,pool_filters, num_1x1, reduce_3x3, num_3x3, reduce_5x5, num_5x5, gain=1.0, bias=0):
    """
    inception module (without the 3x3s1 pooling and projection because that's difficult in Theano right now)
    """
    out_layers = []

    if pool_filters > 0:
        l_pool = lasagne.layers.MaxPool2DLayer(l_in, pool_size=3, stride=1, pad=1)
        l_pool_reduced = lasagne.layers.NINLayer(l_pool, num_units=pool_filters, W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_pool_reduced)

    # 1x1
    if num_1x1 > 0:
        l_1x1 = lasagne.layers.NINLayer(l_in, num_units=num_1x1, W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_1x1)
    
    # 3x3
    if num_3x3 > 0:
        if reduce_3x3 > 0:
            l_reduce_3x3 = lasagne.layers.NINLayer(l_in, num_units=reduce_3x3, W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        else:
            l_reduce_3x3 = l_in
        l_3x3 = Conv2DLayer(l_reduce_3x3, num_filters=num_3x3, filter_size=(3, 3), pad="same", W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_3x3)
    
    # 5x5
    if num_5x5 > 0:
        if reduce_5x5 > 0:
            l_reduce_5x5 = lasagne.layers.NINLayer(l_in, num_units=reduce_5x5, W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        else:
            l_reduce_5x5 = l_in
        l_5x5 = Conv2DLayer(l_reduce_5x5, num_filters=num_5x5, filter_size=(5, 5), pad="same", W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_5x5)
    
    # stack
    l_out = lasagne.layers.concat(out_layers)
    return l_out


def build_cnn(input_var=None):
    
    network = lasagne.layers.InputLayer(shape=(BATCHSIZE, 3, 256, 256),
                                        input_var=input_var)
    
    network = Conv2DLayer(network, num_filters=64, filter_size=7, stride=2, pad=3)

    network = lasagne.layers.MaxPool2DLayer(network,pool_size=3, stride=2, ignore_border=False)

    network= LRNLayer(network, alpha=0.00002, k=1)

    network = lasagne.layers.NINLayer(network, num_units=64, W=lasagne.init.Orthogonal(1), b=lasagne.init.Constant(0))
    network = Conv2DLayer(network, 192, 3, pad=1)
    network = LRNLayer(network, alpha=0.00002, k=1)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)

    
    network = inception_module(
            network,pool_filters=32, num_1x1=64, reduce_3x3=96, num_3x3=128, reduce_5x5=16, num_5x5=32)
    
    network = inception_module(
            network,pool_filters=64, num_1x1=128, reduce_3x3=128, num_3x3=192, reduce_5x5=32, num_5x5=96)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)


    network = inception_module(
            network,pool_filters=64, num_1x1=192, reduce_3x3=96, num_3x3=208, reduce_5x5=16, num_5x5=48)

    network = inception_module(
            network,pool_filters=64, num_1x1=160, reduce_3x3=112, num_3x3=224, reduce_5x5=24, num_5x5=64)

    network = inception_module(
            network,pool_filters=64, num_1x1=128, reduce_3x3=128, num_3x3=256, reduce_5x5=24, num_5x5=64)

    network = inception_module(
            network,pool_filters=64, num_1x1=112, reduce_3x3=144, num_3x3=288, reduce_5x5=32, num_5x5=64)

    network = inception_module(
            network,pool_filters=128, num_1x1=256, reduce_3x3=160, num_3x3=320, reduce_5x5=32, num_5x5=128)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)

    network = inception_module(
            network,pool_filters=128, num_1x1=256, reduce_3x3=160, num_3x3=320, reduce_5x5=32, num_5x5=128)

    network = inception_module(
            network,pool_filters=128, num_1x1=384, reduce_3x3=192, num_3x3=384, reduce_5x5=48, num_5x5=128)

    network = GlobalPoolLayer(network)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.4),
            num_units=344,
            nonlinearity=lasagne.nonlinearities.linear)

    network = lasagne.layers.NonlinearityLayer(network,nonlinearity=lasagne.nonlinearities.softmax)

    return network



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(num_epochs=10):
    # Load the dataset
    print("Loading data...")
    datasets = load_data('resize')
    X_train, y_train = datasets[0], datasets[1]
    X_val, y_val = datasets[2], datasets[3]
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    learnrate=0.005
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    network = build_cnn(input_var)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    l2_penalty = regularize_layer_params(network, l2)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()+0.01*l2_penalty
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learnrate, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    best_val_loss=10
    improvement_threshold=0.999
    best_acc=0
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        if epoch % 8 == 7:
            learnrate*=0.96
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learnrate, momentum=0.9)
        for batch in iterate_minibatches(X_train, y_train,BATCHSIZE, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, BATCHSIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        if val_err/val_batches < best_val_loss*improvement_threshold:
            np.savez('best_model_omit5_v4.npz', *lasagne.layers.get_all_param_values(network))
            best_val_loss=val_err/val_batches
            print("                    best validation loss\t\t{:.6f}".format(best_val_loss))

        if val_acc / val_batches>best_acc:
            best_acc=val_acc / val_batches
            np.savez('best_classification_model_omit5_v4.npz', *lasagne.layers.get_all_param_values(network))
            print('                    saved best classification  model')

    datasets = load_data('flipx')
    X_train, y_train = datasets[0], datasets[1]
    X_val, y_val = datasets[2], datasets[3]
    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        if epoch % 8 == 7:
            learnrate*=0.96
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learnrate, momentum=0.9)
        for batch in iterate_minibatches(X_train, y_train,BATCHSIZE, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, BATCHSIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        if val_err/val_batches < best_val_loss*improvement_threshold:
            np.savez('best_model_omit5_v4.npz', *lasagne.layers.get_all_param_values(network))
            best_val_loss=val_err/val_batches
            print("                    best validation loss\t\t{:.6f}".format(best_val_loss))

        if val_acc / val_batches>best_acc:
            best_acc=val_acc / val_batches
            np.savez('best_classification_model_omit5_v4.npz', *lasagne.layers.get_all_param_values(network))
            print('                    saved best classification  model')

    datasets = load_data('r90')
    X_train, y_train = datasets[0], datasets[1]
    X_val, y_val = datasets[2], datasets[3]
    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        if epoch % 8 == 7:
            learnrate*=0.96
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learnrate, momentum=0.9)
        for batch in iterate_minibatches(X_train, y_train,BATCHSIZE, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, BATCHSIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        if val_err/val_batches < best_val_loss*improvement_threshold:
            np.savez('best_model_omit5_v4.npz', *lasagne.layers.get_all_param_values(network))
            best_val_loss=val_err/val_batches
            print("                    best validation loss\t\t{:.6f}".format(best_val_loss))

        if val_acc / val_batches>best_acc:
            best_acc=val_acc / val_batches
            np.savez('best_classification_model_omit5_v4.npz', *lasagne.layers.get_all_param_values(network))
            print('                    saved best classification  model')

    datasets = load_data('r180')
    X_train, y_train = datasets[0], datasets[1]
    X_val, y_val = datasets[2], datasets[3]
    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        if epoch % 8 == 7:
            learnrate*=0.96
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learnrate, momentum=0.9)
        for batch in iterate_minibatches(X_train, y_train,BATCHSIZE, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, BATCHSIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        if val_err/val_batches < best_val_loss*improvement_threshold:
            np.savez('best_model_omit5_v4.npz', *lasagne.layers.get_all_param_values(network))
            best_val_loss=val_err/val_batches
            print("                    best validation loss\t\t{:.6f}".format(best_val_loss))

        if val_acc / val_batches>best_acc:
            best_acc=val_acc / val_batches
            np.savez('best_classification_model_omit5_v4.npz', *lasagne.layers.get_all_param_values(network))
            print('                    saved best classification  model')

    datasets = load_data('r270')
    X_train, y_train = datasets[0], datasets[1]
    X_val, y_val = datasets[2], datasets[3]
    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        if epoch % 8 == 7:
            learnrate*=0.96
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learnrate, momentum=0.9)
        for batch in iterate_minibatches(X_train, y_train,BATCHSIZE, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, BATCHSIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        if val_err/val_batches < best_val_loss*improvement_threshold:
            np.savez('best_model_omit5_v4.npz', *lasagne.layers.get_all_param_values(network))
            best_val_loss=val_err/val_batches
            print("                    best validation loss\t\t{:.6f}".format(best_val_loss))

        if val_acc / val_batches>best_acc:
            best_acc=val_acc / val_batches
            np.savez('best_classification_model_omit5_v4.npz', *lasagne.layers.get_all_param_values(network))
            print('                    saved best classification  model')


    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
        main(10)
