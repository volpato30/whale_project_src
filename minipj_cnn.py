#!/opt/sharcnet/python/2.7.5/gcc/bin/python

import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params, l2


def load_data():
    
    a=np.load("/scratch/rqiao/miniproject.npz")
    train_data=a['arr_0']
    train_target=a['arr_1']
    valid_data=a['arr_2']
    valid_target=a['arr_3']
    return train_data,train_target,valid_data,valid_target

Conv2DLayer = lasagne.layers.Conv2DLayer
dimension=299

def convlayer(l_in, num_1xd, num_3xd, num_5xd, gain=1.0, bias=0):
    
    out_layers = []

    if num_1xd > 0:
        
        l_2xd = Conv2DLayer(l_in, num_filters=num_1xd, filter_size=(1, dimension), pad='same', W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_2xd)
    
    if num_3xd > 0:
        
        l_3xd = Conv2DLayer(l_in, num_filters=num_3xd, filter_size=(3, dimension), pad='same', W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_3xd)

    if num_5xd > 0:
        
        l_5xd = Conv2DLayer(l_in, num_filters=num_5xd, filter_size=(5, dimension), pad='same', W=lasagne.init.Orthogonal(gain), b=lasagne.init.Constant(bias))
        out_layers.append(l_5xd)
    
    # stack
    l_out = lasagne.layers.concat(out_layers)
    return l_out


def build_cnn(input_var=None):
    
    network = lasagne.layers.InputLayer(shape=(10, 1, 2000, 300),
                                        input_var=input_var)
    
    network = convlayer(network, num_1xd=32, num_3xd=32, num_5xd=32)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2000, 300))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

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


def main(num_epochs=100):
    # Load the dataset
    print("Loading data...")
    datasets = load_data()
    X_train, y_train = datasets[0], datasets[1]
    X_val, y_val = datasets[2], datasets[3]
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    learnrate=0.01
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

    patience=50
    patience_increase=2.5
    best_val_loss=10
    improvement_threshold=0.98
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_batches = 0
        start_time = time.time()
        if epoch % 8 == 7:
            learnrate*=0.96
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learnrate, momentum=0.9)
        for batch in iterate_minibatches(X_train, y_train, 10, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 10, shuffle=False):
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
            patience = max(patience, epoch * patience_increase)
            np.savez('miniproject_bestmodel.npz', *lasagne.layers.get_all_param_values(network))
            best_val_loss=val_err/val_batches
            print("best validation loss\t\t{:.6f}".format(best_val_loss))
        if patience<epoch:
            print("early stop because no significant improvement")
            break

if __name__ == '__main__':
        main(200)
