from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, Dropout, MaxPool1D, GlobalAveragePooling1D, Conv1D, LeakyReLU
from keras import callbacks, backend
from keras.utils.np_utils import to_categorical
from keras.utils import multi_gpu_model
import numpy as np
import os
import random
from hyperopt import partial, Trials, fmin, hp, tpe, rand
from framework import load_labels_tags

model_name = ''
new_model = ''
dataset = ()
percent = 0.35
block_size = 4096
scenario = 1
gpu = 1
output = ''
last_dense_layer = [75, 11, 25, 5, 2, 2]
no_of_classes = last_dense_layer[scenario - 1]


def make_new_dataset(args):
    labels, tags = load_labels_tags(1)
    data_dir = os.path.join(args.output, 'data')
    os.mkdir(data_dir)
    data_dir = os.path.join(args.data_dir, '{}_1'.format(args.block_size))
    global no_of_classes
    with open(args.scale_down) as file_types:
        file_types = list(file_types)
        no_of_classes = len(file_types)
        x, y = np.empty((0, args.block_size), dtype=np.uint8), np.empty(0, dtype=np.uint8)
        for file in ['train.npz', 'val.npz', 'test.npz']:
            data = np.load(os.path.join(data_dir, file))
            x, y = np.concatenate((x, data['x'])), np.concatenate((y, data['y']))
        scale_down_x, scale_down_y = np.empty((0, args.block_size), dtype=np.uint8), np.empty(0, dtype=np.uint8)
        for file_type in file_types:
            index_ftype = labels.index(file_type.lower())
            indices = np.array([i for i in range(len(y)) if y[i] == index_ftype])
            scale_down_x = np.concatenate((scale_down_x, x[indices]))
            scale_down_y = np.concatenate((scale_down_y, y[indices]))
        indices = np.arange(len(scale_down_y))
        for i in range(100):
            random.shuffle(indices)
        scale_down_x = scale_down_x[indices]
        scale_down_y = scale_down_y[indices]
        split_train = int(len(scale_down_y) * 0.8)
        split_val = int(len(scale_down_y) * 0.9)
        np.savez_compressed(os.path.join(data_dir, 'train.npz'), x=scale_down_x[:split_train],
                            y=scale_down_y[:split_train])
        np.savez_compressed(os.path.join(data_dir, 'val.npz'),
                            x=scale_down_x[split_train: split_val], y=scale_down_y[split_train: split_val])
        np.savez_compressed(os.path.join(data_dir, 'test.npz'), x=scale_down_x[split_val:], y=scale_down_y[split_val:])
    load_fft_75(args, data_dir)


def load_fft_75(args, data_dir=None):
    """Loads relevant already prepared FFT-75 dataset"""
    global model_name, dataset
    if data_dir is None:
        if args.block_size == 4096:
            model_name = '4k_{}'.format( args.scenario)
        else:
            model_name = '512_{}'.format( args.scenario)
        data_dir = os.path.join(args.data_dir, model_name)
    else:
        model_name = args.new_model
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    x_train, y_train = train_data['x'], train_data['y']
    one_hot_y_train = to_categorical(y_train)
    print("Training Data loaded with shape: {} and labels with shape - {}".format(x_train.shape, one_hot_y_train.shape))

    val_data = np.load(os.path.join(data_dir, 'val.npz'))
    x_val, y_val = val_data['x'], val_data['y']
    one_hot_y_val = to_categorical(y_val)
    print("Validation Data loaded with shape: {} and labels with shape - {}".format(x_val.shape, one_hot_y_val.shape))
    dataset = x_train, one_hot_y_train, x_val, one_hot_y_val


def train_network(parameters):
    print("\nParameters:")
    print(parameters)
    global dataset
    x_train, one_hot_y_train, x_val, one_hot_y_val = dataset

    try:
        model = Sequential()
        model.add(Embedding(256, parameters['embed_size'], input_length=block_size))
        for i in range(parameters['layers']):
            model.add(Conv1D(filters=int(parameters['filter']), kernel_size=parameters['kernel']))
            model.add(LeakyReLU(alpha=0.3))
            model.add(MaxPool1D(parameters['pool']))

        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.1))
        model.add(Dense(parameters['dense']))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(no_of_classes, activation='softmax'))
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_acc', patience=3, restore_best_weights=True, min_delta=0.01),
            callbacks.ModelCheckpoint(os.path.join(output, '{}.h5'.format(new_model)), monitor='val_acc'),
            callbacks.CSVLogger(filename=os.path.join(output, '{}.log'.format(new_model)), append=True)
        ]

        # transform the model to a parallel one if multiple gpus are available.
        if gpu != 1:
            model = multi_gpu_model(model, gpus=gpu)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()
        history = model.fit(x=x_train[:int(len(x_train) * percent)],
                            y=one_hot_y_train[:int(len(x_train) * percent)],
                            epochs=50, batch_size=128, validation_data=(
                x_val[:int(len(x_val) * percent)], one_hot_y_val[:int(len(x_val) * percent)]),
                            verbose=2, callbacks=callbacks_list)
        score = min(history.history['val_loss'])
        accuracy = max(history.history['val_acc'])
        backend.clear_session()
    except RuntimeError:
        accuracy = 0
        score = np.inf

    print("Final score: {}".format(score))
    print("Accuracy: {:.2%}".format(accuracy))

    return score


def train(args):
    if args.scale_down:
        make_new_dataset(args)
    elif args.scale_up:
        raise SystemExit(
            'Please refer documentation. Requires you to prepare the dataset on your own and then use -d option.')
    else:
        load_fft_75(args)

    # updating global variables. train_network only takes one and only one argument.
    global percent, block_size, scenario, gpu, output
    percent = args.percent
    block_size = args.block_size
    scenario = args.scenario
    gpu = args.gpus
    output = args.output

    parameter_space = {
        'layers': hp.choice('layers', [1, 2, 3]),
        'embed_size': hp.choice('embed_size', [16, 32, 48, 64]),
        'filter': hp.choice('filter', [16, 32, 64, 128]),
        'kernel': hp.choice('kernel', [3, 11, 19, 27, 35]),
        'pool': hp.choice('pool', [2, 4, 6, 8]),
        'dense': hp.choice('dense', [16, 32, 64, 128, 256])
    }

    trials = Trials()

    if args.algo.lower() == 'tpe':
        hyper_algo = tpe.suggest
    elif args.algo.lower() == 'rand':
        hyper_algo = rand.suggest
    else:
        print('ERROR! The requested hyper-parameter algorithm is not supported. Using TPE.')
        hyper_algo = tpe.suggest

    algo = partial(
        hyper_algo,
        n_EI_candidates=1000,
        gamma=0.2,
        n_startup_jobs=int(0.1*args.max_evals),
    )

    best = fmin(
        train_network,
        trials=trials,
        space=parameter_space,
        algo=algo,
        max_evals=args.max_evals,
        show_progressbar=False
    )

    print(args, best)
    # retrain the best again on the full dataset
    args.percent = 1.0
    train_network(best)
    print('The best model has been retrained and saved as {}.'.format(args.model_name))