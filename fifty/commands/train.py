# fifty/commands/train.py

# from .base import Base
import numpy as np
import os
import random
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, Dropout, MaxPool1D, GlobalAveragePooling1D, Conv1D, LeakyReLU
from keras import callbacks, backend
from keras.utils.np_utils import to_categorical
from keras.utils import multi_gpu_model
from hyperopt import partial, Trials, fmin, hp, tpe, rand

from fifty.utilities.framework import read_files, make_output_folder, load_labels_tags


class Train:

    def __init__(self, options, *args):
        random.seed(random.randint(0, 1000))

        self.input = options['<input>']
        if self.input is None:
            self.input = options['--data-dir']
        self.input = os.path.abspath(self.input)

        if options['--model-name'] is not None:
            self.model_name = os.path.abspath(options['--model-name'])
        else:
            self.model_name = None

        self.data_dir = os.path.abspath(options['--data-dir'])
        self.percent = float(options['--percent'])
        self.block_size = int(options['--block-size'])
        self.gpus = int(options['--gpus'])
        self.output = options['--output']
        self.verbose = int(options['-v'])
        self.algo = options['--algo']
        self.scale_down = options['--down']
        self.scale_up = bool(options['--up'])
        self.max_evals = int(options['--max-evals'])
        self.scenario = int(options['--scenario'])
        self.force = bool(options['--force'])
        self.recursive = bool(options['--recursive'])
        self.args = options

        self.dataset = (np.array([]), np.array([]), np.array([]), np.array([]))
        self.last_dense_layer = [75, 11, 25, 5, 2, 2]
        self.no_of_classes = self.last_dense_layer[self.scenario - 1]
        self.best_hparams = {}
        self.df = pd.DataFrame(
            columns=['dense', 'embed_size', 'filter', 'kernel', 'layers', 'pool', 'accuracy', ]).astype(
            {'dense': int, 'embed_size': int, 'filter': int, 'kernel': int, 'layers': int, 'pool': int,
             'accuracy': float, })

    def run(self):
        self.output = make_output_folder(self.input, self.output, self.force)
        self.train_model()

        if self.input is not None:
            model = self.get_model()
            from fifty.commands.whatis import WhatIs
            classifier = WhatIs(self.args)
            gen_files = read_files(self.input, self.block_size, self.recursive)
            try:
                while True:
                    file, file_name = next(gen_files)
                    pred_probability = classifier.infer(model, file)
                    classifier.output_predictions(pred_probability, file_name)
                    del file, file_name
            except Exception as e:
                print("WARNING: encountered error while predicting:", e)
        else:
            print('No input file given for inference on trained model.')
        return

    def get_model(self):
        """Finds and returns a relevant pre-trained model"""
        model = None
        if self.model_name is not None:
            try:
                if os.path.isfile(self.model_name):
                    model = load_model(self.model_name)
                else:
                    raise FileNotFoundError('Could not find the specified model! {}'.format(name_candidates))
            except RuntimeError as re:
                raise RuntimeError('Could not load the specified model! "{}"'.format(self.model_name), re)
            if self.verbose >= 2:
                print('Loaded model: "{}". \nSummary of model:'.format(self.model_name))
                model.summary()
        return model

    def make_new_dataset(self):
        labels, tags = load_labels_tags(1)
        out_data_dir = os.path.join(self.output, 'data')
        os.mkdir(out_data_dir)
        input_data_dir = os.path.join(self.data_dir, '{}_1'.format(self.block_size))

        with open(self.scale_down, 'r') as req_types:
            file_types = []
            for line in req_types:
                file_types.append(line[:-1])
            x, y = np.empty((0, self.block_size), dtype=np.uint8), np.empty(0, dtype=np.uint8)
            for file in ['train.npz', 'val.npz', 'test.npz']:
                data = np.load(os.path.join(input_data_dir, file))
                x, y = np.concatenate((x, data['x'])), np.concatenate((y, data['y']))
            scale_down_x, scale_down_y = np.empty((0, self.block_size), dtype=np.uint8), np.empty(0, dtype=np.uint8)
            for file_type in file_types:
                index_type = labels.index(file_type.lower())
                indices = np.array([i for i in range(len(y)) if y[i] == index_type])
                scale_down_x = np.concatenate((scale_down_x, x[indices]))
                scale_down_y = np.concatenate((scale_down_y, y[indices]))
            del x, y
            indices = np.arange(len(scale_down_y))
            for _ in range(10):
                random.shuffle(indices)
            scale_down_x = scale_down_x[indices]
            scale_down_y = scale_down_y[indices]
            split_train = int(len(scale_down_y) * 0.8)
            split_val = int(len(scale_down_y) * 0.9)
            np.savez_compressed(os.path.join(out_data_dir, 'train.npz'),
                                x=scale_down_x[:split_train], y=scale_down_y[:split_train])
            np.savez_compressed(os.path.join(out_data_dir, 'val.npz'),
                                x=scale_down_x[split_train: split_val], y=scale_down_y[split_train: split_val])
            np.savez_compressed(os.path.join(out_data_dir, 'test.npz'),
                                x=scale_down_x[split_val:], y=scale_down_y[split_val:])
            self.load_dataset(out_data_dir)

    def load_dataset(self, data_dir=None):
        """Loads relevant already prepared FFT-75 dataset"""
        if data_dir is None:
            if self.block_size == 4096:
                model_name = '4k_{}'.format(self.scenario)
            else:
                model_name = '512_{}'.format(self.scenario)
            data_dir = os.path.join(self.data_dir, model_name)
        else:
            self.model_name = 'new_model'
        train_data = np.load(os.path.join(data_dir, 'train.npz'))
        x_train, y_train = train_data['x'], train_data['y']
        one_hot_y_train = to_categorical(y_train)
        print("Training Data loaded with shape: {} and labels with shape - {}".format(x_train.shape,
                                                                                      one_hot_y_train.shape))

        val_data = np.load(os.path.join(data_dir, 'val.npz'))
        x_val, y_val = val_data['x'], val_data['y']
        one_hot_y_val = to_categorical(y_val)
        print(
            "Validation Data loaded with shape: {} and labels with shape - {}".format(x_val.shape, one_hot_y_val.shape))
        self.dataset = x_train, one_hot_y_train, x_val, one_hot_y_val

    def get_best(self, ):
        best_idx = self.df['accuracy'].idxmax()
        return {
            'dense': int(self.df['dense'].loc[best_idx]),
            'embed_size': int(self.df['embed_size'].loc[best_idx]),
            'filter': int(self.df['filter'].loc[best_idx]),
            'kernel': int(self.df['kernel'].loc[best_idx]),
            'layers': int(self.df['layers'].loc[best_idx]),
            'pool': int(self.df['pool'].loc[best_idx]),
        }

    def train_network(self, parameters):
        print(f"\nParameters: {parameters}")
        x_train, one_hot_y_train, x_val, one_hot_y_val = self.dataset
        # formatting data
        x_train_ = x_train[:int(np.ceil(len(x_train) * self.percent))]
        y_train_ = one_hot_y_train[:int(np.ceil(len(x_train) * self.percent))]
        x_val_ = x_val[:int(np.ceil(len(x_val) * self.percent))]
        y_val_ = one_hot_y_val[:int(np.ceil(len(x_val) * self.percent))]

        print('x_train.shape, y_train.shape:' + str((x_train_.shape, y_train_.shape)))

        try:
            model = self.build_model(parameters)

            callbacks_list = [
                callbacks.EarlyStopping(monitor='val_acc', patience=3, restore_best_weights=True, min_delta=0.01),
                callbacks.ModelCheckpoint(os.path.join(self.output, f'{self.model_name}.h5'), monitor='val_acc'),
                callbacks.CSVLogger(filename=os.path.join(self.output, f'{self.model_name}.log'), append=True)
            ]

            history = model.fit(
                x=x_train_,
                y=y_train_,
                epochs=1,
                batch_size=128,
                validation_data=(x_val_, y_val_),
                verbose=self.verbose,
                callbacks=callbacks_list
            )
            loss = min(history.history['val_loss'])
            accuracy = max(history.history['val_acc'])
            backend.clear_session()
            parameters['accuracy'] = accuracy
            self.df = self.df.append(list(parameters.values()))
        except Exception as e:
            print('!!ERROR:' + str(e))
            accuracy = 0
            loss = np.inf

        print("Loss: {}".format(loss))
        print("Accuracy: {:.2%}".format(accuracy))
        return loss

    def build_model(self, parameters):
        """
        this is just a short-hand for calling build_model()
        :param parameters: model parameters, see build_model()
        :return: model
        """
        return build_model(parameters, no_of_classes=self.no_of_classes, input_length=self.block_size, gpus=self.gpus)

    def train_model(self):
        if self.data_dir:
            self.load_dataset(self.data_dir)
        elif self.scale_down:
            self.make_new_dataset()
        elif self.scale_up:
            raise SystemExit('Please refer to the documentation.'
                             'Requires you to prepare the dataset on your own and then use -d option.')
        else:
            self.load_dataset()

        parameter_space = {
            'layers':         hp.choice('layers', [1, 2, 3]),
            'embed_size': hp.choice('embed_size', [16, 32, 48, 64]),
            'filter':         hp.choice('filter', [16, 32, 64, 128]),
            'kernel':         hp.choice('kernel', [3, 11, 19, 27, 35]),
            'pool':             hp.choice('pool', [2, 4, 6, 8]),
            'dense':           hp.choice('dense', [16, 32, 64, 128, 256])
        }

        trials = Trials()

        if self.algo.lower() == 'tpe':
            algo = partial(
                tpe.suggest,
                n_EI_candidates=1000,
                gamma=0.2,
                n_startup_jobs=int(0.1 * self.max_evals),
            )

        elif self.algo.lower() == 'rand':
            algo = rand.suggest
        else:
            print('Warning! The requested hyper-parameter algorithm is not supported. Using TPE.')
            algo = partial(
                tpe.suggest,
                n_EI_candidates=1000,
                gamma=0.2,
                n_startup_jobs=int(0.1 * self.max_evals),
            )

        fmin(
            self.train_network,
            trials=trials,
            space=parameter_space,
            algo=algo,
            max_evals=self.max_evals,
            show_progressbar=False
        )
        self.df.to_csv(os.path.join(self.output, 'parameters.csv'))
        self.best_hparams = self.get_best()
        print('\n-------------------------------------\n')
        print('Hyper-parameter space exploration ended, best hyperparams:', self.best_hparams,
              '\nRetraining the best again on the full dataset.')
        self.percent = 1
        self.train_network(self.best_hparams)
        print('The best model has been retrained and saved as "{}".'.format(self.model_name))


def build_model(parameters, no_of_classes, input_length=None, gpus=1):
    model = Sequential()
    if parameters['embed_size'] is not None:
        model.add(Embedding(256, parameters['embed_size'], input_length=input_length))
    else:  # else use autoencoder
        model.add(Dense(256, activation='tanh'))
    for _ in range(parameters['layers']):
        model.add(Conv1D(filters=int(parameters['filter']), kernel_size=parameters['kernel']))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPool1D(parameters['pool']))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(parameters['dense']))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(no_of_classes, activation='softmax'))
    # transform the model to a parallel one if multiple gpus are available.
    if gpus != 1:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    return model