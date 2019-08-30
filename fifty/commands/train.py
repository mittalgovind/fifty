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
        self.input = os.path.abspath(options['<input>'])
        if options['--model-name'] is not None:
            self.model_name = os.path.abspath(options['--model-name'])
        else:
            self.model_name = None
        self.data_dir = options['--data-dir']
        self.percent = float(options['--percent'])
        self.block_size = options['--block_size']
        self.gpu = int(options['--gpus'])
        self.output = options['--output']
        self.verbose = int(options['-v'])
        self.algo = options['--algo']
        self.down = options['--down']
        self.up = options['--up']
        self.max_evals = int(options['--max-evals'])
        self.args = options

        self.dataset = ()
        self.last_dense_layer = [75, 11, 25, 5, 2, 2]
        self.no_of_classes = last_dense_layer[scenario - 1]
        self.df = pd.DataFrame(columns=['dense', 'embed_size', 'filter', 'kernel', 'layers', 'pool', 'accuracy'])

    def run(self):
        self.output = make_output_folder(self.input, self.output, self.force)
        train()

        if self.input is not None:
            model = get_model()
            files = read_files(self.input, self.block_size, self.recursive)
            from fifty.commands.whatis import WhatIs
            classifier = WhatIs(self.args)
            for file in files:
                pred_probability = classifier.infer(model, file)
                classifier.output_predictions(pred_probability)
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
                    raise FileNotFoundError('Could not find the specified model! {}'.format(self.model_name))
            except RuntimeError:
                raise RuntimeError('Could not load the specified model! {}'.format(self.model_name))
            if self.verbose == 3:
                print('Loaded model: {}. \nSummary of model:'.format(self.model_name))
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
            load_dataset(self, out_data_dir)

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
        best_idx = df['accuracy'].idxmax()
        best = dict()
        best['dense'] = int(df['dense'].loc[best_idx])
        best['embed_size'] = int(df['embed_size'].loc[best_idx])
        best['filter'] = int(df['filter'].loc[best_idx])
        best['kernel'] = int(df['kernel'].loc[best_idx])
        best['layers'] = int(df['layers'].loc[best_idx])
        best['pool'] = int(df['pool'].loc[best_idx])
        return best

    def train_network(self, parameters):
        print("\nParameters:")
        print(parameters)
        x_train, one_hot_y_train, x_val, one_hot_y_val = self.dataset

        try:
            model = Sequential()
            model.add(Embedding(256, parameters['embed_size'], input_length=block_size))
            for _ in range(parameters['layers']):
                model.add(Conv1D(filters=int(parameters['filter']), kernel_size=parameters['kernel']))
                model.add(LeakyReLU(alpha=0.3))
                model.add(MaxPool1D(parameters['pool']))

            model.add(GlobalAveragePooling1D())
            model.add(Dropout(0.1))
            model.add(Dense(parameters['dense']))
            model.add(LeakyReLU(alpha=0.3))
            model.add(Dense(self.no_of_classes, activation='softmax'))
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
            history = model.fit(
                x=x_train[:int(len(x_train) * percent)],
                y=one_hot_y_train[:int(len(x_train) * percent)],
                epochs=1, batch_size=128, validation_data=(
                    x_val[:int(len(x_val) * percent)], one_hot_y_val[:int(len(x_val) * percent)]),
                verbose=verbose, callbacks=callbacks_list)
            loss = min(history.history['val_loss'])
            accuracy = max(history.history['val_acc'])
            backend.clear_session()
            parameters['accuracy'] = accuracy
            df.loc[len(df)] = list(parameters.values())
        except:
            accuracy = 0
            loss = np.inf

        print("Loss: {}".format(loss))
        print("Accuracy: {:.2%}".format(accuracy))
        return loss

    def train_model(self):
        if self.data_dir:
            load_dataset(self, self.data_dir)
        elif self.scale_down:
            
            make_new_dataset(self)
        elif self.scale_up:
            raise SystemExit(
                'Please refer documentation. Requires you to prepare the dataset on your own and then use -d option.')
        else:
            load_dataset(self)
        # updating global variables. train_network only takes one and only one argument.
        global percent, block_size, scenario, gpu, output, verbose, new_model, no_of_classes
        percent = self.percent
        block_size = self.block_size
        scenario = self.scenario
        gpu = self.gpus
        output = self.output
        new_model = self.new_model
        if self.scale_down:
            no_of_classes = len(list(open(self.scale_down, 'r')))
        if self.v:
            verbose = 0
        elif self.vv:
            verbose = 1
        elif self.vvv:
            verbose = 2
        parameter_space = {
            'layers': hp.choice('layers', [1, 2, 3]),
            'embed_size': hp.choice('embed_size', [16, 32, 48, 64]),
            'filter': hp.choice('filter', [16, 32, 64, 128]),
            'kernel': hp.choice('kernel', [3, 11, 19, 27, 35]),
            'pool': hp.choice('pool', [2, 4, 6, 8]),
            'dense': hp.choice('dense', [16, 32, 64, 128, 256])
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
            train_network,
            trials=trials,
            space=parameter_space,
            algo=algo,
            max_evals=self.max_evals,
            show_progressbar=False
        )
        df.to_csv(os.path.join(self.output, 'parameters.csv'))
        best = get_best()
        print('\n-------------------------------------\n')
        print('Hyper-parameter space exploration ended. \nRetraining the best again on the full dataset.')
        percent = 1
        train_network(best)
        print('The best model has been retrained and saved as {}.'.format(self.new_model))
