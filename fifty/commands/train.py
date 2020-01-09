# fifty/commands/train.py

import datetime
import json
import os
import random

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import partial, Trials, fmin, tpe, rand
from keras import callbacks, backend
from keras.models import load_model
from keras.utils.np_utils import to_categorical

from fifty.utilities.framework import read_files, make_output_folder, load_labels_tags
from utilities.framework import build_model
from utilities.utils import json_paramspace2hyperopt_paramspace, dict_to_safe_filename

tf.logging.set_verbosity(tf.logging.ERROR)


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Train:
    def __init__(self, options, *args):
        random.seed(random.randint(0, 1000))

        options['<input>'] = options['--data-dir'] if (options['<input>'] is None) \
            else os.path.abspath(options['<input>'])
        options['--model-name'] = None if (options['--model-name'] is None) \
            else os.path.abspath(options['--model-name'])
        options['--paramspace'] = None if (options['--paramspace'] is None) \
            else os.path.abspath(options['--paramspace'])

        self.input = options['<input>']
        self.model_name = options['--model-name']

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
        self.paramspace = options['--paramspace']
        self.epochs = int(options['--epochs'])
        self.args = args
        self.options = options

        self.dataset = (np.array([]), np.array([]), np.array([]), np.array([]))
        self.last_dense_layer = [75, 11, 25, 5, 2, 2]
        self.no_of_classes = self.last_dense_layer[self.scenario - 1]
        self.best_hparams = {}

        # setting up hparam scores dataframe
        self.df = pd.DataFrame(columns=['dense', 'embed_size', 'filter', 'kernel', 'layers', 'pool', 'accuracy'])
        params_path = os.path.join(self.output, 'parameters.csv')
        if not self.force and os.path.isfile(params_path):
            try:
                self.df = pd.read_csv(params_path).dropna(axis=0)
                print(f"Found existing parameters in \"{params_path}\"")
            except Exception as e:
                print("Couldn\'t read previous parameters: {0}".format(str(e)))

        self.df = self.df.astype(
            {'dense': int, 'embed_size': int, 'filter': int, 'kernel': int, 'layers': int, 'pool': int,
             'accuracy': float, })

    def run(self):
        self.output = make_output_folder(self.input, self.output, self.force)
        self.train_model()

        if self.input is None:
            raise Exception('No input file given for inference on trained model.')

        model = self.get_model()
        from fifty.commands.whatis import WhatIs
        classifier = WhatIs(self.options, *self.args)
        gen_files = read_files(self.input, self.block_size, self.recursive)
        try:
            while True:
                file, file_name = next(gen_files)
                pred_probability = classifier.infer(model, file)
                classifier.output_predictions(pred_probability, file_name)
                del file, file_name
        except Exception as e:
            print("!Error encountered while predicting: " + str(e))
            import traceback
            traceback.print_exc()

    def get_model(self):
        """Finds and returns a relevant pre-trained model"""
        model = None
        if self.model_name is not None:
            try:
                model_path = self.model_dir(ext='.h5')
                if os.path.isfile(model_path):
                    model = load_model(model_path)
                else:
                    raise FileNotFoundError('Could not find the specified model! "{}"'.format(model_path))
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

    def train_network(self, parameters, epochs=1, load=False):
        print(f"\nParameters: {parameters}")
        x_train, one_hot_y_train, x_val, one_hot_y_val = self.dataset
        # == formatting data ==
        # shuffle
        np.random.seed(99); np.random.shuffle(x_train)
        np.random.seed(99); np.random.shuffle(one_hot_y_train)
        np.random.seed(99); np.random.shuffle(x_val)
        np.random.seed(99); np.random.shuffle(one_hot_y_val)
        np.random.sample()
        # trim
        x_train_ = x_train[:int(np.ceil(len(x_train) * self.percent))]
        y_train_ = one_hot_y_train[:int(np.ceil(len(x_train) * self.percent))]
        x_val_ = x_val[:int(np.ceil(len(x_val) * self.percent))]
        y_val_ = one_hot_y_val[:int(np.ceil(len(x_val) * self.percent))]

        print('x_train.shape, y_train.shape: {0}'.format((x_train_.shape, y_train_.shape)))

        # load existing model if exists
        model_dir = self.model_dir(None, params=parameters)
        os.makedirs(model_dir, exist_ok=True)

        try:
            if load:
                model = load_model(self.model_dir('.h5', params=parameters))
            else:
                # default: build new model
                model = self.build_model(parameters)

            print(f"Model in:\"{self.model_dir('.h5', params=parameters)}\"")

            # tensorboard log directory
            log_dir = os.path.join(model_dir, 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            callbacks_list = [
                callbacks.EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True, min_delta=0.01),
                # saving model with exact name
                callbacks.ModelCheckpoint(self.model_dir('.h5'), monitor='val_acc'),
                callbacks.CSVLogger(filename=(self.model_dir('.log')), append=True),
                # saving model hparam_str in name
                callbacks.ModelCheckpoint(self.model_dir('.h5', params=parameters), monitor='val_acc'),
                callbacks.CSVLogger(filename=(self.model_dir('.log', params=parameters)), append=True),
            ]
            if epochs > 1:
                # tensorboard callback
                callbacks_list.append(keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=0))

            history = model.fit(
                x=x_train_,
                y=y_train_,
                epochs=epochs,
                batch_size=128,
                validation_data=(x_val_, y_val_),
                verbose=self.verbose,
                callbacks=callbacks_list
            )
            loss = min(history.history['val_loss'])
            accuracy = max(history.history['val_acc'])
            backend.clear_session()
            parameters['accuracy'] = accuracy
            self.df.loc[len(self.df)] = parameters
        except ValueError as ve:
            print('!!ERROR: {0}'.format(ve))
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
        """ explores the hyperparameter space and then trains the best model"""
        if self.data_dir:
            self.load_dataset(self.data_dir)
        elif self.scale_down:
            self.make_new_dataset()
        elif self.scale_up:
            raise SystemExit('Please refer to the documentation.'
                             'Requires you to prepare the dataset on your own and then use -d option.')
        else:
            self.load_dataset()

        # if empty dataframe, perform hparam search, else hparams space already explored
        if len(self.df) == 0 or self.force:
            try:
                # loading paramspace json file
                with open(self.paramspace, 'r', encoding='utf') as f:
                    paramspace = json_paramspace2hyperopt_paramspace(json.load(f))
            except FileNotFoundError as fnfe:
                print(f'Paramspace file not found: "{self.paramspace}", this file must exist for hyperparameter exploration, please choose it using the "--paramspace" option', fnfe)
                raise fnfe
            self.explore_hparam_space(paramspace)
        else:
            print('Hparam space already explored, using existing "parameters.csv" file')

        self.best_hparams = self.get_best()
        print('Best hyperparams:', self.best_hparams,
              '\nRetraining the best again on the full dataset.')
        self.percent = 1.0
        self.train_network(self.best_hparams, epochs=self.epochs, load=True)
        print('The best model has been retrained and saved as "{}.h5"'.format(self.model_name))

    def explore_hparam_space(self, parameter_space: dict):
        """ explores hparam space according to the algorithm chosen and saves results in "output/parameters.csv" """
        print('Hparam space not explored, exploring hparam space...')
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
        self.df.to_csv(os.path.join(self.output, 'parameters.csv'), index=False)
        print('\n-------------------------------------\n')
        print('Hyper-parameter space exploration ended')

    def model_dir(self, ext=None, params=None):
        """
        :param params: dict
        :param ext: model extension (includes '.'). if None, will return a folder path only, not filename appended
        :return:
        """
        path = self.output

        # append_params
        if params is not None:
            path = os.path.join(path, f'{self.model_name}_{dict_to_safe_filename(params)}')

        if ext is not None:
            path = os.path.join(path, f'{self.model_name}{ext}')

        return path
