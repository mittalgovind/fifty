# fifty/commands/train.py

import json
import os
import random
import time

import numpy as np
import pandas as pd
from hyperopt import partial, Trials, fmin, tpe, rand

import keras
from keras import callbacks, backend, models
from keras.utils.np_utils import to_categorical

from fifty.utilities.framework import read_files, make_output_folder, load_labels_tags
from utilities.framework import build_model, build_autoencoder
from utilities.utils import json_paramspace2hyperopt_paramspace, dict_to_safe_filename
from matplotlib import pyplot as plt


# tf.logging.set_verbosity(tf.logging.ERROR)
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
        self.epochs = int(options['--epochs'])
        self.is_train_autoencoder = bool(options['--autoencoder'])
        self.dataset = (np.array([]), np.array([]), np.array([]), np.array([]))
        self.best_hparams = {}

        self.args = args
        self.options = options

        self.paramspace = options['--paramspace']
        if self.paramspace == "":  # if not passed, autoselect
            self.paramspace = './hparamspace.json' if not self.is_train_autoencoder \
                else './hparamspace_autoencoder.json'
        # loading hparamspace json
        pspace_file = self.paramspace
        try:
            # loading paramspace json file
            with open(pspace_file, 'r', encoding='utf') as f:
                self.paramspace = json_paramspace2hyperopt_paramspace(json.load(f))
        except FileNotFoundError as fnfe:
            print(f'Paramspace file not found: "{pspace_file}". '
                  f'Please specify a valid file for hyperparameter exploration using the "--paramspace" option', fnfe)
            raise fnfe

        # setting up hparam scores dataframe
        types_dict = {'dense': int, 'embed_size': int, 'encoder': str, 'filter': int, 'kernel': int, 'layers': int,
                      'pool': int, 'accuracy': float, 'loss': float, }

        self.df = None
        # try to load dataframe
        params_path = os.path.join(self.output, 'parameters.csv')
        if not self.force and os.path.isfile(params_path):
            try:
                self.df = pd.read_csv(params_path).fillna(0)  # replace NaN values with 0
                print(f'Found existing parameters in "{params_path}" with {len(self.df)} entries')
            except Exception as e:
                print("Couldn't read previous parameters: {0}".format(str(e)))
        # if couln't load, create an empty one
        if self.df is None:
            # limit the attributes to only those in the paramspace json
            types_dict = {k: v for k, v in types_dict.items() if
                          k in (['accuracy', 'loss'] + list(self.paramspace.keys()))}
            self.df = pd.DataFrame(columns=list(types_dict.keys())).astype(types_dict)

        # checking --force option
        if not len(self.df) and not self.force:
            raise FileNotFoundError(
                f"The output run params weren't loaded/don't already exist/are empty: \"{self.output}\". "
                "Ensure a valid output exists, "
                "or force creation of a new output by specify the [-f|--force] option.")

    def run(self):
        self.output = make_output_folder(self.input, self.output, self.force)
        self.train_model()

        if self.input is None:
            raise Exception('No input blocks given for inference on trained model.')

        model = self.get_model()

        print('evaluating model...')
        gen_files = read_files(self.input, self.block_size, self.recursive)

        if self.is_train_autoencoder:
            try:
                start_time = time.time()
                for i, (blocks, file_name) in enumerate(gen_files):
                    # keep only 1/100 of the elements (for performance)
                    blocks = blocks[:int(np.ceil(len(blocks) * 0.01))]

                    print('loaded files in\t{:.2f}s.\t'.format(time.time() - start_time), end='')
                    time_ = time.time()

                    output = model.predict(blocks)

                    print('predicted in\t{:.2f}s.\t'.format(time.time() - time_), end='')
                    time_ = time.time()

                    # flattening
                    plt.plot(blocks.flatten(), label='input', alpha=0.7, linewidth=0.5)
                    plt.plot(output.flatten(), label='output', alpha=0.7, linewidth=0.5)
                    plt.title(f'Autoencoder reconstruction {i}')
                    plt.xlabel(f'Byte index')
                    plt.ylabel(f'Byte value')

                    plt.legend(loc='upper right')

                    plt.savefig(os.path.join(self.model_dir(), f'reconstruction_{i}_{time.time()}.png'))
                    plt.show()

                    print('plotted in\t{:.2f}s, total time:\t{:.2f}s'.format(
                        time.time() - time_, time.time() - start_time))

                    del blocks, file_name
                    start_time = time.time()

            except Exception as e:
                print("!Error encountered while predicting: " + str(e))
                import traceback
                traceback.print_exc()
        else:
            from fifty.commands.whatis import WhatIs
            classifier = WhatIs(self.options, *self.args)
            try:
                start_time = time.time()
                for blocks, file_name in gen_files:
                    print('loaded files in\t{:.2f}s.\t'.format(time.time() - start_time), end='')
                    time_ = time.time()

                    pred_probability = classifier.infer(model, blocks)
                    print('predicted in\t{:.2f}s.\t'.format(time.time() - time_), end='')
                    time_ = time.time()

                    classifier.output_predictions(pred_probability, file_name)
                    print('plotted in\t{:.2f}s, total time:\t{:.2f}s'.format(
                        time.time() - time_, time.time() - start_time))

                    del blocks, file_name
                    start_time = time.time()
            except Exception as e:
                print("!Error encountered while predicting: " + str(e))
                import traceback
                traceback.print_exc()

    def get_model(self) -> keras.Model:
        """Finds and returns a relevant pre-trained model"""
        model = None
        if self.model_name is not None:
            try:
                model_path = self.model_dir(ext='.h5')
                if os.path.isfile(model_path):
                    model = models.load_model(model_path)
                else:
                    raise FileNotFoundError('Could not find the specified model! "{}"'.format(model_path))
            except RuntimeError as re:
                raise RuntimeError('Could not load the specified model! "{}"'.format(self.model_name), re)

            print('Loaded model: "{}".'.format(self.model_dir('.h5')))
            # if self.verbose >= 2:
            #     model.summary()
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

    def get_best(self) -> dict:
        """ :return: dictionary containing the best hyperparameter keys and values """
        if len(self.df) == 0:
            raise Exception("get_best(): hparam dataframe is empty, there were no successfully trained models. "
                            "Make sure the hyperparameters are correct, also try increasing the --max-evals value.")

        # if "accuracy" is unique, then use it
        if 'accuracy' in self.df and self.df[['accuracy']].nunique().sum() > 1:
            best_idx = self.df['accuracy'].idxmax()
            print("Best params achieve accuracy={}".format(self.df.loc[best_idx, 'accuracy']))
        else:
            best_idx = self.df['loss'].idxmin()
            print('info: while getting best params, '
                  '"accuracy" value not found in df or is not unique enough to sort by, '
                  'using "loss" to sort instead.')
            print("Best params achieve loss={}".format(self.df.loc[best_idx, 'loss']))

        return {col: (self.df.loc[best_idx, col]) for col in self.df.columns}

    def train_network(self, parameters, epochs=1, load=False):
        batch_size = 128
        x_train, one_hot_y_train, x_val, one_hot_y_val = self.dataset
        no_of_classes = one_hot_y_train.shape[1]

        if ('embed_size' in parameters and parameters['embed_size']) and \
                ('encoder' in parameters and parameters['encoder']):
            del parameters['encoder']

        print(f"\ntrain_network(parameters={parameters}, epochs={epochs}, load={load}), no_of_classes={no_of_classes}")

        # == formatting data ==
        # trim
        x_train_ = x_train[:int(np.ceil(len(x_train) * self.percent))]
        y_train_ = one_hot_y_train[:int(np.ceil(len(x_train) * self.percent))]
        x_val_ = x_val[:int(np.ceil(len(x_val) * self.percent))]
        y_val_ = one_hot_y_val[:int(np.ceil(len(x_val) * self.percent))]

        ## Training Data loaded with shape:   (819200, 512) and labels with shape - (819200, 2)
        ## Validation Data loaded with shape: (102400, 512) and labels with shape - (102400, 2)

        # load existing model if exists
        model_dir = self.model_dir(None, params=parameters)

        log_dir = os.path.join(model_dir, 'fit')

        parameters['accuracy'] = 0.
        parameters['loss'] = np.inf

        try:
            if load:  # load old model
                print('loading old model...', end=' ')
                start_time = time.time()

                model = models.load_model(self.model_dir('.h5', params=parameters))

                print('old model loaded in {:.2f}s.'.format(time.time() - start_time))
            else:
                # default: build new model
                if self.is_train_autoencoder:
                    optimizer = keras.optimizers.adadelta(learning_rate=1.0, rho=0.95)
                    loss_func = keras.losses.mse

                    encoder, decoder, autoencoder = build_autoencoder(parameters, input_shape=(self.block_size,),
                                                                      gpus=self.gpus, optimizer=optimizer,
                                                                      loss=loss_func)
                    model = autoencoder
                else:
                    # manually defining optimizer and loss
                    optimizer = keras.optimizers.rmsprop(lr=0.0005, rho=0.9)
                    loss_func = keras.losses.categorical_crossentropy

                    model = build_model(parameters, no_of_classes=no_of_classes, input_length=self.block_size,
                                        gpus=self.gpus, optimizer=optimizer, loss=loss_func)

            if self.is_train_autoencoder:
                y_train_ = x_train_
                y_val_ = x_val_

            print(f"Model path: \"{self.model_dir('.h5', params=parameters)}\"")

            monitor = 'val_loss' if self.is_train_autoencoder else 'val_acc'

            callbacks_list = [
                callbacks.EarlyStopping(monitor=monitor, patience=5, restore_best_weights=True, min_delta=0.03),
                # saving model with exact name
                callbacks.ModelCheckpoint(self.model_dir('.h5'), monitor=monitor),
                callbacks.CSVLogger(filename=(self.model_dir('.log')), append=True),
                # saving model hparam_str in name
                callbacks.ModelCheckpoint(self.model_dir('.h5', params=parameters), monitor=monitor),
                callbacks.CSVLogger(filename=(self.model_dir('.log', params=parameters)), append=True),
            ]
            if epochs > 1:
                # tensorboard logging
                callbacks_list.append(callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=0))

            os.makedirs(model_dir, exist_ok=True)
            model.summary()
            history = model.fit(
                x=x_train_,
                y=y_train_,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val_, y_val_),
                verbose=self.verbose,
                callbacks=callbacks_list,
            )
            try:
                keras.utils.plot_model(model, self.model_dir('.png'), show_shapes=True)
            except Exception as e:
                print(f"Couln't plot diagram: {e}")

            if 'val_loss' in history.history:
                parameters['loss'] = max(history.history['val_loss'])
            if 'val_acc' in history.history:
                parameters['accuracy'] = max(history.history['val_acc'])

            print()
            self.df.loc[len(self.df)] = parameters
            backend.clear_session()
        except ValueError as ve:
            print('!!ERROR: {0}'.format(ve))

            # raise exception if not whitelisted
            whitelisted_exceptions = ['Negative dimension size caused by subtracting',
                                      'Error when checking target: expected ',
                                      '']
            if not list(filter(str(ve).__contains__, whitelisted_exceptions)):
                raise ve

        print("Loss: {}".format(parameters['loss']))
        print("Accuracy: {:.2%}".format(parameters['accuracy']))
        return parameters['loss']

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

        # we know at this point that either --force is set, or there exists a dataset
        if self.force:
            # if empty dataframe, perform hparam search, else hparams space already explored
            self.explore_hparam_space(self.paramspace)
        elif len(self.df) > 0:
            print('\n-------------------------------------\n')
            print('Hparam space already explored, using existing "parameters.csv" file')
        else:
            raise Exception()

        self.best_hparams = self.get_best()
        print('Best hyperparams:', self.best_hparams,
              '\nRetraining best network on the full dataset...')
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

    def model_dir(self, ext=None, params=None) -> str:
        """
        :param params: dict
        :param ext: model extension (includes '.'). if None, will return a folder path only, not filename appended
        :return:
        """
        path = self.output
        # removing blacklisted attributes
        if params is not None:
            params = {k: v for k, v in params.items() if k not in ['loss', 'accuracy']}
            if 'encoder' in params:
                params['encoder'] = os.path.split(params['encoder'])[-1]
            for k, v in params.items():
                if type(v) is float:
                    params[k] = int(v)

        # append_params
        if params is not None:
            path = os.path.join(path, f'{self.model_name}{dict_to_safe_filename(params)}')

        if ext is not None:
            path = os.path.join(path, f'{self.model_name}{ext}')

        return path
