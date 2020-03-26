import os
import json
import time

import numpy as np
import shutil
from pathlib import Path

import keras
import tensorflow as tf
from keras.layers import Embedding, Dense, Conv1D, LeakyReLU, MaxPool1D, GlobalAveragePooling1D, Dropout, Reshape, \
    Input, UpSampling1D


def get_utilities_dir():
    return os.path.dirname(__file__)


def read_file(path, block_size):
    data = open(path, 'rb').read()
    if len(data) < block_size:
        print('Skipping {}. Smaller than one block size ({} bytes). Try smaller block size.'.format(path, block_size))
        return None
    bound = (len(data) // block_size) * block_size
    data = data[:bound]
    file = np.array(np.array(list(data), dtype=np.uint8).reshape((-1, block_size)))
    del data
    return file


def read_files(input, block_size, recursive):
    """Reads the data disk or folder for inference"""
    if os.path.isfile(input):
        file_block = read_file(input, block_size)
        if file_block is not None:
            yield file_block, os.path.split(input)[1]
    elif os.path.exists(input):
        pattern = '**/*' if recursive else './*'
        for path in Path(input).glob(pattern):
            path = path.as_posix()
            if os.path.isfile(path):
                file_block = read_file(path, block_size)
                if file_block is not None:
                    try:
                        file_name = os.path.split(path)[1]
                    except:
                        file_name = 'non-alphanumeric-name.xyz'
                    yield file_block, file_name
    else:
        raise FileNotFoundError('Could not find {}'.format(input))


def make_output_folder(input, output, force=False):
    """Prepares output folder"""
    if not output:
        file_name = os.path.split(input)[1]
        if os.path.isfile(input) or os.path.isdir(input):
            if file_name.rfind('.') != -1:
                output = file_name[:file_name.rfind('.')]
            else:
                output = 'fifty_{}'.format(file_name)

    output = (output)
    if os.path.exists(output):
        if force:
            print('Warning! The output folder is being overwritten: "{}"'.format(output))
            try:
                shutil.rmtree(output)
            except:
                pass
        else:  # load use the loaded hparams
            print(f'The output folder already exists: "{output}"'
                  f'\n  use [-f|--force] to overwrite it completely.')
    os.makedirs(output, exist_ok=True)
    return output


def load_labels_tags(scenario):
    """Loads class labels and tags"""
    if os.path.isfile(os.path.join(get_utilities_dir(), 'labels.json')):
        with open(os.path.join(get_utilities_dir(), 'labels.json')) as json_file:
            classes = json.load(json_file)
            labels = classes[str(scenario)]
            tags = classes['tags']
    else:
        raise FileNotFoundError('Please download labels.json to {} directory!'.format(get_utilities_dir()))
    return labels, tags


def build_model(parameters, no_of_classes, input_length, gpus=1, optimizer='rmsprop',
                loss=keras.losses.categorical_crossentropy):
    input_shape = (input_length,)
    model = keras.Sequential()

    if parameters['embed_size'] is not 0:
        model.add(Embedding(256, parameters['embed_size'], input_length=input_length))
    elif parameters['encoder']:  # else use autoencoder
        start_time = time.time()
        print(f'loading encoder: "{parameters["encoder"]}"...', end='')
        try:
            encoder = keras.models.load_model(parameters['encoder']).layers[1]
        except FileNotFoundError as fnfe:
            raise FileNotFoundError(f'Error loading encoder: "{parameters["encoder"]}"'
                                    f'the "encoder" argument must be a path to a saved encoder model, '
                                    f'example: "./encoder_output/encoder_model.h5"', fnfe)

        print('encoder loaded in {:.2f}'.format(time.time() - start_time))

        model.add(encoder)
    else:
        raise ValueError(
            'Both "embed_size"={} and "encoder"={} are invalid values. At least one of them must have a value.'.format(
                parameters["embed_size"], parameters["encoder"]))

    if parameters['layers'] <= 0:
        raise ValueError(
            '\"layers\" parameter must be a positive integer, got \"layers\"={0}'.format(parameters['layers']))

    for _ in range(parameters['layers']):
        model.add(Conv1D(filters=parameters['filter'], kernel_size=parameters['kernel']))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPool1D(parameters['pool']))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(parameters['dense']))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(no_of_classes, activation='softmax'))

    # transform the model to a parallel one if multiple gpus are available.
    if gpus != 1:
        model = keras.utils.multi_gpu_model(model, gpus=gpus)

    if optimizer is not None and loss is not None:
        # compiling model
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        model.build(input_shape=input_shape)
        # model.summary()

    return model


def build_autoencoder(parameters: dict, input_shape: tuple, gpus=1, optimizer='adadelta', loss='mse'):
    """
    :param parameters: dictionary containing params {layers, filter, kernel, pool}
    :param input_shape:
    :param loss:
    :param optimizer:
    :return: encoder, decoder, autoencoder
    """
    input_data = Input(shape=input_shape)
    autoenc_input_shape = input_shape + (1,)

    # reshaping the data
    x = Reshape(autoenc_input_shape)(input_data)

    # ENCODER architecture
    for i in range(parameters['layers']):
        x = Conv1D(parameters['filter'] // (2 ** i), parameters['kernel'], activation='relu', padding='same')(x)
        x = MaxPool1D(parameters['pool'], padding='same')(x)
    encoded = x
    encoder = keras.models.Model(input_data, encoded)
    encoder.name = 'encoder'

    # DECODING architecture
    decoder_input = Input(shape=encoded.shape[1:])  # removing the batch dimension for the encoded shape
    x = decoder_input
    for i in reversed(range(parameters['layers'])):
        x = Conv1D(parameters['filter'] // (2 ** i), parameters['kernel'], activation='relu', padding='same')(x)
        x = UpSampling1D(parameters['pool'])(x)
    decoded = Conv1D(autoenc_input_shape[-1], parameters['kernel'], padding='same')(x)
    decoded = Reshape(decoded.shape[1:-1])(decoded)  # squeezing dimensions, removing that last (..., 1) in the shape
    # define decoder model
    decoder = keras.models.Model(decoder_input, decoded)
    decoder.name = 'decoder'

    # AUTOENCODER, connecting encoder and decoder
    input_ = Input(shape=input_shape)
    encoded_ = encoder(input_)
    decoded_ = decoder(encoded_)

    autoencoder = keras.models.Model(input_, decoded_)
    autoencoder.name = 'autoencoder'

    # transform the model to a parallel one if multiple gpus are available.
    if gpus != 1:
        autoencoder = keras.utils.multi_gpu_model(autoencoder, gpus=gpus)

    autoencoder.compile(optimizer=optimizer, loss=loss)

    print('Autoencoder structure: input:{} -> encoded:{} -> decoded:{}. '
          'Dimensionality reduction: {:.2f}% (lower is better)'.format(
        input_.shape, encoded_.shape, decoded_.shape, np.prod(encoded_.shape[1:]) / np.prod(input_.shape[1:]) * 100))

    return encoder, decoder, autoencoder


def get_latest_tfevent_summary(log_dir):
    latest_tfeventfile_path = os.path.join(log_dir, max(os.listdir(log_dir)))
    # get last summary with max step
    iterator = tf.train.summary_iterator(latest_tfeventfile_path)
    latest_summary = max(iterator, key=lambda s: s.step)
    return latest_summary
