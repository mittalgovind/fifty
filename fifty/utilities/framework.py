import os
import re
import json
import pandas as pd
import numpy as np
import time
import shutil
from pathlib import Path
from pdb import set_trace

from keras import Sequential
from keras.layers import Embedding, Dense, Conv1D, LeakyReLU, MaxPool1D, GlobalAveragePooling1D, Dropout
from keras.utils import multi_gpu_model


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

    output = os.path.abspath(output)
    if os.path.exists(output):
        if force:
            print("Warning! The output folder - {} - is being overwritten.".format(output))
            shutil.rmtree(output)
            os.mkdir(output)
        else:  # load use the loaded hparams
            print(
                f'The output folder - "{output}" - already exists.'
                f'Use [-f|--force] to overwrite it completely.')
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
    model.build()
    model.summary()

    return model
