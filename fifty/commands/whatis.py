# fifty/commands/whatis.py

# from .base import Base
import os
import re
import json
import pandas as pd
import numpy as np
import time
import shutil
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import load_model
from fifty.utilities.framework import read_files, make_output_folder, load_labels_tags, get_utilities_dir

from pdb import set_trace


class WhatIs:
    def __init__(self, options, *args):
        self.input = os.path.abspath(options['<input>'])
        self.file_name = options['<input>']
        self.recursive = options['--recursive']
        self.block_size = int(options['--block-size'])
        self.block_wise = options['--block-wise']
        self.scenario = int(options['--scenario'])
        self.output = options['--output']
        self.force = options['--force']
        self.light = options['--light']
        self.verbose = int(options['-v'])
        if options['--model-name'] is not None:
            self.model_name = os.path.abspath(options['--model-name'])  
        else:
            self.model_name = None
        self.args = args

    def run(self):
        if self.input is None:
            parser.print_usage()
            sys.exit(1)

        if self.verbose >= 1:
            self.output = make_output_folder(self.input, self.output, self.force)
        model = self.get_model()
        files, file_names = read_files(self.input, self.block_size, self.recursive)
        for file, file_name in zip(files, file_names):
            pred_probability = self.infer(model, file)
            self.output_predictions(pred_probability, file_name)
        return

    def get_model(self):
        """Finds and returns a relevant pre-trained model"""
        if self.model_name is not None:
            try:
                if os.path.isfile(self.model_name):
                    model = load_model(self.model_name)
                else:
                    raise FileNotFoundError('Could not find the specified model! {}'.format(self.model_name))
            except RuntimeError:
                raise RuntimeError('Could not load the specified model! {}'.format(self.model_name))
        else:
            if self.block_size not in [512, 4096]:
                raise ValueError('Invalid block size!')
            if self.scenario not in range(1, 7):
                raise ValueError('Invalid scenario!')
            try:
                self.model_name = '{}_{}.h5'.format(self.block_size, self.scenario)
                if self.light:
                    if self.model_name == '4096_1.h5':
                        self.model_name = '4096_1_lighter.h5'
                    else:
                        print('Warning! Lighter version of this case is not available. Using the standard version.')
                model = load_model(os.path.join(get_utilities_dir(), 'models/{}'.format(self.model_name)))
            except RuntimeError:
                raise RuntimeError(
                    'Model unavailable for block size of {} bytes and scenario {}.'.format(self.block_size,
                                                                                           self.scenario))
        if self.verbose == 3:
            print('Loaded model: {}. \nSummary of model:'.format(self.model_name))
            model.summary()
        return model

    def infer(self, model, blocks):
        """Runs the model on the disk image"""
        if self.verbose == 3:
            print('Predicting..... Please be patient!')
        tic = time.perf_counter()
        pred_probability = model.predict_proba(blocks)
        toc = time.perf_counter()
        if self.verbose == 3:
            print('Inference time per sample = {} ms'.format((toc - tic) * 1000 / len(blocks)))
            print('Prediction complete!')
        return pred_probability

    def output_predictions(self, pred_probability, file_name):
        """Saves prediction in relevant format to disk"""
        labels, tags = load_labels_tags(self.scenario)
        pred_class = np.argmax(pred_probability, axis=1)
        pred_label = [labels[i] for i in pred_class]
        pred_probability = np.round(np.max(pred_probability, axis=1) * 100, decimals=1)
        tags = [tags[i] for i in pred_class]
        df = pd.DataFrame(
            {
                'Class Number': pred_class,
                'Class Label': pred_label,
                'Class Probability': pred_probability,
                'Tag': tags
            },
            columns=['Class Number', 'Class Label', 'Class Probability', 'Tag']
        )

        top_labels = df['Class Label'].value_counts()[:4]
        top_labels *= 100/top_labels.sum()
        output = '{}: ('.format(file_name)
        for label, percent  in top_labels.items():
            output += '{}: {:.1f}, '.format(label.upper(), percent)

        print('{})'.format(output[:-2]))

        if self.verbose >= 1:
            try:
                if '.' in file_name:
                    file_name = file_name[:file_name.rfind('.')]
                out_file = open(os.path.join(self.output, '{}.csv'.format(file_name)), 'w')
            except:
                set_trace()
            if self.verbose == 3:
                df.to_csv(out_file, sep=',', encoding='utf-8', index=False)
            elif self.verbose == 1:
                df.to_csv(out_file, sep=',', encoding='utf-8', index=False, columns=['Class Number'])
            out_file.close()
        # if self.verbose == 2:
        # print("Written to {}/output.csv.".format(self.output))
        return

