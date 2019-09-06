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
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pylab as plt
from matplotlib import rcParams, cycler
import matplotlib as mpl
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
        self.labels, self.tags = load_labels_tags(self.scenario)
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

        gen_files = read_files(self.input, self.block_size, self.recursive)
        if self.verbose >= 2:
            print('Predicting..... Please be patient!')
        try:
            while True:
                file, file_name = next(gen_files)
                pred_probability = self.infer(model, file)
                self.output_predictions(pred_probability, file_name)
                del file, file_name
        except:
            pass
        if self.verbose >= 2:
            print('Prediction Complete!')
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
        if self.verbose >= 2:
            print('Loaded model: {}. \nSummary of model:'.format(self.model_name))
            model.summary()
        return model

    def infer(self, model, blocks):
        """Runs the model on the disk image"""
        tic = time.perf_counter()
        pred_probability = model.predict_proba(blocks)
        toc = time.perf_counter()
        if self.verbose >= 2:
            print('Inference time per sample = {} ms'.format((toc - tic) * 1000 / len(blocks)))
        return pred_probability

    def plot_maps(self, df, file_name):
        cmap = plt.cm.coolwarm
        if self.scenario == 1:
            fig, axes = plt.subplots(nrows=2, ncols=1)

            # plotting class labels
            class_numbers = np.array(df['Class Number'])
            width = int(np.sqrt(len(class_numbers)))
            class_numbers = np.concatenate((class_numbers, 75 * np.ones(width - len(class_numbers) % width))).reshape(
                (-1, width))
            label_dict = dict([(label, i) for i, label in enumerate(self.labels)])
            label_dict['empty'] = len(label_dict)
            rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(label_dict))))
            axes[0].imshow(class_numbers, cmap=cmap)
            # axes[0].legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1))
            axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axes[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            # plotting tags
            tags = df['Tag']
            tags_enum = []
            tags_dict = dict([(tag, i) for i, tag in enumerate(np.unique(self.tags))])
            tags_dict['empty'] = len(tags_dict)
            for tag in tags:
                tags_enum.append(tags_dict[tag])
            tags_enum = np.array(tags_enum)
            tags_enum = np.concatenate((
                tags_enum, (len(tags_dict) - 1) * np.ones(width - len(tags_enum) % width))).reshape((-1, width))
            rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(tags_dict))))
            axes[1].imshow(tags_enum, cmap=cmap)
            # axes[1].legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1))
            axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axes[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            class_numbers = np.array(df['Class Number'])
            label_dict = dict([(label, i) for i, label in enumerate(self.labels)])
            label_dict['empty'] = len(label_dict)
            width = int(np.sqrt(len(class_numbers)))
            class_numbers = np.concatenate(
                (class_numbers, (len(label_dict) - 1) * np.ones(width - len(class_numbers) % width))).reshape((-1, width))
            rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(label_dict))))
            axes[0].imshow(class_numbers, cmap=cmap)
            norm = mpl.colors.BoundaryNorm(self.labels, len(label_dict))
            cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
            set_trace()
            cb = mpl.colorbar.ColorbarBase(axes[1], cmap=cmap, norm=norm, ticks=list(label_dict.values()),
                                           boundaries=list(label_dict.values()), format='%1i', drawedges=True)
            # plt.colorbar()
            # plt.legend()
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        plt.savefig(str(file_name))
        plt.cla()
        return

    def output_predictions(self, pred_probability, file_name):
        """Saves prediction in relevant format to disk"""
        pred_class = np.argmax(pred_probability, axis=1)
        pred_label = [self.labels[i] for i in pred_class]
        pred_probability = np.round(np.max(pred_probability, axis=1) * 100, decimals=1)
        tags = [self.tags[i] for i in pred_class]
        df = pd.DataFrame(
            {
                'Class Number': pred_class,
                'Class Label': pred_label,
                'Class Probability': pred_probability,
                'Tag': tags
            },
            columns=['Class Number', 'Class Label', 'Class Probability', 'Tag']
        )

        if not self.block_wise:
            top_labels = df['Class Label'].value_counts()[:4]
            top_labels *= 100 / top_labels.sum()
            output = '{}: {{'.format(file_name)
            for label, percent in top_labels.items():
                if percent >= 0.05:
                    output += '{}: {:.1f}, '.format(label.upper(), percent)
            print('{}}}'.format(output[:-2]))

        try:
            if self.verbose >= 1 or self.block_wise:
                if '.' in file_name:
                    file_name = '{}/{}'.format(self.output, file_name[:file_name.rfind('.')])
                self.plot_maps(df, '{}.png'.format(file_name))
            if self.verbose >= 2:
                df.to_csv('{}.csv'.format(file_name), sep=',', encoding='utf-8', index=False)
                print("Written to {}.png.".format(file_name))
                print("Written to {}.csv.".format(file_name))
        except:
            pass
        return
