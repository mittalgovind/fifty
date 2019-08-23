import os
import re
import json
import pandas as pd
import numpy as np
from keras.models import load_model
import time
import shutil


def read_disk(args):
    """Reads the data disk for inference"""
    try:
        data = open(args.disk_name, 'rb').read()
        if len(data) < args.block_size:
            print('File too small for classification.')
        bound = (len(data) // args.block_size) * args.block_size
        data = data[:bound]
        # convert data into an ndarray of shape - (Samples, Block size)
        blocks = np.array(list(data), dtype=np.uint8).reshape((-1, args.block_size))
        del data
    except RuntimeError:
        raise RuntimeError("Unable to read {}".format(args.disk_name))
    return blocks


def make_output_folder(args):
    """Prepares output folder"""
    if not args.output:
        disk_name = os.path.abspath(args.disk_name)
        match = re.match(r"(.*/[A-Za-z0-9_-]+)\..*", disk_name)
        if match:
            args.output = match.group(1)
        else:
            args.output = './output'

    args.output = os.path.abspath(args.output)
    if os.path.exists(args.output):
        if args.force:
            print("Warning! The output folder is being overwritten.")
            shutil.rmtree(args.output)
        else:
            raise BlockingIOError("The output folder already exists. Use -f to overwrite it completely.")
    os.mkdir(args.output)
    return


def load_labels_tags(scenario):
    """Loads class labels and tags"""
    if os.path.isfile('labels.json'):
        with open('labels.json') as json_file:
            classes = json.load(json_file)
            labels = classes[str(scenario)]
            tags = classes['tags']
    else:
        raise FileNotFoundError('Please download labels.json to the current directory!')
    return labels, tags


def get_model(args):
    """Finds and returns a relevant pre-trained model"""
    if args.model_name is not None:
        try:
            if os.path.isfile(os.path.abspath(args.model_name)):
                model = load_model(args.model_name)
            else:
                raise FileNotFoundError('Could not find the specified model!')
        except RuntimeError:
            raise RuntimeError('Could not load the specified model!')
    elif args.new_model is not None:
        try:
            if os.path.isfile(os.path.abspath(args.new_model)):
                model = load_model(args.new_model)
            else:
                raise FileNotFoundError('Could not find the specified model!')
        except RuntimeError:
            raise RuntimeError('Could not load the specified model!')
    else:
        if args.block_size not in [512, 4096]:
            raise ValueError('Invalid block size!')
        if args.scenario not in range(1, 7):
            raise ValueError('Invalid scenario!')
        try:
            args.model_name = '{}_{}.h5'.format(args.block_size, args.scenario)
            if args.light:
                if args.model_name == '4096_1.h5':
                    args.model_name = '4096_lite.h5'
                else:
                    print('Warning! Lighter version of this case is not available. Using the standard version.')
            model = load_model('models/{}'.format(args.model_name))
        except RuntimeError:
            raise RuntimeError('Model unavailable for block size of {} bytes and scenario {}.'.format(args.block_size,
                                                                                                      args.scenario))
    print('Loaded model: {}. \nSummary of model:'.format(args.model_name))
    model.summary()
    return model


def infer(model, blocks):
    """Runs the model on the disk image"""
    print('Predicting..... Please be patient!')
    tic = time.perf_counter()
    pred_probability = model.predict_proba(blocks)
    toc = time.perf_counter()
    print('Inference time per sample = {} ms'.format((toc - tic) * 1000 / len(blocks)))
    print('Prediction complete!')
    return pred_probability


def output_predictions(args, pred_probability):
    """Saves prediction in relevant format to disk"""
    labels, tags = load_labels_tags(args.scenario)
    pred_class = np.argmax(pred_probability, axis=1)
    pred_label = [labels[i] for i in pred_class]
    pred_probability = np.round(np.max(pred_probability, axis=1) * 100, decimals=1)
    tags = [tags[i] for i in pred_class]
    df = pd.DataFrame(
        {'Class Number': pred_class, 'Class Label': pred_label,
         'Class Probability': pred_probability, 'Tag': tags},
        columns=['Class Number', 'Class Label', 'Class Probability', 'Tag'])
    out_file = open(os.path.join(args.output, 'output.csv'), 'w')

    if args.v:
        df.to_csv(out_file, sep=',', encoding='utf-8', index=False, columns=['Class Number', 'Class Label'])
    elif args.vv:
        df.to_csv(out_file, sep=',', encoding='utf-8', index=False,
                  columns=['Class Number', 'Class Label', 'Class Probability'])
    elif args.vvv:
        if args.scenario == 1:
            df.to_csv(out_file, sep=',', encoding='utf-8', index=False)
        else:
            print('The output of this scenario will not contain tags.')
            df.to_csv(out_file, sep=',', encoding='utf-8', index=False,
                      columns=['Class Number', 'Class Label', 'Class Probability'])
    else:
        df.to_csv(out_file, sep=',', encoding='utf-8', index=False, columns=['Class Number'])
    out_file.close()
    print("Written to {}/output.csv.".format(args.output))
    return
