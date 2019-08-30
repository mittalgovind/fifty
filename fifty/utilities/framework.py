import os
import re
import json
import pandas as pd
import numpy as np
import time
import shutil
from pathlib import Path
from pdb import set_trace


def read_file(path, block_size):
    data = open(path, 'rb').read()
    if len(data) < block_size:
        print('Skipping {}. Smaller than one block size ({} bytes). Try smaller block size.'.format(path, block_size))
        return
    bound = (len(data) // block_size) * block_size
    data = data[:bound]
    file = np.array(list(data), dtype=np.uint8).reshape((-1, block_size))
    del data
    return file


def read_files(input, block_size, recursive):
    """Reads the data disk or folder for inference"""
    files = []
    try:
        if os.path.isfile(input):
            file = read_file(input, block_size)
            if file is not None:
                files.append(file)
        elif os.path.exists(input):
            if recursive:
                pattern = '**/*'
            else:
                pattern = './*'
            for path in Path(type_path).glob(pattern):
                if os.path.isfile(input):
                    file = read_file(path, block_size)
                    if file:
                        files.append(file)
    except:
        raise Error("Unable to read {}".format(input))
    return files


def make_output_folder(input, output, force):
    """Prepares output folder"""
    if not output:
        if os.path.isfile(input):
            match = re.match(r"(.*/[A-Za-z0-9_-]+)\..*", input)
            if match:
                output = match.group(1)
            else:
                output = './output'
        else:
            match = re.match(r"(.*/)([A-Za-z0-9_-]+)", input)
            if match:
                output = '{}fifty_{}'.format(match.group(1), match.group(2))
            else:
                output = './output'

    output = os.path.abspath(output)
    if os.path.exists(output):
        if force:
            print("Warning! The output folder is being overwritten.")
            shutil.rmtree(output)
        else:
            raise BlockingIOError("The output folder already exists. Use -f to overwrite it completely.")
    os.mkdir(output)
    return output


def load_labels_tags(scenario):
    """Loads class labels and tags"""
    if os.path.isfile('fifty/utilities/labels.json'):
        with open('fifty/utilities/labels.json') as json_file:
            classes = json.load(json_file)
            labels = classes[str(scenario)]
            tags = classes['tags']
    else:
        raise FileNotFoundError('Please download labels.json to the current directory!')
    return labels, tags

