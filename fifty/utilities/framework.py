import os
import re
import json
import pandas as pd
import numpy as np
import time
import shutil
from pathlib import Path
from pdb import set_trace


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
    name_pattern = re.compile(r".*/(.+\..*)")
    if os.path.isfile(input):
        file_block = read_file(input, block_size)
        if file_block is not None:
            yield file_block, input
    elif os.path.exists(input):
        if recursive:
            pattern = '**/*'
        else:
            pattern = './*'
        for path in Path(input).glob(pattern):
            if os.path.isfile(path):
                file_block = read_file(path, block_size)
                if file_block is not None:
                    try:
                        file_name = name_pattern.match(str(path)).group(1)
                    except:
                        file_name = 'non-alphanumeric-name.xyz'
                    yield file_block, file_name
    else:
        raise FileNotFoundError('Could not find {}'.format(input))


def make_output_folder(input, output, force):
    """Prepares output folder"""
    if not output:
        if os.path.isfile(input):
            match = re.match(r".*/(.+)i", input)
            if match:
                output = match.group(1)
            else:
                output = './output'
        else:
            match = re.match(r"(.*/)(.+)", input)
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
    if os.path.isfile(os.path.join(get_utilities_dir(), 'labels.json')):
        with open(os.path.join(get_utilities_dir(), 'labels.json')) as json_file:
            classes = json.load(json_file)
            labels = classes[str(scenario)]
            tags = classes['tags']
    else:
        raise FileNotFoundError('Please download labels.json to {} directory!'.format(get_utilities_dir()))
    return labels, tags

