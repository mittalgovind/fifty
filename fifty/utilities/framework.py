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
        return None
    bound = (len(data) // block_size) * block_size
    data = data[:bound]
    file = np.array(list(data), dtype=np.uint8).reshape((-1, block_size))
    del data
    return file


def read_files(input, block_size, recursive):
    """Reads the data disk or folder for inference"""
    blocks = []
    file_names = []
    name_pattern = re.compile(r".*/(.+\..*)")
    try:
        if os.path.isfile(input):
            file_blocks = read_file(input, block_size)
            if file_blocks is not None:
                blocks.append(file_blocks)
                file_names.append(input)
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
                        blocks.append(file_block)
                        file_names.append(file_name)
    except:
        raise RuntimeError("Unable to read {}".format(input))
    return blocks, file_names


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
    if os.path.isfile('fifty/utilities/labels.json'):
        with open('fifty/utilities/labels.json') as json_file:
            classes = json.load(json_file)
            labels = classes[str(scenario)]
            tags = classes['tags']
    else:
        raise FileNotFoundError('Please download labels.json to the current directory!')
    return labels, tags

