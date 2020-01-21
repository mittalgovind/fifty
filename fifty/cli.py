"""
FiFTy: File-Fragment Type Classifier using Neural Networks

Usage:
  fifty whatis <input> [-r] [-b BLOCK_SIZE] [-s SCENARIO] [-o OUTPUT] [--block-wise]  [-f] [-l] [-v] [-vv] [-vvv] [-m MODEL_NAME]
  fifty train [-d DATA_DIR] [-b BLOCK_SIZE] [-o OUTPUT] [-a ALGO] [-g GPUS] [-p PERCENT] [-n MAX_EVALS] [--autoencoder] [-f] [--down SCALE_DOWN] [--up] [--paramspace PARAMSPACE_JSON] [--epochs EPOCHS] [-v] [-vv] [-vvv] [-m MODEL_NAME]
  fifty -h | --help
  fifty --version

Options:
  -h --help                                 Show this screen.
  --version                                 Show version.
  -r, --recursive                           Recursively infer all files in folder. [default: False]
  --block-wise                              Do block-by-block classification. [default: False]
  -b BLOCK_SIZE, --block-size BLOCK_SIZE    For inference, valid block sizes --  512 and 4096 bytes. For training, a positive integer. [default: 4096]
  -s SCENARIO, --scenario SCENARIO          Scenario to assume while classifying. Please refer README for more info. [default: 1]
  -o OUTPUT, --output OUTPUT                Output folder. [default: disk_name]
  -f, --force                               Overwrite output folder, if exists. [default: False]
  -l, --light                               Run a lighter version of scenario #1/4096. [default: False]
  -v                                        Controls verbosity of outputs. Multiple v increases it. Maximum is 2. [default: 0]
  -m MODEL_NAME, --model-name MODEL_NAME    During inference, path to an explicit model to use. During training, name of the new model [default: "new_model"].

  -d DATA_DIR, --data-dir DATA_DIR          Path to the FFT-75 data. Please extract to it to a folder before continuing. [default: ./data]
  -a ALGO, --algo ALGO                      Algorithm to use for hyper-parameter optimization (tpe or rand). [default: tpe]
  -g GPUS, --gpus GPUS                      Number of GPUs to use for training (if any). [default: 1]
  -p PERCENT, --percent PERCENT             Percentage of training data to use. [default: 0.1]
  -n MAX_EVALS, --max-evals MAX_EVALS       Number of networks to evaluate. [default: 225]
  --epochs EPOCHS                           Number of epochs to train the final network [default: 10].
  --down SCALE_DOWN                         Path to file with specific filetypes (from our list of 75 filetypes). See utilities/scale_down.txt for reference [default: ../scale_down.txt].
  --up                                      Train with newer filetypes. Please refer documentation. [default: False]
  --paramspace PARAMSPACE_JSON              Path to JSON file with model hyperparameter choices [defaults to "./hparamspace.json", and "./hparamspace_autoencoder.json" if --autoencoder flag is set]. [default: '']
  -t, --autoencoder                         Set to True if you want to train the autoencoder, else train the FiFTy models. [default: False]

"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from __init__ import __version__ as VERSION
from inspect import getmembers, isclass
from docopt import docopt
from pdb import set_trace


def main():
    import fifty.commands as commands
    options = docopt(__doc__, version=VERSION)

    for k, v in options.items():
        if hasattr(commands, k) and v is True:
            module = getattr(commands, k)
            print(f'running command {k}: {module}')
            commands_ = getmembers(module, isclass)
            command = [command[1] for command in commands_ if command[0] in ['Train', 'WhatIs']][0]
            command = command(options)
            command.run()


if __name__ == '__main__':
    main()
