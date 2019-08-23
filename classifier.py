import argparse
import tensorflow as tf
from framework import *
from train_model import train

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(description='FiFTy: File-Fragment Type Classifier using Neural Networks')
classifier_group = parser.add_argument_group('Classification Options')
train_group = parser.add_argument_group('Training Options')

classifier_group.add_argument('-i', '--disk-name', type=str, help='Disk image file')
classifier_group.add_argument('-b', '--block-size', type=int, default=4096,
                              help='Valid block sizes -- 512 and 4096 bytes. (default: 4096)')
classifier_group.add_argument('-s', '--scenario', type=int, default=1,
                              help='Scenario to assume while classifying. \
                                   Please refer README for for more info. (default: 1)')
classifier_group.add_argument('-o', '--output', type=str, help='Output folder. (default: disk_name)')
classifier_group.add_argument('-f', '--force', action='store_true', help='Overwrite output folder, if exists.')
classifier_group.add_argument('-l', '--light', action='store_true', help='Run a lighter version of scenario #1/4096.')
classifier_group.add_argument('-v', action='store_true', help='Outputs class number and class label.')
classifier_group.add_argument('-vv', action='store_true', help='Outputs class number and class label and probability.')
classifier_group.add_argument('-vvv', action='store_true',
                              help='Outputs class number and class label, probability and tag.')
classifier_group.add_argument('-m', '--model-name', type=str, default=None,
                              help='Path to an explicit model to use for inference.')

train_group.add_argument('-t', '--train', action='store_true', help='Train a new model')
train_group.add_argument('-nm', '--new-model', type=str, default='new_model',
                         help='Name of the new model to train. (default: new_model)')
train_group.add_argument('-d', '--data-dir', type=str, default='./data',
                         help='Path to the FFT-75 data. Please extract to ./data before continuing. (default: ./data)')
train_group.add_argument('-a', '--algo', type=str, default='tpe',
                         help='Algorithm to use for hyper-parameter optimization (tpe or rand). (default: tpe)')
train_group.add_argument('-g', '--gpus', type=int, default=1,
                         help='Number of GPUs to use for training (if any). (default: 1)')
train_group.add_argument('-p', '--percent', type=float, default=0.35,
                         help='Percentage of training data to use. (default: 0.35)')
train_group.add_argument('-n', '--max-evals', type=int, default=225,
                         help='Number of networks to evaluate. (default: 225)')
train_group.add_argument('-sd', '--scale-down', type=str,
                         help='Path to file with specific filetypes (from our list of 75 filetypes')
train_group.add_argument('-su', '--scale-up', action='store_true',
                         help='Train with newer filetypes. Please refer documentation.')

args = parser.parse_args()
make_output_folder(args)

if args.train:
    train(args)

model = get_model(args)
blocks = read_disk(args)
pred_probability = infer(model, blocks)
output_predictions(args, pred_probability)
print('Thanks for using!')
