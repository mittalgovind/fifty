![Image Logo](https://github.com/mittalgovind/fifty/blob/master/fifty_logo.png)
# FiFTy: Large-scale File Fragment Type Identification using Neural Networks

FiFTy is a file type classifier that works much like the ``file`` command in Unix-like systems but with much more cool techniques up its sleeve. It beats several previous benchmarks on the biggest play field there is right now.  FiFTy comes with pre-trained models for six scenarios and for block sizes of 512 and 4096 bytes.  It is retrainable for a subset of our studied filetypes and can be scaled up for newer filetypes and other block sizes too. Please find our corresponding paper at https://arxiv.org/abs/1908.06148 and the ready-to-use open access datasets at [FFT-75](https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset).

## Installation
```
pip3 install fifty
```

## List of Filetypes
The classifier has been tested on the following 75 filetypes :--

| | | | | | 
| :---: | :---: | :---: | :---: | :---: |
| ARW | CR2 | DNG | GPR | NEF |
| NRW | ORF | PEF | RAF | RW2 |
| 3FR | JPG | TIFF | HEIC | BMP |
| GIF | PNG | AI | EPS | PSD |
| MOV | MP4 | 3GP | AVI | MKV |
| OGV | WEBM | APK | JAR | MSI |
| DMG | 7Z | BZ2 | DEB | GZ |
| PKG | RAR | RPM | XZ | ZIP |
| EXE | MACH-O | ELF | DLL | DOC |
| DOCX | KEY | PPT | PPTX | XLS |
| XLSX | DJVU | EPUB | MOBI | PDF |
| MD | RTF | TXT | TEX | JSON |
| HTML | XML | LOG | CSV | AIFF |
| FLAC | M4A | MP3 | OGG | WAV |
| WMA | PCAP | TTF | DWG | SQLITE |

## Usage:

```
FiFTy: File-Fragment Type Classifier using Neural Networks
Usage:
  fifty whatis <input> [-r] [-b BLOCK_SIZE] [-s SCENARIO] [--block-wise] [-o OUTPUT] [-f] [-l] [-v] [-vv] [-vvv] [-m MODEL_NAME]
  fifty train [-d DATA_DIR] [-a ALGO] [-g GPUS] [-p PERCENT] [-n MAX_EVALS] [--down SCALE_DOWN] [--up]
  fifty -h | --help
  fifty --version
Options:
  -h --help                                 Show this screen.
  --version                                 Show version.
  -r, --recursive                           Recursively infer all files in folder. [default: False]
  --block-wise                              Do block-by-block classification. [default: False]
  -b BLOCK_SIZE, --block-size BLOCK_SIZE    For inference, valid block sizes --  512 and 4096 bytes. For training, a positive integer. [default: 4096]
  -s SCENARIO, --scenario SCENARIO          Scenario to assume while classifying. Please refer README for more info. [default: 1]
  -o OUTPUT, --output OUTPUT                Output folder. (default: disk_name)
  -f, --force                               Overwrite output folder, if exists. [default: False]
  -l, --light                               Run a lighter version of scenario #1/4096. [default: False]
  -v                                        Controls verbosity of outputs. Multiple v increases it. Maximum is 2. [default: 0]
  -m MODEL_NAME, --model-name MODEL_NAME    During inference, path to an explicit model to use. During training, name of the new model (default: new_model).
  -d DATA_DIR, --data-dir DATA_DIR          Path to the FFT-75 data. Please extract to it to a folder before continuing. [default: ./data]
  -a ALGO, --algo ALGO                      Algorithm to use for hyper-parameter optimization (tpe or rand). [default: tpe]
  -g GPUS, --gpus GPUS                      Number of GPUs to use for training (if any). [default: 1]
  -p PERCENT, --percent PERCENT             Percentage of training data to use. [default: 0.1]
  -n MAX_EVALS, --max-evals MAX_EVALS       Number of networks to evaluate. [default: 225]
  --down SCALE_DOWN                         Path to file with specific filetypes (from our list of 75 filetypes). See utilities/scale_down.txt for reference.
  --up                                      Train with newer filetypes. Please refer documentation. [default: False]

```

## Scenario Description
We present [models](https://github.com/mittalgovind/fifty/tree/master/fifty/utilities/models) for _six_ scenarios on two popular block sizes of __512__ and __4096__ bytes. File type selection reflects focus on media carving applications, where scenarios \#3 to \#6 are the most relevant:

1. **\#1 (All; 75 classes)**: All filetypes are separate classes; this is the most generic case and can be aggregated into more specialized use-cases.

2. **\#2 (Use-specific; 11)**: Filetypes are grouped into 11 classes according to their use; this information may be useful for more-detailed, hierarchical classification or for determining the primary use of an unknown device.

3. **\#3 (Media Carver - Photos \& Videos; 25)**: Every filetype tagged as a bitmap (6), RAW photo (11) or video (7) is considered as a separate class;  all remaining types are grouped into one _other_ class. 

4. **\#4 (Coarse Photo Carver; 5)**: Separate classes for different photographic types: JPEG, 11 raw images, 7 videos, 5 remaining bitmaps are grouped into one separate class per category; all remaining types are grouped into one _other_ class.  

5. **\#5 (Specialized JPEG Carver; 2)**: JPEG is a separate class and the remaining 74 filetypes are grouped into one _other_ class; scenario intended for analyzing disk images from generic devices.

1. **\#6 (Camera-Specialized JPEG Carver; 2)**: JPEG is a separate class and the remaining photographic/video types (11 raw images, 3GP, MOV, MKV, TIFF and HEIC) are grouped into one _other_ class; scenario intended for analyzing SD cards from digital cameras.

## Training your own file classifier

FiFTy is fully scalable. The file-types can be increased and decreased, depending upon a specific use-case. 

### Scaling Down

For scaling down, i.e., choosing a subset of file-types from the above table, you need to specify path to a text file containing each file-type on a new line and path to the FFT-75 dataset. This will sub-sample the specified samples from FFT-75 dataset and save it as a new dataset. You can use this dataset for further experiments. 

A guideline for making this happen is as follows:

- You need to download the [FFT-75 dataset](https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset]).

- Do go through the documentation attached to that page.

- To sub-select filetypes only download the first (most generic) scenario. You can generate other scenarios training data (atleast a smaller version) using the first scenarios dataset.

- When you download the 4k_1.tar.gz or 512_1.tar.gz, extract it to a folder (say, data) and pass the path to this folder only. Example:

    `fifty train --data_dir /Users/<user_name>/Downloads/data --down scale_down.txt`

Again, to make myself clear - the full path to the extracted folder from tar file should be saved inside a folder and that folder should be passed (eg, for the above case - /Users/<user_name>/Downloads/data/4k_1/train.npz) with a text file containing file-types ([scale_down.txt](https://github.com/mittalgovind/fifty/blob/master/scale_down.txt))

- Fifty should have created a dataset for it in the output directory. (If you don't see this, then training would fail.)

- The training will now start automatically on this dataset (please refer the training options to control this further) and save the best optimized model for your case in the output folder.

- Pass this model next time you run `fifty whatis` command on your test files.

### Scaling Up

If you would like to test FiFTy on new filetypes, you would have to prepare your own dataset and specify it. If you are going to use some file-types that are already in the FFT-75 dataset, you might want to scale-down first and get the smaller dataset. Later you can augment that dataset according to your needs. 

For preparing a dataset that is compatible with FiFTy, please follow these steps:

1.  Download atleast 400 MB of files for each of the new file-type.
2.  Sample 102400 blocks from these files and create an ndarray (say, blocks) of shape: (102400, block-size) and labels (say, classes) of shape - (102400) of class_number you choose.
3.  Concatenate all the blocks (as x) and classes (as y) of all file-types and shuffle using `random.shuffle`. 
4.  Save the np.savez_compressed('new_dataset.npz', x=x, y=y).

## Running from source

The following command assumes that the folder structure is as follows:
    
    fifty/
    ├── data/
    │   ├── 512_6/
    ├── output/
    ├── utilities/
    │── cli.py

And assumes scenario 6, and that the dataset is in a folder is `./data/512_6/`     

```sh
python cli.py train --output ./output -d ./data/512_6 --percent 0.00001 --block-size 512 --scenario 6
```

`--percent 0.00001` is used to speed up testing. 
