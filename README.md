# CS7150 Image Captioning Project
### Carter Ithier and Priyanshu Ranka

## Table of Contents
- [Intro and context](#intro-and-context)
- [Folder structure and files](#folder-structure-and-files)
- [Running the demo](#running-the-demo)

## Intro and context
This repo is for Spring 2026's CS 7150: Deep Learning class. Our project consisted of doing image captioning. Explicitly, we created deep learning models where you feed in one or more images, and the model output is an English caption for each image.

We evaluated 3 different models. The models were a baseline model which consisted of ResNet encoder with a LSTM decoder (see folder ```baseline```), ResNet encoder with a transformer decoder (see folder ```resnet_transformer_decoder```), and a vision transformer (ViT) encoder with a transformer decoder (see folder ```ViT_decoder```).

Refer to final report for more details.

## Folder structure and files
### Main model folders
Each of the above mentioned folders for each model has two files:
1) A model file in pytorch defining the architecture
2) An example training file that can be used to train the model from scratch or from a checkpoint

### Failed architecture folders
We also have a folder for failed architectures. Each subfolder (of which there are four) have model and train files for models we experimented with and had very poor results so we abandoned it. Below is a short description for each:
- ```diy_decoder_only``` -> This is a transformer decoder only architecture. It was created with low level Pytorch layers (Linear, Embedding, etc) and was based off of modified homework assignments. It had no pretrained weights. We were training from scratch
- ```diy_transformer_enc_dec``` -> This is a vision transformer encoder with a transformer decoder. It was created with low level Pytorch layers like the one above based off homeworks. It did not use pretrained weights. We were training it from scratch
- ```pytorch decoder_only``` -> This is a transformer decoder only architecture but uses higher level pytorch layers like TransformerEncoderLayer and TransformerEncoder. We trained this from scratch.
- ```pytroch_transformer_enc_dec```-> This is a vision transformer encoder with a transformer decoder. It was created using higher level pytorch layers like TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder. We trained this from scratch.

The fact we trained all of these from scratch on a small dataset (6k unique train images) is likely why we were having such poor results.

### Misc files in root
```positional_encodings.py``` is a helper class used in multiple models.

The ```evaluation_scripts``` folder has our one eval script that can be used for all models to calculate BLEU, CIDER, and METOER metrics.

```RunningBaselineTraining.md``` has instructions we used of how to run on the cluster.

```baseline_training.sh``` is an example sh file we used for training on the explorer cluster

```dataloader_v2.py``` creates train, validation, and test dataloaders as well as populates a vocabulary object based on the train set. This assumes you have the Flickr8k dataset downloaded in the repo. Flickr8k can be downloaded [here](https://www.kaggle.com/datasets/adityajn105/flickr8k).

```training_helpers.py``` is a file that has helper functions we used during training.

```download_models.py``` allows you to download our best models for each of our three categories.

```demo.py``` Is a file that allows you to do inference on a subset of images with each model and see results

## Running the demo
### Step 1 - Download models
To download the models simply run ```python download_models.py```
This will create a ```models`` folder in the root directory if it doesn't already exist, and download any models specified in the file that you don't have downloaded already.

### Step 2 - Run demo.py
TODO