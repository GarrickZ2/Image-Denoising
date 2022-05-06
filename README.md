# Improving Image Denoising Performance by Data Augmentation Method

COMS 6998 Practical Deep Learning Systems Performance

Project Member: Yifan Yang () / Zixuan Zhang (zz2888)

Affliation: Columbia Univeristy



## Introduction



### Table of Content

## Example and Results



## Environment Configuration

#### Install Gdrive

Important to downlaod our pretrained model, training and testing dataset.

We provide the script to install gdrive on Linux playform.

```bash
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xf gdrive_2.1.1_linux_386.tar.gz
mv gdrive /usr/bin
rm gdrive_2.1.1_linux_386.tar.gz
gdrive about
```

We also provide automatically running make file (only valid for Liunx System), please run

```bash
make gdrive
```

After running the command gdrive about, you need to enter a link and finish the authentication to login a google account, so that you can use google drive services.

### Install Python Environment





## Quick Start

### Download Pretrained Model

#### Download RIDNet Pretrained Model

#### Download Our PNGAN Pretarined Model

#### Download Our Fintuned Restormer Model

### Upload Test Images

Upload your test images (for noise generating or denoising) into the ./test directory. Resutls will saved at the ./save directory.

### Generate Test Results

#### Generate the noise images

```bash
make test_pngan_with_best load_dir="../experiment/PNGAN"
```

The result will be saved at ./save/noise_gene directory.
#### Generate the denoise images



## Dataset Preparation

### Download SIDD Dataset

Download SIDD Dataset into ./Downloads directory

```bash
make download_data
```

For other interests dataset downloads:

```bash
python PNGAN/util/download_data.py
	--data [str: select from 'train', 'test' or 'train-test']
	--dataset [str: select from 'SIDD' or 'DND']
	--noise	[str: select from 'real' or 'gaussian']
```

### Preprocessed Dataset

Cut images into patches with 128*128 sizes

```bash
make preprocessed
```

Crop images with other patch size, overlap size or step size:

```bash
python PNGAN/util/generate_patches_sidd.py # for sidd train dataset
	--size [int: patch size]
	--overlap [int: overlap size between patches]
	--padding [int: padding images with size]
	
python PNGAN/util/generate_patches_sidd_val.py # for sidd validation dataset
	--size [int: patch size]
	--overlap [int: overlap size between patches]
	--padding [int: padding images with size]
```

## Train on PNGAN

### Train from Scratch

```bash
make train_pngan
```

### Train with pretrained model

Please download the pretrained model firstly and save it into the directory ./experiment/PNGAN/

```bash
make train_with_best load_dir="../experiment/PNGAN"
```

### Train with Options

```bash
cd PNGAN
python main.py 
	--dir_data [str: Root Dataset Path] 
	--partial_data # Only use part training data
	--n_train [int: num of partial training data] --n_val [int: num of partial val data]
	--patch_size [int: default 128, image pathc used for training]
	--noise [int: generated guassian noise image deviation]
	--n_feats [int: hidden layer channels number used for generator]
	--epochs [int: training epochs]
	--batch_size [int: trianing batch size]
	--lr [float: default 2e-4, initial learning rate]
	--lr_min [float: minimal learning rate with lr scheduler]
	--lr_deacy_step [int: steps to drop lr from lr to lr_min]
	--load_models # load models to train
	--load_best # load the best model to train
	--load_dir [str: directory you download pretrained model]
	--load_epoch [int: provide the epoch num if you hope load from certain epoch]
	--save_models # saved the model after training
	--save [str: places you store the trained model]
```

## Test on PNGAN

### Test with pretrained model

Before you test with our pretrained model, place upload your images into ./test directory. Please with 3 channel images. The output results will be saved at ./save directory.

```bash
make test_pngan_with_best load_dir="../experiment/PNGAN"
```

### Generated Fintuned Dataset for Restormer

Generate the augemented dataset used for finetune restormer or other denoising model. The dataset will be stored into two directories: 

```bash
make generate_finetune load_dir="../experiment/PNGAN" test_path='generate data path' save_path='save data path'
```

### Test with Options

```bash
cd PNGAN
python main.py
	--test_only # Used for test
	--dir_data [str: Root Dataset Path] 
	--partial_data # Only use part training data
	--n_train [int: num of partial training data] --n_val [int: num of partial val data]
	--patch_size [int: default 128, image pathc used for training]
	--noise [int: generated guassian noise image deviation]
	--n_feats [int: hidden layer channels number used for generator]
	--epochs [int: training epochs]
	--batch_size [int: trianing batch size]
	--lr [float: default 2e-4, initial learning rate]
	--lr_min [float: minimal learning rate with lr scheduler]
	--lr_deacy_step [int: steps to drop lr from lr to lr_min]
	--load_models # load models to train
	--load_best # load the best model to train
	--load_dir [str: directory you download pretrained model]
	--load_epoch [int: provide the epoch num if you hope load from certain epoch]
```



## FineTune Restormer

### (TODO: reorganize) Finetune Restormer Denoiser with Real and PNGAN Noisy Images

First, follow the [installation instruction](https://github.com/swz30/Restormer/blob/main/INSTALL.md) provided by the official Restormer authors to install the depenedencies required to run Restormer.



Download the official pre-trained Restormer [models](https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0?usp=sharing) and place them in `./Restormer/Denoising/pretrained_models/`. Optionally, you can also download the data from shell or notebook using the mirror we provided:

```bash
wget https://storage.googleapis.com/yy3185/real_denoising.pth -O ./Restormer/Denoising/pretrained_models/real_denoising.pth
```



Download the training and validation data. We are using 128x128 RGB images.

```bash
# These commands, along with the following commands, must run in the Restormer directory
cd ./Restormer/

# Download and unzip real-noisy and fake-noisy training data, and real-noisy validation data
wget https://storage.googleapis.com/yy3185/SIDD_train_patches.zip
unzip -q SIDD_train_patches.zip -d ./Denoising/
wget https://storage.googleapis.com/yy3185/SIDD_val_patches.zip
unzip -q SIDD_val_patches.zip -d ./Denoising/
wget https://storage.googleapis.com/yy3185/PNGAN_train.zip
unzip -q PNGAN_train.zip -d ./Denoising/

# Copy all real-noisy and fake-noisy training data to one folder
mkdir -p ./Denoising/Datasets/train/PNGAN+SIDD/input_crops/
mkdir -p ./Denoising/Datasets/train/PNGAN+SIDD/target_crops/
cp -r ./Denoising/Datasets/train/PNGAN/input_crops/ ./Denoising/Datasets/train/PNGAN+SIDD/input_crops/
cp -r ./Denoising/Datasets/train/SIDD/input_crops/ ./Denoising/Datasets/train/PNGAN+SIDD/input_crops/
cp -r ./Denoising/Datasets/train/PNGAN/target_crops/ ./Denoising/Datasets/train/PNGAN+SIDD/target_crops/
cp -r ./Denoising/Datasets/train/SIDD/target_crops/ ./Denoising/Datasets/train/PNGAN+SIDD/target_crops/
```



Run the training command to start the training process, please note that most of the configuration parameters are specified in the configuration YAML file (e.g., number of steps, learning rate, batch size, metrics, etc.) so you may read the YAML file to find more about them. You may also change them to run properly in your compute (e.g., configure the number of GPUs).

```bash
python train.py -opt ./Denoising/Options/PNGANRealDenoising_Restormer.yml --pretrained_weights ./Denoising/pretrained_models/real_denoising.pth
```



Then you may evaluate using the following script.

```bash
python evaluate.py -opt ./Denoising/Options/PNGANRealDenoising_Restormer.yml --pretrained_weights ./experiment/PNGANRealDenoising_Restormer/models/net_g_latest.pth
```

### (TODO: reorganize) Finetune Restormer Denoiser with Real and Gaussian Noisy Images

As part of our ablation studies, we explore finetuning Restormer on the task of real noise denoising with data augmentation using simple Gaussian noise (as opposed to noise modeled by PNGAN). We generated additional clean-noise image pairs for training, by applying Gaussian noise ($\sigma=15$) on the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) high-resolution image dataset.

```bash
# Download the additional training data
wget https://storage.googleapis.com/yy3185/Gaussian_train.zip
unzip -q Gaussian_train.zip -d ./Denoising/

# Copy the data to the training folder
mkdir -p ./Denoising/Datasets/train/Gaussian+SIDD/input_crops/
mkdir -p ./Denoising/Datasets/train/Gaussian+SIDD/target_crops/
cp -r ./Denoising/Datasets/train/Gaussian/input_crops/ ./Denoising/Datasets/train/Gaussian+SIDD/input_crops/
cp -r ./Denoising/Datasets/train/SIDD/input_crops/ ./Denoising/Datasets/train/Gaussian+SIDD/input_crops/
cp -r ./Denoising/Datasets/train/Gaussian/target_crops/ ./Denoising/Datasets/train/Gaussian+SIDD/target_crops/
cp -r ./Denoising/Datasets/train/SIDD/target_crops/ ./Denoising/Datasets/train/Gaussian+SIDD/target_crops/
```



Run the training command as follows:

```bash
python train.py -opt ./Denoising/Options/GaussianRealDenoising_Restormer.yml --pretrained_weights ./Denoising/pretrained_models/real_denoising.pth
```



Evaluate using the following command:

```bash
python evaluate.py -opt ./Denoising/Options/GaussianRealDenoising_Restormer.yml --pretrained_weights ./experiment/GaussianRealDenoising_Restormer/models/net_g_latest.pth
```

## Test Fintuned Restormer

To establish our baseline, we show the performance of the pre-trained Restormer denoiser on the [SIDD real-noisy dataset](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php). The original evaluation [script](https://github.com/swz30/Restormer/blob/main/Denoising/evaluate_sidd.m) calls for Matlab, but we provided a Python alternative.

To prepare data, download and unzip the SIDD eval dataset.

```bash
wget https://storage.googleapis.com/yy3185/SIDD_val_patches.zip
unzip -q SIDD_val_patches.zip -d ./Denoising/
```

Then, download the the official pre-trained Restormer [checkpoint](https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0?usp=sharing) and place them in `./Restormer/Denoising/pretrained_models/`
    - Optionally, you can also download the data from shell or notebook using the mirror we provided:

```bash
wget https://storage.googleapis.com/yy3185/real_denoising.pth -O ./Restormer/Denoising/pretrained_models/real_denoising.pth
```

Then you may evaluate using the following script.

```bash
python evaluate.py -opt ./Denoising/Options/RealDenoising_Restormer.yml --pretrained_weights ./Denoising/pretrained_models/real_denoising.pth
```

## Contribution

## References