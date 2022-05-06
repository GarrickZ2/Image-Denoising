# Deep Learning Practice Project (Tentative Title)

## Progress
### Phase 1: Prepare Basic Model
 - [x] Finish PNGAN Model
 - [x] Implement the loss function
 - [x] Finish the RIDNet Model
 - [ ] Finish the training process
 - [x] Train the RIDNet Model
 - [ ] Collect random dataset or use external dataset (find and download)
 - [ ] Download and pre-process the training and testing data
 - [ ] Write the synthesis image generator
 - [ ] Finish the first round training and see the results

### Phase 2: Create a work flow for the whole process
 - [ ] Create a basic data augmentation method
 - [ ] Create a more complex data augmentation method
 - [ ] Create our augmentation method (with trained model)
 - [ ] Create the workflow with limited dataset
 - [ ] Create the workflow with extended dataset

### Phase 3: Finish Experiments and Collect Data
 - [ ] Experiment on 3 methods on limited dataset
 - [ ] Experiment on 3 methods on extended dataset

### Phase 4: Finish an Essay, Readme and run.sh
 - [ ] Abstract
 - [ ] Introduction
 - [ ] Model and Method
 - [ ] Experiments (Dataset, Model Setting, Exp Method, Result)
 - [ ] Conclusion
 - [ ] Readme
 - [ ] Run.sh

# (TODO: reorganize) Evaluate Restormers Denoiser on Real Noisy Data

To establish our baseline, we show the performance of the pre-trained Restormer denoiser on the [SIDD real-noisy dataset](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php). The original evaluation [script](https://github.com/swz30/Restormer/blob/main/Denoising/evaluate_sidd.m) calls for Matlab, but we provided a Python alternative.

To prepare data, download and unzip the SIDD eval dataset.

```
wget https://storage.googleapis.com/yy3185/SIDD_val_patches.zip
unzip -q SIDD_val_patches.zip -d ./Denoising/
```

Then, download the the official pre-trained Restormer [checkpoint](https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0?usp=sharing) and place them in `./Restormer/Denoising/pretrained_models/`
    - Optionally, you can also download the data from shell or notebook using the mirror we provided:

```bash
wget https://storage.googleapis.com/yy3185/real_denoising.pth -O ./Restormer/Denoising/pretrained_models/real_denoising.pth
```

Then you may evaluate using the following script.

```
python evaluate.py -opt ./Denoising/Options/RealDenoising_Restormer.yml --pretrained_weights ./Denoising/pretrained_models/real_denoising.pth
```

 # (TODO: reorganize) Finetune Restormer Denoiser with Real and PNGAN Noisy Images

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

## (TODO: reorganize) Finetune Restormer Denoiser with Real and Gaussian Noisy Images

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
