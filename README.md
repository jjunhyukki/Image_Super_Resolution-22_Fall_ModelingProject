# Image Super-Resolution using EDSR & SwinIR

***
Collaborators : 최명헌, 장준혁, 최윤서, 황진우, 조보경
***

Through the Dacon contest dataset, we implemented Image Super-Resolution with our custom model using EDSR and SwinIR as streamlit app.

## Overview

### Dataset
* Train data : 1640 LR(512×512) & HR(2048×2048) paired images
* Test data : 18 LR images
* Dacon contest link : [AI 양재 허브 인공지능 오픈소스 경진대회](https://dacon.io/competitions/official/235977/overview/description)

### Data Preprocessing
We cut the images into 64 patches regardless of resolution and made upscaling model(64×64 ➡ 256×256)

### Custom Model
<img width="1009" alt="custom_model" src="https://user-images.githubusercontent.com/108822253/194754445-b9e1445f-b1a5-4ccd-a377-78b42404e5b7.png">


## Streamlit

### Installation
* Run the command ___pip install -r requirements.txt___ to install requirements

### Usage
1. Fork this repository and install the requirements as mentioned above
2. Run super_resolution.py with streamlit
```
streamlit run super_resolution.py
```
3. Upload your low-resolution image and get high-resolution image

### Results
![example](https://user-images.githubusercontent.com/108822253/194754773-d34beee2-4ef7-48ba-87a6-a2c5565fce82.png)

## File Description

### models
1. data
* crop_attach_image.py : Crop a low-resolution image to 64 patches and attach upscaling images to restore
* customTrain.csv / customValid.csv / test.csv : CSV files which are the lists of train images, valid images and test images

2. modules
* pretrained models : SwinIR pretrained models
* dataset.py : Transform images and build custom dataset
* edsr.py : Customized EDSR model
* swinir.py : Customized SwinIR model
* losses.py : For L1 loss in train process
* model.py : Our own made super-resolution model code

3.others
* inference_dacon.py : Code of super resolution for test set
* main.py : The whole process from building dataset to super resolution
* train.py : Train process code

### results
* CustomModel_30.pt : pretrained custom super resolution model

### streamlit
* app_funcs.py : Functions that are used in streamlit app
* super_resolution.py : Code to run streamlit app
* uploads : test images

### others
* requirements.txt : Required dependencies to run the streamlit app
