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

## File Description
