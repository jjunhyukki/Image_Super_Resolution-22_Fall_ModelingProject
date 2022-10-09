
from tqdm import tqdm
import streamlit as st
import numpy as np
import torch
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.modules.model import CustomSRModel


def crop_lr_image(img):
    img_h, img_w, _ = img.shape
    crop_lr_img_lst = []

    for i in range(img_h // 64):
        h_start = i * 64
        h_end = (i + 1) * 64
        for j in range(img_w // 64):
            w_start = j * 64
            w_end = (j + 1) * 64
            crop_lr_img_lst.append(img[h_start:h_end, w_start:w_end, :])

    return crop_lr_img_lst


def attach_image(h_, w_, crop_output):
    h_ = 4 * h_
    w_ = 4 * w_
    empty_numpy = np.zeros((h_, w_, 3))
    for i in range(h_ // 256):
        h_start = 256 * i
        h_end = 256 * (i + 1)
        for j in range(w_ // 256):
            w_start = 256 * j
            w_end = 256 * (j + 1)
            empty_numpy[h_start:h_end, w_start:w_end,
                        :] = crop_output[(w_ // 256)*i+j]
    return empty_numpy


@st.cache(persist=True, allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def instantiate_model():
    model_path = 'results/CustomModel_30.pt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = CustomSRModel(
        device=device, swinir_pretrain=True, edsr_pretrain=False)
    model.load_state_dict(torch.load(
        model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)
    print('Model path {:s}. \nModel Loaded successfully...'.format(model_path))
    return model


@st.cache(persist=True, allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def image_super_resolution(uploaded_image, downloaded_image, model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img = cv2.imread(uploaded_image, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    print(h, w)
    if h == 64 and w == 64:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(
            img, (2, 0, 1))).float().unsqueeze(0)
        img_LR = img.to(device)
        model.eval()
        with torch.no_grad():
            output = model(img_LR).data.squeeze(
            ).float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite(downloaded_image, output)
    else:
        resize_h = 64 * (h // 64 + 1)
        resize_w = 64 * (w // 64 + 1)
        img = cv2.resize(img, (resize_w, resize_h),
                         interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop_img = crop_lr_image(img)
        crop_output_img = []
        model.eval()
        for i in tqdm(crop_img):
            i = i * 1.0 / 255
            i = torch.from_numpy(np.transpose(
                i, (2, 0, 1))).float().unsqueeze(0)
            i_LR = i.to(device)
            with torch.no_grad():
                output = model(i_LR).data.squeeze().float(
                ).cpu().clamp_(0, 1).permute(1, 2, 0).numpy()
            crop_output_img.append(output)
        attach = (attach_image(resize_h, resize_w,
                  crop_output_img) * 255.).round()
        output = attach[:, :, [2, 1, 0]]
        output = cv2.resize(output, (w * 4, h * 4),
                            interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(downloaded_image, output)


@st.cache(persist=True, allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def downloaded_success():
    st.balloons()
    st.success('✅ Download Successful !!')
