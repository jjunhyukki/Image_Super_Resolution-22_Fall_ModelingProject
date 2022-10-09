import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.model import CustomSRModel
from data.crop_attach_image import crop_lr_image, attach_image

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    result_dir = os.path.join(os.getcwd(), 'results')
    data_dir = os.path.join(os.getcwd(), 'models', 'data')
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    model = CustomSRModel(device=device,
                          swinir_pretrain=False,
                          edsr_pretrain=False)

    trained_state_dict = torch.load(os.path.join(
        result_dir, 'CustomModel_30.pt'), map_location=device)
    model.load_state_dict(trained_state_dict)

    crop_img = []
    img = []
    for i in tqdm(test['LR']):
        img.append(cv2.cvtColor(cv2.imread(
            i, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))

    for i in img:
        crop_img.append(crop_lr_image(i))

    model.eval()
    c = 20000
    for crop in tqdm(crop_img):
        crop_output_img = []
        for i in crop:
            i = i * 1.0 / 255
            i = torch.from_numpy(np.transpose(
                i, (2, 0, 1))).float().unsqueeze(0)
            i_LR = i.to(device)
            with torch.no_grad():
                output = model(i_LR).data.squeeze().float(
                ).cpu().clamp_(0, 1).permute(1, 2, 0).numpy()
            crop_output_img.append(output)
        attach = (attach_image(512, 512, crop_output_img) * 255.).round()
        output = attach[:, :, [2, 1, 0]]
        cv2.imwrite(os.path.join(result_dir, f'{c}.png'), output)
        c += 1
