import os
import cv2
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transform():
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        ToTensorV2(p=1.0)
    ], additional_targets={'image': 'image', 'label': 'image'})


class CustomDataset(Dataset):

    def __init__(self, df, data_dir, test=False, transform=None):
        self.input = df['LR'].tolist()
        if not test:
            self.target = df['HR'].tolist()
            if transform == None:
                raise ValueError(
                    'Transformer must be determined when train phase. Please Determine Transfomer!!!')
        self.data_dir = data_dir
        self.test = test
        self.transform = transform

    def __getitem__(self, idx):
        input_path = os.path.join(self.data_dir, self.input[idx])
        input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
        if self.test:
            return ToTensorV2(p=1.0)(input_img) / 255.
        else:
            target_path = os.path.join(self.data_dir, self.target[idx])
            target_img = cv2.cvtColor(
                cv2.imread(target_path), cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=input_img, label=target_img)
            return transformed['image'] / 255., transformed['label']/255.

    def __len__(self):
        return len(self.input)
