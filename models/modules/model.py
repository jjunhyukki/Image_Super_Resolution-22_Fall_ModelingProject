import os
import torch
import torch.nn as nn

from .edsr import EDSR
from .swinir import SwinIR

model_dir = os.path.join(os.getcwd(), 'models', 'modules', 'pretrained_models')


class CustomSRModel(nn.Module):
    '''
    Pretrained Model Description
        SwinIR Base 
        : depths = [6, 6, 6, 6, 6, 6]
        : swinir_embed_dim = 180
        : num_heads = [6, 6, 6, 6, 6, 6]
        : upsampler = 'pixelshuffle'

        SwinIR Small
        : depths = [6, 6, 6, 6]
        : swinir_embed_dim = 60
        : num_heads = [6, 6, 6, 6]
        : upsampler = 'pixelshuffledirect'

        EDSR Base
        : n_resblocks = 16
        : edsr_embed_dim = 64

        EDSR Large
        : n_resblocks = 32
        : edsr_embed_dim = 256
    '''

    def __init__(self,
                 device,
                 swinir_pretrain,
                 edsr_pretrain,
                 upscale=4,
                 img_channel=3,
                 img_height=64,
                 img_width=64,
                 window_size=8,
                 img_range=1.,
                 depths=[6, 6, 6, 6],
                 swinir_embed_dim=60,
                 num_heads=[6, 6, 6, 6],
                 mlp_ratio=2,
                 upsampler='pixelshuffledirect',
                 n_resblocks=32,
                 edsr_embed_dim=256):
        #  conv = default_conv):
        super(CustomSRModel, self).__init__()

        self.SwinIR_model = SwinIR(upscale=upscale, img_size=(img_height, img_width),
                                   window_size=window_size, img_range=img_range,
                                   depths=depths, embed_dim=swinir_embed_dim, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, upsampler=upsampler)
        # self.EDSR_model = EDSR(n_resblocks = n_resblocks, n_feats = edsr_embed_dim,
        #                        scale = upscale, rgb_range = 1.,
        #                        res_scale = res_scale, n_colors = img_channel, conv = conv)
        self.EDSR_model = EDSR(scale_factor=4, num_channels=3,
                               num_feats=edsr_embed_dim, num_blocks=n_resblocks, res_scale=1.0)
        if swinir_pretrain:

            if len(depths) == 4:
                swin_pretrained = torch.load(os.path.join(model_dir, '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth'),
                                             map_location=device)['params']
                self.SwinIR_model.load_state_dict(swin_pretrained)
                print(
                    'SwinIR loads state dict of the pretrained model (Small) successfully!!!')

            elif len(depths) == 6:
                swin_pretrained = torch.load(os.path.join(model_dir, '001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'),
                                             map_location=device)['params']
                self.SwinIR_model.load_state_dict(swin_pretrained)
                print(
                    'SwinIR loads state dict of the pretrained model (Base) successfully!!!')

            else:
                raise ValueError(
                    'There is no pretrained model for your hyperparameters!!!')

        if edsr_pretrain:

            if edsr_embed_dim == 256:
                edsr_pretrained = torch.load(os.path.join(model_dir, 'edsr_x4-4f62e9ef.pt'),
                                             map_location=device)
                self.EDSR_model.load_state_dict(edsr_pretrained, strict=False)
                print(
                    'EDSR loads state dict of the pretrained model (Large) successfully!!!')

            elif edsr_embed_dim == 64:
                edsr_pretrained = torch.load(os.path.join(model_dir, 'edsr_baseline_x4-6b446fab.pt'),
                                             map_location=device)
                self.EDSR_model.load_state_dict(edsr_pretrained)
                print(
                    'EDSR loads state dict of the pretrained model (Base) successfully!!!')

            else:
                raise ValueError(
                    'There is no pretrained model for your hyperparameters!!!')

            # self.EDSR_model.sub_mean = MeanShift(rgb_range)
            # self.EDSR_model.add_mean = MeanShift(rgb_range, sign=1)

        self.conv1x1 = nn.Conv2d(in_channels=img_channel * 2,
                                 out_channels=img_channel,
                                 kernel_size=1)

    def forward(self, x):

        swinir_output = self.SwinIR_model(x)
        edsr_output = self.EDSR_model(x)

        concat_output = torch.cat((swinir_output, edsr_output), axis=1)

        output = self.conv1x1(concat_output)

        return output
