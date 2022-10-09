import numpy as np


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
