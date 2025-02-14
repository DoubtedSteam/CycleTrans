from __future__ import absolute_import

from torchvision.transforms import *
import torchvision.utils as vutils
#from PIL import Image
import random
import math
import numpy as np
import torch
import cv2
import copy

def mix_up(data):

    probability = 0.5
    min_out = 1e-6
    sl = 0.02
    sh = 0.4
    r1 = 0.3
    out = data.clone()
    B, C, H, W = data.size()
    for img_num in range(B):
        if random.uniform(0,1) > probability:
            continue
        area = H * W
        for attempt in range(100):
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                alpha = random.uniform(0,1)
                x1 = random.randint(0, H - h)
                y1 = random.randint(0, W - w)
                out[img_num, :, x1:x1+h, y1:y1+w] = data[(img_num + B // 2) % B, :, x1:x1+h, y1:y1+w] * alpha + data[img_num, :, x1:x1+h, y1:y1+w] * (1 - alpha)
                break
    return out