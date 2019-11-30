import torch

import numpy as np

from PIL import Image

import torchvision.transforms as transforms


def scaleup(raw_image):

    raw_image = raw_image.transpose(1, 2, 0)

    raw_image = raw_image * np.array((0.380, 0.365, 0.365)) + np.array((0.596, 0.546, 0.506))
    #based on paper
    raw_image = raw_image.clip(0, 1)

    return raw_image


def feature_selection(raw_image, ls=None):
    f_collect = {}
    temp = raw_image
    for id, num in model._modules.items():
        temp = num(temp)
        if id in ls:
            f_collect[ls[id]] = temp
            
    return f_collect

def geti(path, size=None):

    raw_image = Image.open(path)

    if size is not None:
        raw_image = raw_image.resize((size, size))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.596, 0.546, 0.506), (0.380, 0.365, 0.365))#directly taken from paper implemention
    ])
    
    return raw_image


def gm_calculate(raw_image):
    v1, v2, v3, v4 = raw_image.size() #batch channel height width
    raw_image = raw_image.view(v1*v2, v3*v4)
    gm_calculate = torch.mm(raw_image, raw_image.t())
    return gm_calculate