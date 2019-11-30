import torch

import torch.nn.functional as F
 
from config import *

import os
from PIL import Image
import numpy as np 

import torch.nn as nn
import matplotlib.pyplot as plt



from tqdm import tqdm
from torchvision.models import features_v19
from features_vNN import features_vNN

from utils import *
import argparse

    
wt_stls = {
    'c1': 0.2,
    'c2': 0.25,
    'c3': 0.34,
    'c4': 0.76,
    'c5': 2.6,
}

learning_par_init = {
    'lr': 5e-3,
    'step': 200,
    'epochs': 1000,
    'after_epochs': 100,
    'gamma_val': 0.9
}


def training_(args, device, learning_par_init=learning_par_init):
    cnt_l = []
    features_v = features_v19(pretrained=True).features
    wt_cnt = args.wt_cnt
    
    vgg_nn_ = features_vNN()
    
    
    image_get = geti(os.path.join(args.raw_image_root, args.image_get), size=args.content_size)
    image_get_feat = feature_selection(image_get, features_v)

    image_get_stl = geti(os.path.join(args.raw_image_root, args.image_get_stl))
    image_get_stl_feat = feature_selection(image_get_stl, features_v)
    
    wt_stl = args.wt_stl
    fnl_image_raw = image_get.clone().requires_grad_(True)

    stl_l = []
    gm_stl = {l: gm_calculate(image_get_stl_feat[l]) for l in image_get_stl_feat}
    
    fin_l = [] #loss fun
    
    fin_img = image_get

    Ad_optimizer = torch.optim.Adam(vgg_nn_.valueseters(), lr=learning_par_init['lr'])
    after_epoch = torch.optim.StepLR(Ad_optimizer, step=learning_par_init['step'], gamma_val=learning_par_init['gamma_val'])
    
    

    
    for epoch in tqdm(range(learning_par_init['epochs']+1)):

        ls_stl = 0
        after_epoch.step()

        sel_feat = feature_selection(fnl_image_raw, features_v)

        for l in wt_stls:
            fnl_image_raw_feature = sel_feat[l]
            fin_gm = gm_calculate(fnl_image_raw_feature)
            stl_gm = gm_stl[l]
            
            l_ls_stl = wt_stls[l] * torch.mean((fin_gm - stl_gm) ** 2)
            v1, v2, v3, v4 = fnl_image_raw_feature.shape
            ls_stl += l_ls_stl / (v2*v3*v4)
            
        fin_ls = wt_cnt * l_cnt + wt_stl * ls_stl

        fin_l.append(fin_ls.item())
        fin_ls.backward()

        stl_l.append(wt_stl * ls_stl)
        cnt_l.append(wt_cnt * l_cnt.item())
        
        Ad_optimizer.zero_grad()
        Ad_optimizer.step()
        
        if epoch % learning_par_init['after_epochs'] == 0:
            print('SL: ', ls_stl.item())
            print('CL: ', l_cnt.item())
            print('TL ', fin_ls.item())
            plt.imshow(scaleup(fnl_image_raw))
            plt.show()

        fin_img = fnl_image_raw

    

    
if __name__ == "__main__":

    #used from paper implementation. Use of cuda is faster
    device = torch.device('cpu')

    training_(args, device)