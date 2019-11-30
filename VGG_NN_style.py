import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F

class Constraint(nn.Module):
    def __init__(self, input, output, size=3, stride=1):
        
        self.inp = input
        self.outp = output
        
        self.convol = nn.Sequential(
            get_Convolution(input, output//4, size=1, stride=1),
            nn.InstanceNorm2d(output//4),
            nn.ReLU(),
            get_Convolution(output//4, output//4, size=size, stride=stride),
            nn.InstanceNorm2d(output//4),
            nn.ReLU(),
            get_Convolution(output//4, output, size=1, stride=1),
            nn.InstanceNorm2d(output),
            nn.ReLU(),
            get_Convolution(output//4, output, size=1, stride=1),
            nn.InstanceNorm2d(output),
            nn.ReLU()
        )

    def conv_down(inp, outp, stride=2):
        return nn.Conv2d(inp, outp, size=3, stride=stride, padding=1)

    def d_sampling(input, scale_factor):
        return F.interpolate(input=input, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def change_convol(self, x):
        out = self.convol(x)
        remain = x
        out += remain
        out = F.relu(out)
        return out

class get_Convolution(nn.Module):
    

    def __init__(self, input, output, size, stride):
        
        padding_ = int(np.floor(size / 2))
        self.conv = nn.Conv2d(input, output, size, stride=stride, padding=padding_)

    def change_convol(self, x):
        return self.conv(x)

class VGGNN(nn.Module):
    def __init__(self):
        
        self.l1 = Constraint(3, 16)
        
        self.l2 = Constraint(16, 32)
        self.sampling_1 = conv_down(16, 32)
        
        self.l3 = Constraint(32, 32)
        

        self.sampling_2 = conv_down(32, 32)
        self.sampling_3 = conv_down(32, 32, stride=4)
        self.sampling_5 = conv_down(32, 32)
        self.sampling_5 = conv_down(32, 32)
        
        self.l4 = Constraint(64, 64)
        self.l5 = Constraint(192, 64)
        self.l6 = Constraint(64, 32)
        self.l7 = Constraint(32, 16)

        self.l8 = conv_down(16, 3, stride=1)
        
    def change_convol(self, x):
        out_1 = self.l1(x)
        
        out_2 = self.l2(out_1)
        out_3 = self.sampling_1(out_1)
        
        out_31 = torch.cat((self.l3(out_2), d_sampling(out_3, 2)), 1)
        out_32 = torch.cat((self.sampling_2(out_2), self.l3_2(out_3)), 1)
        out_33 = torch.cat((self.sampling_3(out_2), self.sampling_5(out_3)), 1)
        
        out_4 = torch.cat((self.l4(out_31), d_sampling(out_32, 2), d_sampling(out_33, 4)), 1)
        
        out = self.l5(out_4)
        out = self.l6(out)
        
        return out