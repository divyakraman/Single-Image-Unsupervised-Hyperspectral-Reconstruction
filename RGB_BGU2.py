from __future__ import print_function
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from models import *

import torch
import torch.optim

#from skimage.measure import compare_psnr
#from models.downsampler import Downsampler

from utils.sr_utils import *
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smisc

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.DoubleTensor
#dtype = torch.DoubleTensor

import h5py
hyperspectral = h5py.File('hyperspectral2.mat', 'r')
items = hyperspectral.items()
for var in items:
    print(var[1])
hs = hyperspectral['rad'].value
hs = hs/np.max(hs)
hs = torch.from_numpy(hs)
hs = hs.unsqueeze(0)
hs = hs.type(dtype)  

'''
out = hs
reflectances = out[0,:,:,:]#.detach().numpy()
reflectances=reflectances.permute(1,2,0)
#reflectances = reflectances/torch.max(reflectances)
illum_6500 = sio.loadmat('illum_6500.mat', squeeze_me=True)
illum_6500 = illum_6500['illum_6500']
radiances_6500 = torch.zeros(reflectances.shape) #initialize array
illum_6500 = torch.from_numpy(illum_6500)
illum_6500 = illum_6500.cuda()
for i in range(31):
    radiances_6500[:,:,i] = reflectances[:,:,i]*illum_6500[i];
radiances = radiances_6500
radiances = radiances.cuda()
[r,c,w] = radiances.shape
radiances = torch.reshape(radiances, (r*c, w));
xyzbar = sio.loadmat('xyzbar.mat',squeeze_me=True)
xyzbar = xyzbar['xyzbar']
xyzbar = torch.from_numpy(xyzbar)
xyzbar = xyzbar[:31,:]
xyzbar=xyzbar.cuda()
XYZ = torch.t(torch.matmul(torch.t(xyzbar.double()),torch.t(radiances.double())))
XYZ = torch.reshape(XYZ, (r, c, 3))
XYZ[XYZ < 0] = 0
XYZ = XYZ/torch.max(XYZ);
d = XYZ.shape;
r = d[0]*d[1]   # product of sizes of all dimensions except last, wavelength
w = d[2]             # size of last dimension, wavelength
XYZ = torch.reshape(XYZ, (r,w))
#Forward transformation from 1931 CIE XYZ values to sRGB values (Eqn 6 in IEC_61966-2-1.pdf).
M = torch.tensor([[3.2406,-1.5372,-0.4986],
    [-0.9689,1.8758,0.0414],
     [0.0557,-0.2040,1.0570]])
M=M.cuda()
sRGB = torch.t(torch.matmul(M.double(),torch.t(XYZ.double())))
#Reshape to recover shape of original input.
sRGB = torch.reshape(sRGB, d)
sRGB[sRGB<0]=0
sRGB[sRGB>1]=1
'''
from models.unet import *
INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'

tv_weight = 1.0

OPTIMIZER = 'adam'
input_depth = 32
net = UNet(num_input_channels=3, num_output_channels=31, pad=pad)
net = net.type(dtype)
net = torch.load('UNet_NTIRE.pth')

# Losses
mse = torch.nn.MSELoss().type(dtype)
'''
img = sRGB[0:1024,0:1024,:]
inputs = np.zeros((1,3,1024,1024))
inputs[0,:,:,:]=np.transpose(img,(2,0,1))
inputs=inputs/255.0
inputs=torch.from_numpy(inputs)
inputs=inputs.double()
inputs = inputs.type(dtype)
net_input = inputs
'''
inputs = hyperspectral['rgb'].value
inputs = np.transpose(inputs, (1, 2, 0))
inputs = smisc.imresize(inputs,(256,256))
inputs = inputs/np.max(inputs)
inputs = np.transpose(inputs,(2,0,1))
inputs = torch.from_numpy(inputs)
inputs = inputs.unsqueeze(0)
inputs = inputs[:,:,0:256,0:256].type(dtype) 


def closure():
    global i
    out=net(inputs)
    reflectances = out[0,:,:,:]#.detach().numpy()
    reflectances=reflectances.permute(1,2,0)
    reflectances = reflectances/torch.max(reflectances)
    illum_6500 = sio.loadmat('illum_6500.mat', squeeze_me=True)
    illum_6500 = illum_6500['illum_6500']
    radiances_6500 = torch.zeros(reflectances.shape) #initialize array
    illum_6500 = torch.from_numpy(illum_6500)
    illum_6500 = illum_6500.cuda()
    for i in range(31):
      radiances_6500[:,:,i] = reflectances[:,:,i]*illum_6500[i];
    radiances = radiances_6500
    radiances = radiances_6500.cuda()
    [r,c,w] = radiances.shape
    radiances = torch.reshape(radiances, (r*c, w));
    xyzbar = sio.loadmat('xyzbar.mat',squeeze_me=True)
    xyzbar = xyzbar['xyzbar']
    xyzbar = torch.from_numpy(xyzbar)
    xyzbar = xyzbar[:31,:]
    xyzbar=xyzbar.cuda()
    XYZ = torch.t(torch.matmul(torch.t(xyzbar.double()),torch.t(radiances.double())))
    XYZ = torch.reshape(XYZ, (r, c, 3))
    XYZ[XYZ < 0] = 0
    XYZ = XYZ/torch.max(XYZ);
    d = XYZ.shape;
    r = d[0]*d[1]   # product of sizes of all dimensions except last, wavelength
    w = d[2]             # size of last dimension, wavelength
    XYZ = torch.reshape(XYZ, (r,w))
    #Forward transformation from 1931 CIE XYZ values to sRGB values (Eqn 6 in IEC_61966-2-1.pdf).
    M = torch.tensor([[3.2406,-1.5372,-0.4986],
        [-0.9689,1.8758,0.0414],
         [0.0557,-0.2040,1.0570]])
    M=M.cuda()
    sRGB = torch.t(torch.matmul(M.double(),torch.t(XYZ.double())))
    #Reshape to recover shape of original input.
    sRGB = torch.reshape(sRGB, d)
    sRGB[sRGB<0]=0
    sRGB[sRGB>1]=1
    sRGB = sRGB.permute(2,0,1)
    sRGB = sRGB.unsqueeze_(0)
    total_loss = mse(inputs,sRGB)#+tv_loss(sRGB,1)
    total_loss.backward()
    a = torch.zeros(1,3,256,256)
    a = a.cuda()
    a = a.type(dtype)
    nsr = mse(inputs,sRGB)/mse(inputs,a)
    print(nsr)
    return total_loss

LR = 0.001
parameters = get_params(OPT_OVER, net, inputs)
optimizer = torch.optim.Adam(parameters, lr=LR) 
num_iter=1500
optimize(OPTIMIZER, parameters, closure, LR, num_iter)

LR = 0.0001
num_iter=8500#500
optimize(OPTIMIZER, parameters, closure,LR , num_iter)

out = net(inputs)
out = out.detach()
out = out.cpu()
out = out.numpy()
np.save('dipRGB_BGU2_hyperspectraloutput.npy',out)
torch.save(net,'dipRGB_BGU2_hyperspectraloutput.pth')

