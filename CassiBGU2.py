# coding: utf-8

# In[108]:

from __future__ import print_function
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
#from models.downsampler import Downsampler

from utils.sr_utils import *
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smisc

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor


# In[89]:

import h5py
hyperspectral = h5py.File('data/NTIRE/bgu2/hyperspectral.mat', 'r')
items = hyperspectral.items()
for var in items:
    print(var[1])
hs = hyperspectral['rad'].value
hs = hs/np.max(hs)
hs = torch.from_numpy(hs)
hs = hs.unsqueeze(0)
hs = hs.type(dtype)
#hs.shape


# In[90]:

# Simulate CASSI (Spatial-spectral or SS-CASSI, to be specific)
#import numpy as np
#import torch
from scipy.ndimage.interpolation import shift
# Generate random binary mask
height = 1392
width = 1300
inchannels = 31
mask = np.random.choice(2, size=(height, width+inchannels), p=(0.5, 0.5))
plt.imshow(mask)
# Shear mask by different amounts for each channel
middlechannel = int((inchannels - 1)/2)
shifts = [ shift for shift in range(-middlechannel, middlechannel+1) ]
# Apply mask to each channel (element-wise multiplication)
maskblock = np.zeros((inchannels, height, width+inchannels), dtype=np.float32)
for i, shift_val in enumerate(shifts):
#     print(shift_val)
    # Shift mask along horizontal axis only
    # in (C, H, W) order (for pytorch later)
    maskblock[i,:,:] = shift(mask, (0, shift_val), mode='constant')
    middle = (width+inchannels+1)/2
# print(middle)
# print(range(int(middle-width/2),int(middle+width/2)))
maskblock = maskblock[:, :, int(middle-width/2):int(middle+width/2)]
# print(maskblocksquare.shape)
# plt.imshow(maskblocksquare[25,:,:])
maskblock = torch.Tensor(maskblock).type(dtype)
maskblock = torch.autograd.Variable(maskblock, requires_grad=False)
maskblock = maskblock.unsqueeze(0) # Make 4d tensor
#print(maskblock.shape)


# In[91]:

grayscale = maskblock*hs
grayscale = torch.sum(grayscale, dim=1).unsqueeze(0)


# In[92]:

#plt.imshow(grayscale[0,0,:,:].numpy())


# In[93]:

#plt.imshow(hs[0,7,:,:].numpy())


# In[109]:

from models.unet import *
INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'

LR = 0.001
tv_weight = 1.0

OPTIMIZER = 'adam'
input_depth = 32
net = UNet(num_input_channels=1, num_output_channels=31, pad=pad)
net = net.type(dtype)

# Losses
mse = torch.nn.MSELoss().type(dtype)
kl = torch.nn.KLDivLoss().type(dtype)
inputs = torch.zeros(1,1,1024,1024)
inputs[0,:,:,:]=grayscale[:,:,0:1024,0:1024]
#inputs=inputs/torch.max(inputs)
inputs = inputs.cuda()


# In[110]:

def closure():
    global net_input
    out = net(inputs)
    #out = hs
    new_grayscale = maskblock[:,:,0:1024,0:1024]*out[:,:,:1024,:1024]
    new_grayscale = torch.sum(new_grayscale, dim=1).unsqueeze(0)
    total_loss = mse(new_grayscale,inputs) #+ kl(new_grayscale,inputs) + tv_loss(out,1)
    total_loss.backward()
    nsr = mse(new_grayscale,inputs)/mse((torch.zeros(1,1,1024,1024)).cuda(),inputs)
    print('NSR is ', nsr)
    return (total_loss)

#closure()

# In[111]:

LR = 0.0001
parameters = get_params(OPT_OVER, net, inputs)
optimizer = torch.optim.Adam(parameters, lr=LR)
num_iter=100
optimize(OPTIMIZER, parameters, closure,LR , num_iter)


# In[ ]:

LR = 0.0001
num_iter=2500
optimize(OPTIMIZER, parameters, closure,LR , num_iter)

LR = 0.00001
num_iter=0#500
optimize(OPTIMIZER, parameters, closure,LR , num_iter)


out = net(inputs)
out = out.detach()
out = out.cpu()
out = out.numpy()
np.save('CassiBGU2_hyperspectraloutput_notv.npy',out)
torch.save(net,'dipCASSIBGU2_notv.pth')
