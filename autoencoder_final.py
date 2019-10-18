from __future__ import division, print_function, absolute_import
import glob
import tensorflow as tf 
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smisc
import random
import tensorflow.contrib.layers as lays
import sys
import scipy.io as sio
import h5py
sys.path.append("/home/divya/Desktop/NTIRE18_dataset")
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Import data 

f = glob.glob('/home/divya/Desktop/NTIRE18_dataset/NTIRE2018_Train1_Clean/*.png')
f.sort()
#print(f)
read_img = np.zeros((100,1392,1300,3))
i=0
for img in f[20:21]:
	n= smisc.imread(img)
	plt.imshow(n)
	#n = smisc.imresize(n,(1392,1300,3))
	read_img[i,0:n.shape[0],0:n.shape[1],:]=np.asarray(n)
	#read_img[i]=np.asarray(n)
	print(img)
	i=i+1

fs = glob.glob('/home/divya/Desktop/NTIRE18_dataset/NTIRE2018_Train1_Spectral/*.mat')
fs.sort()
#print(f)
read_spec = np.zeros((100,1392,1300,31))#31,1083, 1392
#print(f)
#print(fs)
i=0
for fmat in fs[20:21]:
	f = h5py.File(fmat,'r')
	print(n.shape)
	n = f['rad'].value
	for j in range(31):	
		read_spec[i,0:n.shape[2],0:n.shape[1],j]=np.transpose(n[j,:,:])
	i=i+1
	print(fmat)


#print(read_img[1].shape)#1392,1300,3
#print(type(read_img))#list
#read_img = np.asarray(read_img)
train_input = []
train_spec = []
for i in range(1):
	for j in range(1):
		#xstart = random.randint(0,1360)
		#ystart = random.randint(0,1268)
		xstart = 10
		ystart = 10
		#n = read_img[i,xstart:xstart+32,ystart:ystart+32,:]
		n = read_img[i,xstart:xstart+256,ystart:ystart+256,:]
		spec = read_spec[i,xstart:xstart+256,ystart:ystart+256,:]
		#print(n.shape)
		train_input.append(n)
		train_spec.append(spec)
#print(type(train_input))

train_input = (train_input - np.min(train_input)) / (np.max(train_input) - np.min(train_input))
train_spec=np.asarray(train_spec)
train_spec = (train_spec - np.min(train_spec)) / (np.max(train_spec) - np.min(train_spec))
print(type(train_spec))

plt.imshow(train_input[0,:,:,:])
plt.show()
plt.imshow(train_spec[0,:,:,0], cmap = 'gray')
plt.show()
#train_input = np.expand_dims(train_input, 0) 
#train_spec = tf.convert_to_tensor(train_spec)

#Autoencoder: http://machinelearninguru.com/deep_learning/tensorflow/neural_networks/autoencoder/autoencoder.html
#batch_size = 5000  # Number of samples in each batch
batch_size=1
epoch_num = 5000     # Number of epochs to train the network
lr = 0.001        # Learning rate

def autoencoder(inputs):
	# encoder
	# 32 x 32 x 3   ->  16 x 16 x 64
	# 16 x 16 x 64  ->  8 x 8 x 40
	# 8 x 8 x 40    ->  2 x 2 x 31
	net = lays.conv2d(inputs, 2, [5, 5], stride=2, padding='SAME')
	net = lays.conv2d(net, 1, [5, 5], stride=2, padding='SAME')
	net = lays.conv2d(net, 1, [5, 5], stride=4, padding='SAME')
	# decoder
	# 2 x 2 x 8    ->  8 x 8 x 16
	# 8 x 8 x 16   ->  16 x 16 x 32
	# 16 x 16 x 32  ->  32 x 32 x 1
	net = lays.conv2d_transpose(net, 12, [5, 5], stride=4, padding='SAME')
	net = lays.conv2d_transpose(net, 16, [5, 5], stride=2, padding='SAME')
	net = lays.conv2d_transpose(net, 31, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
	return net

	# calculate the number of batches per epoch

#ae_inputs = tf.placeholder(tf.float32, (5000, 32, 32, 3))  # input to the network
ae_inputs = tf.placeholder(tf.float32, (1, 256,256, 3))
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network
ground_truth = tf.placeholder(tf.float32, (1,256,256,31))
# calculate the loss and optimize the network

loss = tf.reduce_mean(tf.square(ae_outputs - ground_truth))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config = config) as sess:
	sess.run(init)
	for ep in range(epoch_num):  # epochs loop
		_, c = sess.run([train_op, loss], feed_dict={ae_inputs: train_input, ground_truth: train_spec})
		if ep % 200 == 0:
			print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
		#print(ae_outputs.shape,type(ae_outputs))

	recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: train_input, ground_truth: train_spec})[0]
	plt.figure(1)
	plt.title('Reconstructed Images')
	plt.imshow(recon_img[0,:,:, 0], cmap='gray')
	plt.show()
	#print(recon_img[0,:,:,:])
