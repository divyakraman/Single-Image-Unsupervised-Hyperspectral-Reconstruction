# Unsupervised Single Image Hyperspectral Reconstruction 
Abstract: Reconstruction of the hyperspectral image from a compressively sensed image or an RGB image is a challenging task. The existing supervised regression methods have been shown to perform well in solving such ill-posed inverse problems as compared to traditional iterative methods, but their performance is data dependent and the reconstructed results exhibit inconsistent quality for different datasets. To overcome the drawbacks of supervised learning for hyperspectral reconstruction, we propose an unsupervised deep learning based pipeline that can reconstruct the hyperspectral image from a RGB image or compressively sensed (CASSI) data by solving an inverse optimization problem. We exploit the fact that the hyperspectral to RGB or CASSI image formation model is differentiable and can be used in a gradient-based learning algorithm. We utilize the imaging model to directly reconstruct the hyperspectral image while testing by optimizing in the neural network weight space. Our method is based on deep image prior and since it does not require a training step, it is robust to the problem of domain-shift. We show that even with a primitive network (UNet), our method competes well with much complex architectures that other supervised methods use.

The paper can be found at 'Unsupervised Single Image Hyperspectral Reconstruction.pdf'.

This repository contains the code for the project. The code is heavily borrowed from https://github.com/DmitryUlyanov/deep-image-prior. 
