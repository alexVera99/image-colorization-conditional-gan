# Image colorization GAN
It is a conditional GAN that aims to colorize an image given user guidance.

# Code
In the notebook `Project_Final_Code.ipynb` you may find the code for implementing the network as well as the training and testing.

# Data
The dataset we have used is STL10. You can easily download the dataset using the DataLoader class `MySTL10`. Just make sure that the parameter `download` is set to `True`

# Model
In the folder `Results`you may find the models trained. The best model is called `cgan_stl_10_100ep_60pts.ckpt`.