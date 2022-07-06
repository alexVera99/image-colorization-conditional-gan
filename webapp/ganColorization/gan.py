import torch 
import torch.nn as nn
# Convolution + BatchNormnalization + ReLU or Tanh block for the encoder
class ConvBNReLU(nn.Module):
  def __init__(self,in_channels, out_channels, activation: str = "relu"):
    super(ConvBNReLU, self).__init__()
    self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,
                          padding = 1)
    self.bn = nn.BatchNorm2d(out_channels)

    if activation.lower() == "relu":
      self.activation = nn.ReLU(inplace=True)
    elif activation.lower() == "tanh":
      self.activation = nn.Tanh()
    else:
      raise Exception("Activations available are: relu or tanh")

  def forward(self,x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.activation(out)   
    return out


# Downsampling Convolution + BatchNormnalization + ReLU block for the encoder
class DownsamplinConvBNReLU(nn.Module):
  def __init__(self,in_channels, out_channels):
    super(DownsamplinConvBNReLU, self).__init__()
    self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,
                          stride= 2, padding = 1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self,x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)   
    return out

# Upsampling Convolution (Deconvolution) + BatchNormnalization + ReLU block for the encoder
class UpsamplinConvBNReLU(nn.Module):
  def __init__(self,in_channels, out_channels):
    super(UpsamplinConvBNReLU, self).__init__()
    self.deconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,
                          stride= 2, padding = 1, output_padding = 1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self,x):
    out = self.deconv(x)
    out = self.bn(out)
    out = self.relu(out)   
    return out


# Residual blocks + BatchNormnalization + ReLU block for the encoder
# it keeps the same number of channels
class ResidualBlockBNReLU(nn.Module):
  def __init__(self,channels):
    super(ResidualBlockBNReLU, self).__init__()

    self.conv1 = ConvBNReLU(channels, channels, activation = "relu")
    self.conv2 = ConvBNReLU(channels, channels, activation = "relu")

  def forward(self,x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = out + x # Residual connection
    return out 

class Generator(nn.Module):
  def __init__(self, img_channels = 4, base_channels=64, out_channels = 3):
    super(Generator, self).__init__()
    #self.base_channels = base_channels
    #self.in_features = img_channels
    #self.decoder = Decoder(in_features,base_channels)
    # First convolutional block
    self.first_conv = ConvBNReLU(img_channels, base_channels)

    # Downsampling blocks
    self.down_conv1 = DownsamplinConvBNReLU(base_channels, base_channels*2)
    self.down_conv2 = DownsamplinConvBNReLU(base_channels*2, base_channels*4)

    # Residual blocks
    self.res_block1 = ResidualBlockBNReLU(base_channels*4)
    self.res_block2 = ResidualBlockBNReLU(base_channels*4)
    self.res_block3 = ResidualBlockBNReLU(base_channels*4)
    self.res_block4 = ResidualBlockBNReLU(base_channels*4)
    self.res_block5 = ResidualBlockBNReLU(base_channels*4)
    self.res_block6 = ResidualBlockBNReLU(base_channels*4)
    self.res_block7 = ResidualBlockBNReLU(base_channels*4)
    self.res_block8 = ResidualBlockBNReLU(base_channels*4)
    self.res_block9 = ResidualBlockBNReLU(base_channels*4) 

    # Upsampling blocks
    self.up_conv1 = UpsamplinConvBNReLU(base_channels*4, base_channels*2)
    self.up_conv2 = UpsamplinConvBNReLU(base_channels*2, base_channels)

    # Last 
    self.last_conv = ConvBNReLU(base_channels, out_channels, activation = "tanh")

  def forward(self, x):
    out = self.first_conv(x)

    # Downsampling
    out = self.down_conv1(out)
    out = self.down_conv2(out)

    # Residual blocks
    out = self.res_block1(out)
    out = self.res_block2(out)
    out = self.res_block3(out)
    out = self.res_block4(out)
    out = self.res_block5(out)
    out = self.res_block6(out)
    out = self.res_block7(out)
    out = self.res_block8(out)
    out = self.res_block9(out)

    # Upsampling blocks
    out = self.up_conv1(out)
    out = self.up_conv2(out)

    # Last conv
    out = self.last_conv(out)

    return out
  
class Discriminator(nn.Module):
  def __init__(self, input_channels = 6, base_channels=64):
    super(Discriminator, self).__init__()
       
    # First Downsampling blocks
    self.conv1_1 = ConvBNReLU(input_channels, base_channels)
    self.down_conv1_1 = DownsamplinConvBNReLU(base_channels, base_channels*2)
    self.down_conv1_2 = DownsamplinConvBNReLU(base_channels*2, base_channels*4)
    self.down_conv1_3 = DownsamplinConvBNReLU(base_channels*4, base_channels*8)

    # Mean pool for multiscale discrimination
    self.av_pool = nn.AvgPool2d(kernel_size = 2, stride = 2)

    # Second Downsampling blocks
    self.conv2_1 = ConvBNReLU(input_channels, base_channels)
    self.down_conv2_1 = DownsamplinConvBNReLU(base_channels, base_channels*2)
    self.down_conv2_2 = DownsamplinConvBNReLU(base_channels*2, base_channels*4)
    self.down_conv2_3 = DownsamplinConvBNReLU(base_channels*4, base_channels*8)

    # Sigmoid function for the final output
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # First scale discriminator
    out_disc_1_conv = self.conv1_1(x)
    # Downsampling of the first discriminator
    out_disc_1 = self.down_conv1_1(out_disc_1_conv)
    out_disc_1 = self.down_conv1_2(out_disc_1)
    out_disc_1 = self.down_conv1_3(out_disc_1)
    out_disc_1 = torch.mean(out_disc_1, dim=[1,2,3])

    # Second scale discriminator
    out_down = self.av_pool(x) # Mean Pooling Downsampling
    # Downsampling of the second discriminator
    out_disc_2_conv = self.conv2_1(out_down)
    out_disc_2 = self.down_conv2_1(out_disc_2_conv)
    out_disc_2 = self.down_conv2_2(out_disc_2)
    out_disc_2 = self.down_conv2_3(out_disc_2) 
    out_disc_2 = torch.mean(out_disc_2, dim=[1,2,3])

    out = out_disc_1 + out_disc_2
    out = self.sigmoid(out)

    return out