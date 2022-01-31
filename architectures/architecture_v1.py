import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """ Double convolution """

    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(DoubleConv, self).__init__()
        
        if not mid_channels:
            mid_channels = out_channels
            
        kernel_size = 7 
        assert kernel_size % 2
        padding = int((kernel_size - 1)/2)
        
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size, padding = padding, bias = False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv1d(mid_channels, out_channels, kernel_size, padding = padding, bias = False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)

    
class Down(nn.Module):
    """ Downscaling with 1d maxpooling, then double conv """

    def __init__(self, in_channels, out_channels, factor = 2):
        super(Down, self).__init__()
        assert (factor + 1) % 2
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(factor),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)    

    
class Up(nn.Module):
    """ Upscaling, then double conv """

    def __init__(self, in_channels, out_channels, linear = True, factor = 2):
        super(Up, self).__init__()
        assert (factor + 1) % 2
        
        # if linear = True, we use the convolutions to reduce the # of channels
        if linear:
            self.up = nn.Upsample(scale_factor = factor, mode = 'linear', align_corners = True) # mode='nearest'
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels , in_channels // 2, kernel_size = factor, stride = factor)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)
    
    
class OutConv(nn.Module):
    """ Final (output) convolution """    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out_conv = nn.Conv1d(in_channels, out_channels, kernel_size = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, comp_score):
        x = self.out_conv(x)
        if comp_score:
            x = self.sigmoid(x)
        
        return x
  
    
class DeepSleepNet(nn.Module):
    """ Build-up the DeepSleep 2.0 network from the previous modules """
    def __init__(self, in_channels = 13, linear = True):
        super(DeepSleepNet, self).__init__()
        
        self.in_channels = in_channels
        self.linear = linear

        self.inpc = DoubleConv(in_channels, 15)          # 13 x 4096*2048 -> 15  x 4096*2048
        self.down1 = Down(15, 30, 4)                     # 15 x 4096*2048 -> 30  x 4096*512
        self.down2 = Down(30, 60, 8)                     # 30 x 4096*512  -> 60  x 4096*64
        self.down3 = Down(60, 120, 16)                   # 60 x 4096*64   -> 120 x 4096*4
        factor = 2 if linear else 1
        self.down4 = Down(120, 240 // factor, 32)        # 120 x 4096*4   -> 240 x 512
        
        self.up1 = Up(240, 120 // factor, linear, 32)    # 240 x 4096*128 -> (120 // factor) x 4096*128
        self.up2 = Up(120, 60 // factor, linear, 16)
        self.up3 = Up(60, 30 // factor, linear, 8)
        self.up4 = Up(30, 15, linear, 4)
        self.outc = OutConv(15, 1)      
        
        """Initialize the weights of all convolutional layers with Xavier uniform"""
        for m in self.modules():
            if type(m) == nn.Conv1d:
                nn.init.xavier_uniform_(m.weight, gain = nn.init.calculate_gain('relu'))        
        
        
    def forward(self, x, comp_score = False):

        x1 = self.inpc(x)          
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.down4(x4)      

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x, comp_score)
        
        return x
    