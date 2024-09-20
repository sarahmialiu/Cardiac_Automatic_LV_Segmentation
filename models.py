import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
#from torchvision.models import resnet101


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, upsample=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=upsample),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet_LSTM(nn.Module): # Not working
    def __init__(self, num_classes=2, img_ch=1, output_ch=1, first_layer_numKernel=64):
        super(UNet_LSTM, self).__init__()
        self.num_lstm_layers = 3
        self.hidden_size = 64*first_layer_numKernel

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=first_layer_numKernel)
        self.Conv2 = conv_block(ch_in=first_layer_numKernel, ch_out=2*first_layer_numKernel)
        self.Conv3 = conv_block(ch_in=2*first_layer_numKernel, ch_out=4*first_layer_numKernel)
        self.Conv4 = conv_block(ch_in=4*first_layer_numKernel, ch_out=8*first_layer_numKernel)
        self.Conv5 = conv_block(ch_in=8*first_layer_numKernel, ch_out=16*first_layer_numKernel)
        self.Conv6 = conv_block(ch_in=16*first_layer_numKernel, ch_out=32*first_layer_numKernel)
        self.Conv7 = conv_block(ch_in=32*first_layer_numKernel, ch_out=64*first_layer_numKernel)

        # self.lstm = nn.LSTM(input_size=8*first_layer_numKernel, hidden_size=self.hidden_size, 
        #                     num_layers=self.num_lstm_layers, batch_first=True, bidirectional=True)

        self.lstm = nn.LSTM(input_size=64*first_layer_numKernel, hidden_size=64*first_layer_numKernel, 
                            num_layers=3, batch_first=True, bidirectional=True)

        self.Up7 = up_conv(ch_in=2*64*first_layer_numKernel, ch_out=32*first_layer_numKernel,upsample=3) #bidirectional doubles the number of output features
        self.Up_conv7 = conv_block(ch_in=64*first_layer_numKernel, ch_out=32*first_layer_numKernel)

        self.Up6 = up_conv(ch_in=32*first_layer_numKernel, ch_out=16*first_layer_numKernel, upsample=5)
        self.Up_conv6 = conv_block(ch_in=32*first_layer_numKernel, ch_out=16*first_layer_numKernel)

        self.Up5 = up_conv(ch_in=16*first_layer_numKernel, ch_out=8*first_layer_numKernel, upsample=5)
        self.Up_conv5 = conv_block(ch_in=16*first_layer_numKernel, ch_out=8*first_layer_numKernel)

        self.Up4 = up_conv(ch_in=8*first_layer_numKernel, ch_out=4*first_layer_numKernel)
        self.Up_conv4 = conv_block(ch_in=8*first_layer_numKernel, ch_out=4*first_layer_numKernel)

        self.Up3 = up_conv(ch_in=4*first_layer_numKernel, ch_out=2*first_layer_numKernel)
        self.Up_conv3 = conv_block(ch_in=4*first_layer_numKernel, ch_out=2*first_layer_numKernel)

        self.Up2 = up_conv(ch_in=2*first_layer_numKernel, ch_out=first_layer_numKernel)
        self.Up_conv2 = conv_block(ch_in=2*first_layer_numKernel, ch_out=first_layer_numKernel)

        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(first_layer_numKernel, output_ch, kernel_size=1, stride=1, padding=0), nn.Sigmoid()  # Use sigmoid activation for binary segmentation
        )
        
       
    def forward(self, x_3d):
        x1 = self.Conv1(x_3d)
        # print(x1.shape)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # print(x2.shape)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # print(x3.shape)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        # print(x4.shape)

        x5 = self.Maxpool1(x4)
        x5 = self.Conv5(x5)
        # print(x5.shape)
                
        x6 = self.Maxpool2(x5)
        x6 = self.Conv6(x6)
        # print(x6.shape)

        x7 = self.Maxpool3(x6)
        x7 = self.Conv7(x7)
        # print(x7.shape)

        frames = x7.squeeze()

        # print(frames.unsqueeze(0).shape)
        # torch.cat(x) # (405, 4096)
        
        # Pass latent representation of frame through lstm and update hidden state
        out, hidden = self.lstm(frames.unsqueeze(0))         
        # print(out.shape)
        out = out.squeeze().unsqueeze(-1).unsqueeze(-1)
        # print(out.shape)

        # pass list of 1d tensors through lstm: out, hidden = self.lstm(list)
        d7 = self.Up7(out)
        # print(d7.shape)
        d7 = torch.cat((x6, d7), dim=1)
        d7 = self.Up_conv7(d7)

        d6 = self.Up6(d7)
        # print(d6.shape)
        d6 = torch.cat((x5, d6), dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(d6)
        # print(d5.shape)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        # print(d4.shape)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # print(d3.shape)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # print(d2.shape)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        # print(d1.shape)    
    
        return d1
    
class UNet_LSTM2(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, first_layer_numKernel=32):
        super(UNet_LSTM2, self).__init__()
        self.num_lstm_layers = 2
        self.lstm_size = 15*15*16*first_layer_numKernel
        self.hidden_size = 15*15*4*first_layer_numKernel

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=5, stride=5)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=first_layer_numKernel)
        self.Conv2 = conv_block(ch_in=first_layer_numKernel, ch_out=2*first_layer_numKernel)
        self.Conv3 = conv_block(ch_in=2*first_layer_numKernel, ch_out=4*first_layer_numKernel)
        self.Conv4 = conv_block(ch_in=4*first_layer_numKernel, ch_out=8*first_layer_numKernel)
        self.Conv5 = conv_block(ch_in=8*first_layer_numKernel, ch_out=16*first_layer_numKernel)
        
        # self.lstm = nn.LSTM(input_size=8*first_layer_numKernel, hidden_size=self.hidden_size, 
        #                     num_layers=self.num_lstm_layers, batch_first=True, bidirectional=True)

        self.lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.hidden_size, 
                            num_layers=self.num_lstm_layers, batch_first=True, bidirectional=True)
        
        self.Up5 = up_conv(ch_in=8*first_layer_numKernel, ch_out=8*first_layer_numKernel, upsample=5)
        self.Up_conv5 = conv_block(ch_in=16*first_layer_numKernel, ch_out=8*first_layer_numKernel)

        self.Up4 = up_conv(ch_in=8*first_layer_numKernel, ch_out=4*first_layer_numKernel)
        self.Up_conv4 = conv_block(ch_in=8*first_layer_numKernel, ch_out=4*first_layer_numKernel)

        self.Up3 = up_conv(ch_in=4*first_layer_numKernel, ch_out=2*first_layer_numKernel)
        self.Up_conv3 = conv_block(ch_in=4*first_layer_numKernel, ch_out=2*first_layer_numKernel)

        self.Up2 = up_conv(ch_in=2*first_layer_numKernel, ch_out=first_layer_numKernel)
        self.Up_conv2 = conv_block(ch_in=2*first_layer_numKernel, ch_out=first_layer_numKernel)

        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(first_layer_numKernel, output_ch, kernel_size=1, stride=1, padding=0), nn.Sigmoid()  # Use sigmoid activation for binary segmentation
        )
       
    def forward(self, x_3d):
        x1 = self.Conv1(x_3d)
        print(x1.shape)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        print(x2.shape)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        print(x3.shape)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        print(x4.shape) # (21, 256, 75, 75)

        x5 = self.Maxpool1(x4)
        x5 = self.Conv5(x5)
        print(x5.shape) # (21, 512, 15, 15)

        flat = x5.view((21, -1, 1, 1)) # (21, 115200, 1, 1)
 
        flat = flat.squeeze().unsqueeze(0) # (1, 21, 115200)

        print(flat.shape) # (1, 21, 115200)
        
        # Pass latent representation of frame through lstm and update hidden state
        out, hidden = self.lstm(flat)         
        print(out.shape) # (1, 21, 25600)
        out = out.squeeze().unsqueeze(-1).unsqueeze(-1)
        print(out.shape) # (21, 25600, 1, 1)

        out = out.view(21, -1, 15, 15)
        print(out.shape) # (21, 256, 15, 15)

        # pass list of 1d tensors through lstm: out, hidden = self.lstm(list)
        d5 = self.Up5(out)
        print(d5.shape)
        d5 = torch.cat(x4, d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        print(d4.shape)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        print(d3.shape)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        print(d2.shape)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        print(d1.shape)    
    
        return d1

class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, first_layer_numKernel=64):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=first_layer_numKernel)
        self.Conv2 = conv_block(ch_in=first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Conv3 = conv_block(ch_in=2 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Conv4 = conv_block(ch_in=4 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        #self.Conv5 = conv_block(ch_in=8 * first_layer_numKernel, ch_out=16 * first_layer_numKernel)

        #self.Up5 = up_conv(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        #self.Up_conv5 = conv_block(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)

        self.Up4 = up_conv(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Up_conv4 = conv_block(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)

        self.Up3 = up_conv(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Up_conv3 = conv_block(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)

        self.Up2 = up_conv(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)
        self.Up_conv2 = conv_block(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)

        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(first_layer_numKernel, output_ch, kernel_size=1, stride=1, padding=0), nn.Sigmoid()  # Use sigmoid activation for binary segmentation
        )

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # x5 = self.Maxpool(x4)
        # print(x5.shape)
        # x5 = self.Conv5(x5)
        # print(x5.shape)

        # decoding + concat path
        # d5 = self.Up5(x5)
        # print(d5.shape)  
        # d5 = torch.cat((x4, d5), dim=1) # PROBLEM HERE
        # print(d5.shape)

        # d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4) # d5
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1