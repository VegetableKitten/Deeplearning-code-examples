import torch
from torch import nn
from torch.nn import functional as F

class Resblk(nn.Module):

    def __init__(self, ch_in,ch_out,stride = 1):
        super(Resblk,self).__init__()
    
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        #self.extra = nn.Sequential()
        #if ch_in != ch_out:
        self.extra = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x): 
    
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.extra(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
    
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64)
        )

        self.blk1 = Resblk(64,128,stride=2)
        self.blk2 = Resblk(128,256,stride=2)
        self.blk3 = Resblk(256,512,stride=2)
        self.blk4 = Resblk(512,512,stride=2)


        self.outlayer = nn.Linear(512,10)
    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = F.adaptive_avg_pool2d(x,[1,1])
        x = x.view(x.size(0),-1)
        x = self.outlayer(x)

        return x

def main():
    res = ResNet18()
    tmp = torch.randn(2,3,32,32)
    out = res(tmp)
    print(out.shape)
if __name__ == '__main__':
    main()