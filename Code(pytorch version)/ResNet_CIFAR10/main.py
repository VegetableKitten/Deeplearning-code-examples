import torch
from torch import nn,optim
from torch.utils.data import DataLoader
import torchvision
from Resnet import ResNet18,Resblk
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

Dictionary = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
def plt_image(imag,label,label2):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        image = imag[i]
        r = image[0] 
        g = image[1] 
        b = image[2] 
        r = (r)*0.229+0.485
        g = (g)*0.224+0.456
        b = (b)*0.225+0.406
        img = torch.zeros((32,32,3))
        img[:,:,0]=r
        img[:,:,1]=g
        img[:,:,2]=b
        plt.imshow(img)
        plt.title("{}:{}\n{}:{}".format('Pred',Dictionary[label[i].item()],'True',Dictionary[label2[i].item()]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def main():
    batch_size = 64
    cifar_train = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./cifar10/',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((32,32)),
                torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                ])),
            batch_size = batch_size,shuffle=True)

    cifar_test = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./cifar10/',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((32,32)),
                torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                ])),
            batch_size = batch_size,shuffle=True)
    
    device = torch.device('cuda')
    net = ResNet18()
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(),lr=1e-3)
    accmax = 0
    #net.load_state_dict(torch.load('Resnet.pt'))
    for epoch in range(8) :
        net.train()
        for (batchidx,(x,y)) in enumerate(cifar_train):
            x,y=x.to(device),y.to(device)
            x=net(x)
            loss = F.cross_entropy(x,y).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, loss.item())
        net.eval()
        with torch.no_grad():
            acc_sum = 0
            acc_num = 0
            for (batchidx,(x,y)) in enumerate(cifar_test):
                x,y=x.to(device),y.to(device)
                x = net(x)
                pred = x.argmax(dim=1)
                acc_sum+= torch.eq(pred,y).float().sum().item()
                acc_num+= x.size(0)

            acc = acc_sum/acc_num
            if acc>accmax :
                accmax = acc
                torch.save(net.state_dict(),'Resnet.pt')
            print(acc)

# Check the effect
    x,y = next(iter(cifar_train))
    x = x.to(device)
    out = net(x)
    pred = out.argmax(dim=1)
    plt_image(x,pred,y)
if __name__ =='__main__':
    main()