from cProfile import label
from operator import index
from turtle import color
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
 
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

def one_hot (label):
    out = torch.zeros(label.size(0),10)
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index=idx,value=1)
    return out

def Visualization(data):
    fig = plt.figure()
    plt.plot(range(len(data)),data,color='blue')
    plt.legend(['value'],loc='upper right')
    plt.xlabel("step")
    plt.ylabel("value")
    plt.show()

def plot_image(img,label,name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1037,cmap='gray',interpolation=None)
        plt.title("{}:{}".format(name,label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)
 
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(28*28,256)
        self.fc2=nn.Linear(256,64)
        self.fc3=nn.Linear(64,10)
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

net = Net()
train_loss = []
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.09)
for epoch in range(10):
    for batch_idx,(x,y) in enumerate(train_loader):
        x = x.view(x.size(0), 28*28)
        out = net(x)
        y_one_hot=one_hot(y)
        loss = F.mse_loss(out,y_one_hot)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        # w' = w-lr*grad
        optimizer.step()

        if batch_idx % 10 == 0:
            print(epoch, batch_idx , loss.item())

Visualization(train_loss)

#net = Net()
#net.load_state_dict(torch.load('DNN.pt'))

total_correct = 0
for x,y in test_loader:
    x = x.view(x.size(0),28*28)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
total_num = len(test_loader.dataset)
acc = total_correct/total_num
print("test acc:",acc)

# Check the effect
x,y = next(iter(train_loader))
out = net(x.view(x.size(0),28*28))
pred = out.argmax(dim=1)
plot_image(x,pred,'test')
torch.save(net.state_dict(),'DNN.pt')

# Final Correct Rate ： 0.927
