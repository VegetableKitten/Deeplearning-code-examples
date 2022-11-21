import torch
from torch.utils.data import DataLoader
import visdom
from torch import optim
import dataset,Resnet
from torch.nn import functional as F

train_data = dataset.PoKemon('pokeman',224,'train')
val_data = dataset.PoKemon('pokeman',224,'val')
text_data = dataset.PoKemon('pokeman',224,'text')

train_data = DataLoader(train_data,shuffle=True,batch_size=32)
val_data = DataLoader(val_data,shuffle=True,batch_size=32)
text_data = DataLoader(text_data,shuffle=True,batch_size=32)

device = torch.device('cuda')
net = Resnet.ResNet18()
opt = optim.Adam(net.parameters(),lr=1e-3)
net.to(device)
best_acc = 0

#for epoch in range(10):
#    for i,(x,y) in enumerate(train_data):
#        x = x.to(device)
#        y = y.to(device)
#        x = net.forward(x)
#        loss = F.cross_entropy(x,y).to(device)
#        opt.zero_grad()
#        loss.backward()
#        opt.step()
#        print(epoch, i, loss.item())
#
#        if i%10 == 0:
#            num_acc = 0
#            sum_acc = 0
#            acc = 0
#            for i2,(xx,yy) in enumerate(val_data):
#                xx = xx.to(device)
#                yy = yy.to(device)
#                xx = net.forward(xx)
#                pred = xx.argmax(dim=1)
#                sum_acc += torch.eq(pred,yy).float().sum().item()
#                num_acc += xx.size(0)
#            
#            acc = sum_acc/num_acc
#            print(acc)
#            if acc>best_acc :
#                best_acc = acc
#                torch.save(net.state_dict(),'DIYnet.pt')

net.load_state_dict(torch.load('DIYnet.pt'))
sum_acc = 0
num_acc = 0

for i3,(xxx,yyy) in enumerate(text_data):
    xxx = xxx.to(device)
    yyy = yyy.to(device)
    xxx = net.forward(xxx)
    pred = xxx.argmax(dim=1)
    sum_acc += torch.eq(pred,yyy).float().sum().item()
    num_acc += xxx.size(0)

acc = sum_acc/num_acc
print(acc)