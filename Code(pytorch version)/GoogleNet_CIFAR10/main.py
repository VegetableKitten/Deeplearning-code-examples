import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from GoogleNet import GoogleNet

'''定义超参数'''

epoch_total=10
learning_rate=1e-3


'''获取数据'''

train_data = torchvision.datasets.CIFAR10(root='./cifar10/',train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]))
text_data = torchvision.datasets.CIFAR10(root='./cifar10/',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]))

'''装载数据'''

train_loader = DataLoader(train_data,shuffle=True,batch_size=64)
text_loader = DataLoader(text_data,shuffle=True,batch_size=64)

'''调用模型'''

device = torch.device('cuda')
model=GoogleNet()
model.to(device)


'''设置损失函数和优化器'''
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

'''开始训练'''

def train():
    
    step_total=len(train_loader)
    
    for epoch in range(epoch_total):
        model.train()
        for step,(image,label) in enumerate(train_loader):
            
            image,label = image.to(device),label.to(device)

            pred=model(image)
            
            loss=loss_function(pred,label)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            if (step+1) % 20 == 0:
                
                print("Epoch:[{}/{}],step:[{}/{}],epoch:{:.4f}".format(epoch,epoch_total,step, step_total,loss.item()))

                model.eval()

                with torch.no_grad():

                    num = 0
                    sum = 0
                    acc = 0
                    max_acc = 0
                    for step,(image,label) in enumerate(text_loader):
                    
                        image,label = image.to(device),label.to(device)

                        pred = model(image)

                        pred = pred.argmax(dim=1)

                        sum += torch.eq(pred,label).float().sum().item()

                        num += image.size(0)

                    acc = sum/num
                    if acc>max_acc :
                        max_acc = acc
                        torch.save(model.state_dict(),'GoogleNet.pt')
                    print(acc)

'''调用train'''
if __name__ == '__main__':
    
    train()