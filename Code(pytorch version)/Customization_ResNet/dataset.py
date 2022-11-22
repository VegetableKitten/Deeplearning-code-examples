import torch
import os,glob,visdom
import random,csv
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms

class PoKemon(Dataset):
    def __init__(self,root,resize,mode):
        super(PoKemon,self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root,name)):
                continue
            
            self.name2label[name] = int(len(self.name2label.keys()))
        
        self.images,self.labels = self.load_csv('Data.csv')

        if mode =='train':
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode =='val':
            self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        elif mode =='text':
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]
        
    def load_csv(self,filename):

        if not os.path.isdir(os.path.join(self.root,filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root,name,'*.png'))
                images += glob.glob(os.path.join(self.root,name,'*.jpg'))
                images += glob.glob(os.path.join(self.root,name,'*.jpeg'))
            
            random.shuffle(images)

            with open(os.path.join(self.root,filename),mode='w',newline='') as f:
                write = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    write.writerow([img,label])
        
        with open(os.path.join(self.root,filename),mode='r') as f:
            images,labels = [],[]
            read = csv.reader(f)
            for row in read:
                img,label = row
                images.append(img)
                labels.append(int(label))
        
        assert len(images)==len(labels)

        return images,labels

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):
        
        img,label = self.images[index],self.labels[index]
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),#string ==> img Data
            transforms.Resize((int(self.resize*1.25),int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ])

        img = tf(img)
        label = torch.tensor(label)
        
        return img,label

    def denormalize(self,x):

        mean = [0.485,0.456,0.406]
        std = [0.229,0.224,0.225]

        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x*std+mean
        return x

def main():
    tep =PoKemon('pokeman',224,'train')
if __name__ == '__main__':
    main()