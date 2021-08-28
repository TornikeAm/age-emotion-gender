import os 
import glob 
from imgaug import augmenters as iaa 
import shutil
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models



class AgeDataset(Dataset):
  def __init__(self,path):
    self.path=path
    self.images=os.listdir('Face')

  def generate_class(self,age):
    label=0
    if age >=0 and age<=12:
      label=0
    elif age >=12 and age<=18:
      label=1
    elif age >=18 and age<=30:
      label=2
    elif age >=30 and age<=50:
      label=3
    elif age >=50 and age<=70:
      label=4
    elif age >=70 and age<=106:
      label=5
    return label


  def __getitem__(self,index):
    img_path=os.path.join(self.path,self.images[index])
    image=cv.imread(img_path)
    image=cv.resize(image,(112,112))
    age=int(self.images[index].split('_')[1])
    label=self.generate_class(age)
    image=torch.tensor(image).float()
    label=torch.tensor(label)
    return image, label


  def __len__(self):
    return len(self.images)



ageData=AgeDataset('Face')

train_size = int(0.9 * len(ageData))
test_size = len(ageData) - train_size
age_train_dataset, age_test_dataset = torch.utils.data.random_split(ageData, [train_size, test_size])

age_train=DataLoader(age_train_dataset,batch_size=32,shuffle=True)
age_test=DataLoader(age_test_dataset,batch_size=16)

def load_age_model():
  resnet50 = models.resnet50(pretrained=True)
  num_ftrs=resnet50.fc.in_features
  resnet50.fc=nn.Linear(num_ftrs,6)
  return resnet50

torch.manual_seed(41)
ageModel=load_age_model()
ageModel

from tqdm import tqdm
def train(model,optim,loss_f,num_of_epochs,path):
  try:
    os.mkdir(path)
  except:
    path=path
  
  min_acc=30
  print('Training Age Model')
  for epoch in tqdm(range(num_of_epochs)):
    trn_corr,tst_corr=0,0
    train_losses=[]
    train_acc=[]
    for step,(x_train,y_train) in enumerate(age_train):
      x_train,y_train=x_train.to(device),y_train.to(device).long()
      x_train = x_train.permute(0, 3, 1, 2)
      y_pred=model(x_train)
      y_pred=F.log_softmax(y_pred,dim=1)
      loss=loss_f(y_pred,y_train)

      predicted = torch.max(y_pred.data, 1)[1]
      batch_corr = (predicted == y_train).sum()
      trn_corr += batch_corr

      optim.zero_grad()
      loss.backward()
      optim.step()
    tr_acc=trn_corr.item()*100/(32*step)
    train_losses.append(loss)
     
    
    with torch.no_grad():
      for step, (x_test,y_test) in enumerate(age_test):
        x_test,y_train=x_test.to(device),y_train.to(device)
        x_test = x_test.permute(0, 3, 1, 2)
        y_val=model(x_test)
        predicted=torch.max(y_val.data,1)[1]
        tst_corr+=(predicted==y_test).sum()
      val_loss=loss_f(y_val,y_test.long())
    vl_acc=tst_corr.item()*100/(16*step)

    if min_acc<vl_acc:
      min_acc=vl_acc
      torch.save(model.state_dict(),f"{path}/AgeModel.pt")
    
     
    print(f'epoch: {epoch:2} train_loss: {loss.item():10.4f}  \
    train_accuracy: {tr_acc:7.3f}% val_loss : {val_loss} val_acc :{vl_acc:7.3f}% ')

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(ageModel.parameters(),lr=0.01)
device=torch.device('cpu')

train(ageModel,optimizer,criterion,3,'models')


