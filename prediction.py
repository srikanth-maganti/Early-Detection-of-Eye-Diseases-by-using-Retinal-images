import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import numpy as np
import streamlit as st

class ImageClassificationBase(nn.Module):
    def training_step(self,batch):
        images,labels=batch
        outputs=self(images)
        loss=F.cross_entropy(outputs,labels)
        return loss
    def validation_step(self,batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
    def validation_epoch_end(self,outputs):
        acc=[x["val_acc"] for x in outputs]
        loss=[x['val_loss'] for x in outputs]

        mean_acc=torch.stack(acc).mean()
        mean_loss=torch.stack(loss).mean()
        return {"val_loss":mean_loss.item(),"val_acc":mean_acc.item()}
    def epoch_end(self,epoch,result):
        print("Epoch [{}], train_loss:{:.4f} val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result["train_loss"],result['val_loss'], result['val_acc']))

def accuracy(outputs,labels):
    _,preds=torch.max(outputs,dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))
def conv_block(in_channels,out_channels,pool=False):
    layers=[nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=1),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace=True)]
    if pool:layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        #3X32X32
        self.conv1=conv_block(in_channels,64)#64X128X128
        self.conv2=conv_block(64,128,pool=True)#128X64X64
        self.res1=nn.Sequential(conv_block(128,128),
                               conv_block(128,128))#128X64X64
        self.conv3=conv_block(128,256,pool=True) #256X32X32
        self.conv4=conv_block(256,512,pool=True) #512X16X16
        self.res2=nn.Sequential(conv_block(512,512),
                               conv_block(512,512))

        # self.conv5=conv_block(512,1024,pool=True) #1024X8x8
        # self.res3=nn.Sequential(conv_block(1024,1024),
        #                        conv_block(1024,1024))
        self.classifier=nn.Sequential(nn.MaxPool2d(4), #512X4X4
                                     nn.Flatten(),#512
                                     nn.Dropout(0.2),#to avoid overfitting (randomly pick 20% output and set it to 0)
                                     nn.Linear(512*4*4,num_classes)) #10
    def forward(self,xb):
        out=self.conv1(xb)
        out=self.conv2(out)
        out=self.res1(out)+out
        out=self.conv3(out)
        out=self.conv4(out)
        out=self.res2(out)+out
        # out=self.conv5(out)
        # out=self.res3(out)+out
        out=self.classifier(out)
        return out
        

# Function to preprocess image for PyTorch
def preprocess_image(image):
    # Define the same transforms used during model training
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply transformations
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Move to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        
    return input_batch

# Function to load PyTorch model
def load_model():
    try:
        model=ResNet9(3,10)
        if torch.cuda.is_available():
            model=model.to(torch.device("cuda"),non_blocking=True)
        else:
            model=model.to(torch.device("cpu"),non_blocking=True)
        model.load_state_dict(torch.load('final.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Function to make predictions
def predict(model, image):
   
    try:
        
        input_batch = preprocess_image(image)
        with torch.no_grad():
             output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities.cpu().numpy()
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return np.ones(10) / 10 # Return uniform distribution on error
