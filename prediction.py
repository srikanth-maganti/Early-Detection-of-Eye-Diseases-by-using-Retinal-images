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


class CNN_Model(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            #input 3X128X128
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
            #output 32X128X128
            nn.ReLU(),
            #output 32X128X128
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
           
            nn.ReLU(),
            
            nn.MaxPool2d(2,2),#64X64X64

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),#128X32X32

            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),#256X16X16

            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),#256X8X8

            nn.Flatten(),
            nn.Linear(256*8*8,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,9))
    def forward(self,xb):
        return self.network(xb)
    

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
        model=CNN_Model()
        if torch.cuda.is_available():
            model=model.to(torch.device("cuda"),non_blocking=True)
        else:
            model=model.to(torch.device("cpu"),non_blocking=True)
        model.load_state_dict(torch.load('classifier_model.pth'))
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
        return np.ones(9) / 9 # Return uniform distribution on error
