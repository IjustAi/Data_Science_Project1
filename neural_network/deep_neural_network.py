import pandas as pd 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import  accuracy_score

heart_disease = fetch_ucirepo(id=45)
  
names=heart_disease.variables['name'][0:13]
columns_names= [name for name in names]
heart_disease_df= pd.DataFrame(heart_disease.data.features, columns=columns_names)
heart_disease_df['num']=heart_disease.data.targets


imputer = SimpleImputer(strategy='mean')
heart_disease_df = pd.DataFrame(imputer.fit_transform(heart_disease_df),columns=heart_disease_df.columns)
#heart_disease_df .info()


x= heart_disease_df.drop('num', axis=1)
y = heart_disease_df['num']        

x_np=x.to_numpy()
y_np = y.to_numpy()

x_tensor = torch.tensor(x_np,dtype=torch.float32).to('cpu')
y_tensor = torch.tensor(y_np,dtype=torch.long).to('cpu')


x_train,x_test,y_train,y_test = train_test_split(x_tensor,y_tensor,test_size=0.1,random_state=123)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test,y_test)

class Heart_disease(nn.Module):
    def __init__(self,input_size,h1,h2,h3,num_classes):
        super(Heart_disease,self).__init__()
        self.l1=nn.Linear(input_size,h1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(h1,h2)
        self.l3 = nn.Linear(h2,h3)
        self.l4 = nn.Linear(h3,num_classes)

    def forward(self,x):
        out = self.l1(x)
        out =self.relu(out)
        out = self.l2(out)
        out =self.relu(out)
        out =self.l3(out)
        out = self.relu(out)
        out =self.l4(out)
        return out 

input_size=13
h1=64
h2=32
h3=16
num_classes =5
num_epochs=100
batch_size=10
learning_rate=0.001

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

model =Heart_disease(input_size,h1,h2,h3,num_classes)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

print('strat training ')
for epoch in range(num_epochs):
    for i,(x,y) in enumerate(train_loader):
        outputs = model(x)
        Loss = loss(outputs,y)
    
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        
        print(f"epoch {epoch+1} / {num_epochs}, loss = {Loss.item():.4f} ")


with torch.no_grad():
    pred = np.argmax(model(x_test), axis=1)

    print('Results for the Model')
    print(accuracy_score(y_test, pred))
