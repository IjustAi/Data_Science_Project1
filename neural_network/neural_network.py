import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR, StepLR

heart_disease = fetch_ucirepo(id=45)
names = heart_disease.variables['name'][0:13]
columns_names = [name for name in names]
heart_disease_df = pd.DataFrame(heart_disease.data.features, columns=columns_names)
heart_disease_df['num'] = heart_disease.data.targets

#missing_values_count = heart_disease_df.isnull().sum()
#total_missing_values = missing_values_count.sum()
imputer = SimpleImputer(strategy='mean')
heart_disease_df = pd.DataFrame(imputer.fit_transform(heart_disease_df), columns=heart_disease_df.columns)

x = heart_disease_df.drop('num', axis=1)
y = heart_disease_df['num']
x_np = x.to_numpy()
y_np = y.to_numpy()
x_tensor = torch.tensor(x_np, dtype=torch.float32).to('cpu')
y_tensor = torch.tensor(y_np, dtype=torch.long).to('cpu')

x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.1, random_state=1)
y_train[y_train > 0] = 1
y_test[y_test > 0] = 1

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

input_size = 13
h1 = 32
h2 = 64
h3 = 128
h4 = 64
h5 = 16
num_epochs = 200
batch_size = 16
learning_rate = 1e-3

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class HeartDiseaseModel(nn.Module):
    def __init__(self, input_size, h1, h2, h3, h4, h5):
        super(HeartDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.relu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(h1, h2)
        self.relu2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(h2, h3)
        self.relu3 = nn.LeakyReLU()
        self.drop3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(h3, h4)
        self.relu4 = nn.LeakyReLU()
        self.drop4 = nn.Dropout(0.1)
        self.fc5 = nn.Linear(h4, h5)
        self.relu5 = nn.LeakyReLU()
        self.fc6 = nn.Linear(h5, 1)

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.drop3(self.relu3(self.fc3(x)))
        x = self.drop4(self.relu4(self.fc4(x)))
        x = self.relu5(self.fc5(x))
        y = torch.sigmoid(self.fc6(x))
        return y

model = HeartDiseaseModel(input_size, h1, h2, h3, h4, h5)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

losses = []
accuracies = []

warmup_epochs = 100
scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min((epoch + 1) / warmup_epochs, 1))

print('Start warm-up training')
for epoch in range(warmup_epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for i, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(x_batch).squeeze()
        y_batch = y_batch.float()
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    epoch_loss /= len(train_loader)
    accuracy = 100 * correct / total
    losses.append(epoch_loss)
    accuracies.append(accuracy)
    scheduler_warmup.step()
    print(f"Warm-up Epoch {epoch + 1}/{warmup_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)

for epoch in range(warmup_epochs, num_epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for i, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(x_batch).squeeze()
        y_batch = y_batch.float()
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    epoch_loss /= len(train_loader)
    accuracy = 100 * correct / total
    losses.append(epoch_loss)
    accuracies.append(accuracy)
    scheduler.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Training Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

model.eval()
with torch.no_grad():
    pred = np.round(model(x_test).squeeze().cpu().numpy()).astype(int)
    print('Results for Model')
    print(accuracy_score(y_test, pred))
  