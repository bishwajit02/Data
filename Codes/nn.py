import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Define Model
class Model(nn.Module):
    def __init__(self, in_features=5, h1=256, h2=512, h3=256, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.out = nn.Linear(h3, out_features)
        
        # Weight Initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.out(x)
        return x

# Pick a manual seed for randomization - helps with reproductibility
torch.manual_seed(42)

# Create an instance for the model

model = Model()

# Load Data
url = r"G:\Bishwajit\Dataset\data\Dataset\p.csv"
df = pd.read_csv(url)
df.drop(columns=['target_name', 'target_classification'], inplace=True)

# X, y Split
X = df.drop('classification_id', axis=1).values
y = df['classification_id'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 500
losses = []

for i in range(epochs):
  y_pred = model.forward(X_train)

  loss = criterion(y_pred, y_train)

  losses.append(loss.detach().numpy())

  if i % 100 == 0:
    print(f"Epoch: {i} and Loss: {loss}")

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

with torch.no_grad():
  y_eval = model.forward(X_test)

  loss = criterion(y_eval, y_test)
  print(loss)

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        predicted_classes = torch.argmax(y_pred, dim=1)
        correct += (predicted_classes == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
