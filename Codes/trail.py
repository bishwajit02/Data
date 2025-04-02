import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Astro(nn.Module):
  def __init__(self, in_features=5, h1=10, h2=12, out_features=139):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x

torch.manual_seed(42)
model = Astro()

url = r"G:\Bishwajit\Dataset\data\Dataset\w.csv"
df = pd.read_csv(url)

columns_to_drop = ['target_name', 'target_classification']
df = df.drop(columns=columns_to_drop)

X = df.drop('classification_id', axis=1)
y = df['classification_id']

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train).squeeze()  # Shift to zero-indexing
y_test = torch.LongTensor(y_test).squeeze() 
# Ensure labels start from 0
y_train = y_train - y_train.min()
y_test = y_test - y_test.min()


print("Unique y_train values after shifting:", y_train.unique())

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
losses = []

for i in range(epochs):
    # Forward pass: compute predicted y by passing X to the model
    y_pred = model(X_train)  # Model forward pass

    # Check the shape of y_pred and y_train
    print(f"y_pred shape: {y_pred.shape}")  # Should be (batch_size, 139)
    print(f"y_train shape: {y_train.shape}")  # Should be (batch_size,)

    # Compute the loss
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())

    # Print loss every 10 epochs
    if i % 10 == 0:
        print(f'Epoch: {i} | Loss: {loss.item()}')

    # Zero gradients, backpropagate, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()