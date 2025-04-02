import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 🔹 Load Data
url = r"G:\Bishwajit\Dataset\data\Dataset\p.csv"
df = pd.read_csv(url)

# Drop unwanted columns
columns_to_drop = ['target_name', 'target_classification']
df = df.drop(columns=columns_to_drop)

# 🔹 Encode labels correctly (Ensure 0-based indexing)
df['classification_id'], _ = pd.factorize(df['classification_id'])

# Define features and target
X = df.drop('classification_id', axis=1).values
y = df['classification_id'].values

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 🔹 Ensure labels are in the correct range
num_classes = len(set(y_train.tolist()))
print(f"Number of unique classes: {num_classes}")

# 🔹 Check if labels are within the expected range
assert y_train.min() == 0, "Error: y_train should start at 0"
assert y_train.max() == num_classes - 1, f"Error: y_train max should be {num_classes - 1}, but got {y_train.max()}"

# 🔹 Define Neural Network using Sequential API
class Astro(nn.Module):
    def __init__(self, input_dim, hidden1=10, hidden2=12, output_dim=139):
        super(Astro, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.network(x)  # No activation for logits

# Initialize model
model = Astro(input_dim=X_train.shape[1], output_dim=num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🔹 Mini-batch training setup
batch_size = 64
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 🔹 Training loop
epochs = 50
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()

    # Print loss every 10 epochs
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

print("✅ Training complete")
