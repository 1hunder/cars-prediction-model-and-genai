import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# === 1. Load and encode data ===
df = pd.read_csv("data/data1.csv")
df = df.drop(columns=["Horsepower", "ColorGroup"])

# One-hot encoding for Brand and Model
df = pd.get_dummies(df, columns=["Brand", "Model"])

# Encode 'Fuel Type'
df["Fuel Type"] = LabelEncoder().fit_transform(df["Fuel Type"])

# === 2. Prepare data ===
X = df.drop(columns=["Price"])
y = df["Price"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# === 3. Create DataLoaders ===
def to_loader(X, y, batch_size=16):
    X_tensor = torch.tensor(X.reshape(X.shape[0], 1, X.shape[1]), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

train_loader = to_loader(X_train, y_train)
val_loader = to_loader(X_val, y_val)
test_loader = to_loader(X_test, y_test)

# === 4. Define CNN ===
class HousePriceCNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size)
            out = self.forward_conv(dummy)
            size = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(size, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward_conv(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x).squeeze()

# === 5. Define loss and metrics ===
class MAPELoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.abs((target - pred) / torch.clamp(target, min=1.0))) * 100

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(0, y_pred))))
    r2 = r2_score(y_true, y_pred)
    return r2, mae, rmse, mape, rmsle

# === 6. Train CNN ===
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HousePriceCNN(X_train.shape[1]).to(device)
criterion = MAPELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

best_loss = float("inf")
patience = 10
wait = 0

for epoch in range(100):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1} - Train Loss: {np.mean(train_losses):.2f}, Val Loss: {val_loss:.2f}")

    if val_loss < best_loss:
        best_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# === 7. Test Evaluation ===
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy()
        y_pred.extend(np.maximum(preds, 0))
        y_true.extend(yb.numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

r2, mae, rmse, mape, rmsle = evaluate(y_true, y_pred)
print(f"\n📊 Test Metrics:")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"RMSLE: {rmsle:.4f}")
