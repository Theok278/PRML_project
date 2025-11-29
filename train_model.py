#!/usr/bin/env python3

import glob
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from pretreat_points import pretreat_points, resample_points

class StrokeDataset(Dataset):
    def __init__(self, files, seq_len=None, orientate=True, normalization=True, rectification=True):
        self.files = files
        self.seq_len = seq_len
        self.orientate = orientate
        self.normalization = normalization
        self.rectification = rectification

        self.data = []
        self.labels = []

        for f in files:
            df = pd.read_csv(f, header=None)
            pts = df.to_numpy()
            pts = pretreat_points(pts, orientate, normalization, rectification)
            if seq_len:
                pts = resample_points(pts, seq_len)

            self.data.append(pts.astype(np.float32))
            
            base = os.path.basename(f) 
            label = int(base.split("_")[1])
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return (sequence, label)
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 3)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_model(files, seq_len=50, batch_size=32, epochs=20, lr=0.001, test_split=0.2):
    # Create dataset
    dataset = StrokeDataset(files, seq_len=seq_len)
    
    # Split train/test
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_size=3, hidden_size=256, num_layers=3, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    accs = []
    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Accuracy: {acc:.4f}")
        accs.append(acc)
        losses.append(avg_loss)
    return model, accs, losses

files = sorted(glob.glob("digits_3d/training_data/stroke_*_*.csv"))
model, accs, losses = train_model(files, seq_len=50, batch_size=32, epochs=30, lr=0.001, test_split=0.5)
torch.save(model, "lstm_model.pth")

# Plot
fig, ax1 = plt.subplots()

color1 = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color=color1)
ax1.plot(accs, color=color1, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Loss', color=color2)
ax2.plot(losses, color=color2, label='Loss')
ax2.tick_params(axis='y', labelcolor=color2)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

plt.title('Training Accuracy and Loss')
plt.show()

# Confusion matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Recreate the dataset and split again (to get test set)
dataset = StrokeDataset(files, seq_len=50)
test_size = int(len(dataset) * 0.5)
train_size = len(dataset) - test_size
_, test_dataset = random_split(dataset, [train_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Optional: display nicely
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix on Test Set")
plt.show()
