import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import os

# -------------------------
# CONFIG
# -------------------------
BATCH_SIZE = 32
NUM_CLASSES = 20
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./data"  # should have 'train' and 'val' subfolders

# -------------------------
# DATASET + AUGMENTATION
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -------------------------
# MODEL
# -------------------------
model = EfficientNet.from_pretrained("efficientnet-b0")
model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# -------------------------
# LOSS, OPTIMIZER, SCHEDULER
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# -------------------------
# TRAINING + VALIDATION LOOP
# -------------------------
def train_one_epoch(epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch [{epoch}] Train Loss: {total_loss/total:.4f}, Accuracy: {acc:.4f}")

def validate(epoch):
    model.eval()
    correct, total = 0, 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Epoch [{epoch}] Val Loss: {val_loss/total:.4f}, Accuracy: {acc:.4f}")

# -------------------------
# MAIN TRAINING LOOP
# -------------------------
for epoch in range(1, NUM_EPOCHS + 1):
    train_one_epoch(epoch)
    validate(epoch)
    scheduler.step()

# -------------------------
# SAVE MODEL
# -------------------------
torch.save(model.state_dict(), "efficientnet_classification.pt")
print("Model saved.")
