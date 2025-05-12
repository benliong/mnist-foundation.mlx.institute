# train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import BATCH_SIZE, EPOCHS, DEVICE, MNIST_MEAN, MNIST_STD

# 1 - MNIST loaders ----------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))         # mean / std
])

train_set = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# 2 - Tiny CNN ---------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),          # 28→26
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),         # 26→24
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 24→12
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = Net().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3 - Training loop ----------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    # quick test accuracy each epoch
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    print(f"Epoch {epoch}: {(100*correct/total):.2f}% test accuracy")

# 4 - Save weights -----------------------------------------------------------
torch.save(model.state_dict(), "model.pt")
print("Saved → model.pt")