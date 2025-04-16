import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from common.utils import *
from common.train_utils import *


class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding="same")
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, padding="same")
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))  # 16 x 16 x 32
        x = self.bn1(x)
        x = torch.relu(F.max_pool2d(self.conv2(x), 2))  # 8 x 8 x 64
        x = self.bn2(x)
        x = torch.relu(F.max_pool2d(self.conv3(x), 2))  # 4 x 4 x 128
        x = self.bn3(x)
        x = torch.relu(F.max_pool2d(self.conv4(x), 2))  # 2 x 2 x 256
        x = self.bn4(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--b", type=int, default=64, help="Batch size")
    parser.add_argument("--e", type=int, default=25, help="Number of epochs")
    args = parser.parse_args()

    # Load the data
    train_loader, test_loader = get_data('cifar10', batch_size=args.b)

    # Create a model
    model = Net()
    print("Model Parameter Count:", sum(p.numel() for p in model.parameters()))

    # Create an optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    train(model, train_loader, optimizer, epochs=args.e)

    # Evaluate the model
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()