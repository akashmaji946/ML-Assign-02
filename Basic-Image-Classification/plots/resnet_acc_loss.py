import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['ResNet-20', 'ResNet-56', 'ResNet-100']
train_loss = [0.0010, 0.0001, 0.0001]
test_loss = [0.9151, 0.8591, 0.8637]
train_acc = [0.9999, 1.0000, 1.0000]
test_acc = [0.8482, 0.8735, 0.8800]

x = np.arange(len(models))
width = 0.35

# ----------- Accuracy Plot -----------
plt.figure(figsize=(10, 5))
bars1 = plt.bar(x - width/2, train_acc, width, label='Train Accuracy', color='skyblue')
bars2 = plt.bar(x + width/2, test_acc, width, label='Test Accuracy', color='salmon')

# Add value labels
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', fontsize=9)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', fontsize=9)

plt.ylabel('Accuracy')
plt.ylim(0.8, 1.05)
plt.title('Train vs Test Accuracy for ResNet Models')
plt.xticks(x, models)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ----------- Loss Plot -----------
plt.figure(figsize=(10, 5))
bars1 = plt.bar(x - width/2, train_loss, width, label='Train Loss', color='lightgreen')
bars2 = plt.bar(x + width/2, test_loss, width, label='Test Loss', color='orange')

# Add value labels
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', fontsize=9)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', fontsize=9)

plt.ylabel('Loss')
plt.title('Train vs Test Loss for ResNet Models')
plt.xticks(x, models)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
