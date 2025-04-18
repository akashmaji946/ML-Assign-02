import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['PlainNet-20', 'PlainNet-56', 'PlainNet-110']
train_loss = [0.0051, 0.3165, 1.6808]
test_loss = [0.9543, 0.8537, 1.6730]

train_acc = [0.9989, 0.8882, 0.3479]
test_acc = [0.8299, 0.7465, 0.3511]

x = np.arange(len(models))  # Label locations
width = 0.35  # Bar width

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
plt.ylim(0, 1.05)
plt.title('Train vs Test Accuracy for Models')
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
plt.title('Train vs Test Loss for Models')
plt.xticks(x, models)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
