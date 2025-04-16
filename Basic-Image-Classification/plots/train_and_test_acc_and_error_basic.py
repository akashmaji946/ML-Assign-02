import re
import matplotlib.pyplot as plt

# Input file paths
file_paths = [
    "../original-v9-runs/run1.txt",
    "../original-v9-runs/run2.txt",
    "../original-v9-runs/run3.txt"
]

# Initialize lists to store metrics
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Function to extract metrics from a file
def extract_metrics(file_path):
    train_loss = []
    train_acc = []
    test_loss = None
    test_acc = None

    with open(file_path, "r") as f:
        for line in f:
            # Match train loss and accuracy
            train_match = re.search(r"Epoch \d+ / \d+ \| Train Loss: ([0-9.]+) \| Train Acc: ([0-9.]+)", line)
            if train_match:
                train_loss.append(float(train_match.group(1)))
                train_acc.append(float(train_match.group(2)))

            # Match test loss and accuracy
            test_match = re.search(r"Test Loss: ([0-9.]+) \| Test Acc: ([0-9.]+)", line)
            if test_match:
                test_loss = float(test_match.group(1))
                test_acc = float(test_match.group(2))

    return train_loss, train_acc, test_loss, test_acc

# Extract metrics from all files
for file_path in file_paths:
    train_loss, train_acc, test_loss, test_acc = extract_metrics(file_path)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

# Compute averages for train loss and accuracy across epochs
avg_train_loss = [sum(epoch) / len(epoch) for epoch in zip(*train_losses)]
avg_train_acc = [sum(epoch) / len(epoch) for epoch in zip(*train_accuracies)]

# Compute averages for test loss and accuracy
avg_test_loss = sum(test_losses) / len(test_losses)
avg_test_acc = sum(test_accuracies) / len(test_accuracies)

# Print average test metrics
print(f"Average Test Loss: {avg_test_loss:.4f}")
print(f"Average Test Accuracy: {avg_test_acc:.4f}")

# Plot train and test metrics
epochs = range(1, len(avg_train_loss) + 1)

plt.figure(figsize=(12, 12))

# Train Loss
plt.subplot(2, 2, 1)
plt.plot(epochs, avg_train_loss, label="Train Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Average Train Loss")
plt.grid(True)
plt.legend()

# Train Accuracy
plt.subplot(2, 2, 2)
plt.plot(epochs, avg_train_acc, label="Train Accuracy", color="green")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Average Train Accuracy")
plt.grid(True)
plt.legend()

# # Test Loss
# plt.subplot(2, 2, 3)
# plt.plot(epochs, [avg_test_loss] * len(epochs), label="Test Loss", color="red")  # Constant value across epochs
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Average Test Loss")
# plt.grid(True)
# plt.legend()

# # Test Accuracy
# plt.subplot(2, 2, 4)
# plt.plot(epochs, [avg_test_acc] * len(epochs), label="Test Accuracy", color="orange")  # Constant value across epochs
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Average Test Accuracy")
# plt.grid(True)
# plt.legend()

plt.tight_layout()
plt.show()