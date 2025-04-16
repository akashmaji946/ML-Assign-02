import re
import matplotlib.pyplot as plt
import numpy as np

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

# Compute train error (1 - accuracy)
avg_train_error = [1 - acc for acc in avg_train_acc]

# Plot bar chart for train accuracy and error
epochs = range(1, len(avg_train_acc) + 1)
bar_width = 0.35  # Width of the bars

plt.figure(figsize=(10, 6))

# Bar positions
x = np.arange(len(epochs))

# Plot bars
plt.bar(x - bar_width / 2, avg_train_acc, bar_width, label="Train Accuracy", color="green")
plt.bar(x + bar_width / 2, avg_train_error, bar_width, label="Train Error", color="red")

# Add labels and title
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Train Accuracy and Error per Epoch")
plt.xticks(x, epochs)  # Set x-axis ticks to epoch numbers
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()