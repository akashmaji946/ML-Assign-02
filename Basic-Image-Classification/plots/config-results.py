import os
import re
import pandas as pd

# Define the folder containing the logs
log_folder = "../output-logs"

# Define the hyperparameter values
learning_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [32, 64, 128, 256, 512, 1024]
epochs_list = [25, 50, 75, 100]

# Initialize a dictionary to store the results
results = {lr: pd.DataFrame(index=batch_sizes, columns=epochs_list) for lr in learning_rates}

# Function to extract test accuracy from a log file
def extract_test_accuracy(file_path):
    with open(file_path, "r") as f:
        for line in f:
            match = re.search(r"Test Acc: ([0-9.]+)", line)
            if match:
                return float(match.group(1))
    return None

# Iterate over all files in the log folder
for file_name in os.listdir(log_folder):
    # Match the file name pattern
    match = re.match(r"output_lr([0-9.]+)_b(\d+)_e(\d+)\.txt", file_name)
    if match:
        lr = float(match.group(1))
        batch_size = int(match.group(2))
        epochs = int(match.group(3))
        
        # Extract test accuracy from the file
        file_path = os.path.join(log_folder, file_name)
        test_accuracy = extract_test_accuracy(file_path)
        
        # Store the result in the corresponding DataFrame
        if lr in results:
            results[lr].loc[batch_size, epochs] = f"{test_accuracy:.4f}" if test_accuracy is not None else None

# Print the tables for each learning rate
for lr, df in results.items():
    print(f"Learning Rate: {lr}")
    print(df)
    print("\n")