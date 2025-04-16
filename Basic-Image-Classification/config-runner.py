import subprocess

# Define the hyperparameter values
learning_rates = [0.01, 0.001, 0.0001]
epochs_list = [25, 50, 75, 100]
batch_sizes = [32, 64, 128, 256, 512, 1024]


# Iterate over all combinations of hyperparameters
for lr in learning_rates:
    for e in epochs_list:
        for b in batch_sizes:
            print(f"Running with lr={lr}, batch_size={b}, epochs={e}")
            # Define the output file name
            output_file = f"./output-logs/output_lr{lr}_b{b}_e{e}.txt"
            # Open the file in write mode
            with open(output_file, "w") as f:
                # Call the version10.py script with the current hyperparameters
                subprocess.run(
                    [
                        "python",
                        "version10.py",
                        "--lr", str(lr),
                        "--b", str(b),
                        "--e", str(e)
                    ],
                    stdout=f,  # Redirect standard output to the file
                    stderr=subprocess.STDOUT  # Redirect standard error to the same file
                )