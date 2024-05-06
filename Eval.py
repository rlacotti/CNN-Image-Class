import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

character_names = ["Brook", "Chopper", "Franky", "Jimbei",
                       "Luffy", "Nami", "Robin", "Sanji", "Usopp", "Zoro"]

def read_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    all_train_accuracies = []
    all_test_accuracies = []
    all_train_losses = []
    all_test_losses = []
    all_train_precision = []
    all_test_precision = []
    all_train_recall = []
    all_test_recall = []

    # iterate lines for extracting
    for i, line in enumerate(lines):
        if line.startswith("Train accuracies:"):
            train_accs = []
            # Extract all training accuracies until the next section
            while i+1 < len(lines) and not lines[i+1].startswith("Test accuracies:"):
                train_accs.extend([float(acc) for acc in lines[i+1].strip().split()])
                i += 1
            all_train_accuracies.append(train_accs)
        elif line.startswith("Test accuracies:"):
            test_accs = []
            # Extract all test accuracies until the next section
            while i+1 < len(lines) and not lines[i+1].startswith("Train losses:"):
                test_accs.extend([float(acc) for acc in lines[i+1].strip().split()])
                i += 1
            all_test_accuracies.append(test_accs)
        elif line.startswith("Train losses:"):
            train_losses = []
            # Extract all training losses until the next section
            while i+1 < len(lines) and not lines[i+1].startswith("Test losses:"):
                train_losses.extend([float(loss) for loss in lines[i+1].strip().split()])
                i += 1
            all_train_losses.append(train_losses)
        elif line.startswith("Test losses:"):
            test_losses = []
            # Extract all test losses until the next section
            while i+1 < len(lines) and not lines[i+1].startswith("Train precision:"):
                test_losses.extend([float(loss) for loss in lines[i+1].strip().split()])
                i += 1
            all_test_losses.append(test_losses)
        elif line.startswith("Train precision:"):
            train_precision = []
            # Extract all training precision until the next section
            while i+1 < len(lines) and not lines[i+1].startswith("Test precision:"):
                train_precision.extend([float(val.strip('[],')) for val in lines[i+1].strip().split()])
                i += 1
            all_train_precision.append(train_precision)
        elif line.startswith("Test precision:"):
            test_precision = []
            # Extract all test precision until the next section
            while i+1 < len(lines) and not lines[i+1].startswith("Train recall:"):
                test_precision.extend([float(val.strip('[],')) for val in lines[i+1].strip().split()])
                i += 1
            all_test_precision.append(test_precision)
        elif line.startswith("Train recall:"):
            train_recall = []
            # Extract all training recall until the next section
            while i+1 < len(lines) and not lines[i+1].startswith("Test recall:"):
                train_recall.extend([float(val.strip('[],')) for val in lines[i+1].strip().split()])
                i += 1
            all_train_recall.append(train_recall)
        elif line.startswith("Test recall:"):
            test_recall = []
            # Extract all test recall until the next section (including skipping over iteration heading)
            while i+1 < len(lines) and not lines[i+1].startswith("Train accuracies:") and not lines[i+1].startswith("Iteration"):
                test_recall.extend([float(val.strip('[],')) for val in lines[i+1].strip().split()])
                i += 1
            all_test_recall.append(test_recall)


    # Convert lists to numpy arrays
    train_accuracies = np.array(all_train_accuracies)
    test_accuracies = np.array(all_test_accuracies)
    train_losses = np.array(all_train_losses)
    test_losses = np.array(all_test_losses)

    # Concatenation for values by epoch
    train_precision_concat = np.concatenate(all_train_precision)
    test_precision_concat = np.concatenate(all_test_precision)
    train_recall_concat = np.concatenate(all_train_recall)
    test_recall_concat = np.concatenate(all_test_recall)

    # Reshape to (num_epochs, num_classes)
    train_precision_reshaped = train_precision_concat.reshape(-1, len(character_names))
    test_precision_reshaped = test_precision_concat.reshape(-1, len(character_names))
    train_recall_reshaped = train_recall_concat.reshape(-1, len(character_names))
    test_recall_reshaped = test_recall_concat.reshape(-1, len(character_names))

    # Convert to Pandas dataframe
    train_precision = pd.DataFrame(data=train_precision_reshaped,
                                   index= [f"Epoch_{i}" for i in range(1, len(train_precision_reshaped) + 1)],
                                   columns=character_names)
    test_precision = pd.DataFrame(data = test_precision_reshaped,
                                  index=[f"Epoch_{i}" for i in range(1, len(test_precision_reshaped) + 1)],
                                  columns=character_names)
    train_recall = pd.DataFrame(data=train_recall_reshaped,
                                index = [f"Epoch_{i}" for i in range(1, len(train_recall_reshaped) + 1)],
                                columns=character_names)
    test_recall = pd.DataFrame(data=test_recall_reshaped,
                               index = [f"Epoch_{i}" for i in range(1, len(test_recall_reshaped) + 1)],
                               columns=character_names)
    

    return train_accuracies, test_accuracies, train_losses, test_losses, train_precision, test_precision, train_recall, test_recall

def plot_accuracies_and_losses(train_accuracies, test_accuracies, train_losses, test_losses, save_pth=None):
    # Reshape test and train accuracies and losses to be continuous
    test_accuracies_flat = test_accuracies.flatten()
    train_accuracies_flat = train_accuracies.flatten()
    test_losses_flat = test_losses.flatten()
    train_losses_flat = train_losses.flatten()

    # Create epochs for continuous plotting
    epochs = range(1, len(test_accuracies_flat) + 1)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot accuracies
    ax1.plot(epochs, test_accuracies_flat, label='Test Accuracies', color='red')
    ax1.plot(epochs, train_accuracies_flat, label='Train Accuracies', color='black')
    ax1.set_title('Train and Test Accuracies Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot losses
    ax2.plot(epochs, test_losses_flat, label='Test Losses', color='red')
    ax2.plot(epochs, train_losses_flat, label='Train Losses', color='black')
    ax2.set_title('Train and Test Losses Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_pth:
        plt.savefig(save_pth)
    plt.show()



def plot_lines(dataframe, title, interval=5, save_pth=None):
    plt.figure(figsize=(12,8))
    dataframe.index = dataframe.index.str.split('_').str[1].astype(int)  # Extract the epoch number
    for column in dataframe.columns:
        plt.plot(dataframe.index[::interval], dataframe[column][::interval], label=column)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    if save_pth:
        plt.savefig(save_pth)
    plt.show()

def main():
    file_path = r"C:\Users\ryanl\OneDrive\Documents\CNN Image Class\checkpoints\metrics_epoch_1.txt"
    train_accuracies, test_accuracies, train_losses, test_losses, train_precision, test_precision, train_recall, test_recall = read_data(file_path)
    save_dir = r"C:\Users\ryanl\OneDrive\Documents\CNN Image Class\Plots"
    plot_accuracies_and_losses(train_accuracies, test_accuracies, train_losses, test_losses, save_pth=os.path.join(save_dir, "train_test_acc.png"))
    plot_lines(test_recall, "Test Recall over Epochs", interval=5, save_pth=os.path.join(save_dir, "test_recall.png"))



if __name__ == "__main__":
     main()


