import os
import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split

class CharacterData(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(225, 225)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.classes = sorted(os.listdir(root_dir))

    def calculate_mean_std(self):
        transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

        dataset = datasets.ImageFolder(self.root_dir, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_samples = 0

        for images, _ in loader:
            total_samples += 1
            mean += torch.mean(images, dim=(0, 2, 3))
            std += torch.std(images, dim=(0, 2, 3))

        mean /= total_samples
        std /= total_samples

        return mean.tolist(), std.tolist()

    def __len__(self):
        return sum([len(files) for _, _, files in os.walk(self.root_dir)])
    
    def __getitem__(self, idx):
        class_idx = 0
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            num_files = len(os.listdir(class_path))
            if idx < num_files:
                img_name = os.listdir(class_path)[idx]
                img_path = os.path.join(class_path, img_name)
                image = Image.open(img_path)

                # Convert to RGB from RGBA (rgba just has an added channel for transparency)
                image = image.convert("RGB")

                # Resize image within a bounding box while preserving aspect ratio
                image = self.resize_image(image, self.target_size)

                if self.transform:
                    image = self.transform(image)
                return image, class_idx
            else:
                idx -= num_files
                class_idx += 1

    def resize_image(self, image, target_size):
        width, height = image.size
        target_width, target_height = target_size

        # Calculate scaling factors for width and height
        width_ratio = target_width / width
        height_ratio = target_height / height

        # Choose the smaller scaling factor to preserve aspect ratio
        min_ratio = min(width_ratio, height_ratio)

        # Resize image while preserving aspect ratio
        resized_width = int(width * min_ratio)
        resized_height = int(height * min_ratio)
        image = image.resize((resized_width, resized_height), Image.ANTIALIAS)

        # Create a new blank image of the target size
        new_image = Image.new("RGB", target_size, (255, 255, 255))

        # Calculate padding
        pad_width = target_width - resized_width
        pad_height = target_height - resized_height
        left_pad = pad_width // 2
        top_pad = pad_height // 2
        right_pad = pad_width - left_pad
        bottom_pad = pad_height - top_pad

        # Paste the resized image onto the blank image with padding
        new_image.paste(image, (left_pad, top_pad))

        return new_image


def create_dataloader(data_root, batch_size, random_seed=42):
    # set seed for reproducibility
    torch.manual_seed(random_seed)

    full_dataset = CharacterData(root_dir=data_root)
    mean, std = full_dataset.calculate_mean_std()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    full_dataset.transform = transform
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def show_images(dataset, num_images = 2):
    fig, axes = plt.subplots(1, num_images, figsize=(225,225))

    for i in range(num_images):
        sample_idx = np.random.randint(len(dataset))
        image, label = dataset[sample_idx]
        image = image.permute(1,2,0)    # convert from (C,H,W) --> (H,W,C)
        axes[i].imshow(image)
        axes[i].set_title(f"{label}")
        axes[i].axis("off")

    plt.show()

# Parameters
batch_size = 64
data_root = r'C:\Users\ryanl\OneDrive\Documents\CNN Image Class\DATA'

if __name__ == "__main__":
    # Create data loaders
    train_loader, test_loader = create_dataloader(data_root, batch_size)

    # Print dataset sizes
    print("Number of images in training dataset:", len(train_loader.dataset))
    print("Number of images in the testing dataset:", len(test_loader.dataset))

    # Display sample images
    show_images(train_loader.dataset)
