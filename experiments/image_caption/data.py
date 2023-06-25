#%%
import os
import random
import torch
import torchvision

from PIL import Image
import matplotlib.pyplot as plt

# Load the dataset
 
class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, caption_file, image_folder, transform=None, caption_transform=None):
        self.image_folder = os.path.expanduser(image_folder)
        self.caption_file = os.path.expanduser(caption_file)
        self.transform = transform
        self.caption_transform = transform
        
        self.image_paths = []
        self.captions = []
        
        with open(self.caption_file, 'r') as f:
            next(f) # ignore the first line
            for line in f:
                image_path_caption = line.strip().split(',', 2)
                image_path = image_path_caption[0]
                caption = image_path_caption[1]
                self.image_paths.append(image_path)
                self.captions.append(caption)
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        image = Image.open(
            os.path.join(self.image_folder, image_path)).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        if self.caption_transform is not None:
            caption = self.caption_transform(caption)
        
        return image, caption

# %%
def plot_image_caption_grid(dataset, num_images=4):
    fig, axs = plt.subplots(nrows=num_images, figsize=(10, 10))
    random_indices = random.sample(range(len(dataset)), num_images)
    for i in range(num_images):
        image, caption = dataset[random_indices[i]]
        axs[i].imshow(image)
        axs[i].set_title(caption)
        axs[i].axis('off')

    plt.show()

# Test the function
dataset = CaptionDataset(caption_file="~/.dataset/flickr8k_captions.txt", image_folder="~/.dataset/flickr8k")
plot_image_caption_grid(dataset, num_images=5)