#%%
import os
import random
import torch
import torchvision

from PIL import Image
import matplotlib.pyplot as plt

# %%
# Dataset Loader
class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, caption_file, image_folder, transform=None, caption_transform=None):
        self.image_folder = os.path.expanduser(image_folder)
        self.caption_file = os.path.expanduser(caption_file)
        self.transform = transform
        self.caption_transform = caption_transform
        
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
        return len(self.captions)
    
    def __str__(self):
        return f"CaptionDataset(size={len(self)})"

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            dataset = CaptionDataset(
                self.caption_file, self.image_folder,
                self.transform, self.caption_transform)
            dataset.image_paths = [self.image_paths[ii] for ii in indices]
            dataset.captions = [self.captions[ii] for ii in indices]
            return dataset
        else:
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

# %%
# Vocabulary Encoding

from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_freq = Counter()
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        self.word_freq[word] += 1
            
    def __len__(self):
        return len(self.word2idx)

    def __str__(self):
        return f"Vocabulary(size={len(self)})"

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.idx2word[idx]
        elif isinstance(idx, str):
            return self.word2idx[idx]
        else:
            raise ValueError(f"Invalid argument type: {type(idx)}")

    def to_indices(self, tokens):
        return [self.word2idx[token] for token in tokens]

    def to_tokens(self, indices):
        return [self.idx2word[index] for index in indices]

    @staticmethod
    def from_corpus(corpus, tokenize: callable):
        vocab = Vocabulary()
        for sentence in corpus:
            for token in tokenize(sentence):
                vocab.add_word(token)
        return vocab

with_start_end = lambda tokens: ['<start>'] + tokens + ['<end>']
word_tokenizer = lambda s: with_start_end(s.split())
char_tokenizer = lambda s: with_start_end(list(s))

vocab = Vocabulary.from_corpus(dataset.captions, char_tokenizer)
 
plt.figure(figsize=(10, 5))
plt.bar(vocab.word_freq.keys(), vocab.word_freq.values())
plt.title("Vocabulary Word Frequency")
plt.xlabel("Words")
plt.ylabel("Frequency (log)")
plt.yscale('log')
plt.xticks()
plt.show()

# %%
# Build data loader to support train / test / validate split and batch size


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # NOTE: Depends on the pre-trained model 
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
caption_transform = word_tokenizer
dataset = CaptionDataset(
    caption_file="~/.dataset/flickr8k_captions.txt", 
    image_folder="~/.dataset/flickr8k",
    transform=transform,
    caption_transform=caption_transform,
)

def build_data_loader(dataset, batch_size=128):
    n_data = len(dataset)
    n_train = int(n_data * 0.8)
    n_valid = int(n_data * 0.9)
    n_test = n_data

    train_dataset = dataset[:n_train]
    valid_dataset = dataset[n_train:n_valid]
    test_dataset = dataset[n_valid:n_test]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               num_workers=8,
                                               shuffle=True,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader