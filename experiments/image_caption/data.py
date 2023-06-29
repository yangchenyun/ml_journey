#%%
import json
import os
import random
import copy
import torch
import torchvision

from PIL import Image
import matplotlib.pyplot as plt

# %%
# Dataset Loader


class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 json_file,
                 image_folder,
                 transform=None,
                 token_processer=None,
                 split=None):
        """
        A PyTorch Dataset class to load image-caption pairs from the Andrew Karpathy Image Caption File.

        Args:
            json_file (str): Path to the Andrew Karpathy Image Caption File.
            image_folder (str): Root folder to load images.
            transform (callable, optional): Optional transform to be applied on a sample.

        """

        json_file = os.path.expanduser(json_file)
        with open(json_file, 'r') as f:
            data = json.load(f)['images']

        if split is not None:
            assert split in {'train', 'val', 'test'}
            self.data = [item for item in data if item['split'] == split]
        else:
            self.data = data

        self.image_folder = os.path.expanduser(image_folder)
        self.transform = transform
        self.token_processer = token_processer
        
        self.image_filenames = []
        self.raw_captions = []
        self.token_captions = []

        for item in self.data:
            for sent in item['sentences']:
                self.image_filenames.append(item['filename'])
                self.raw_captions.append(sent['raw'])
                self.token_captions.append(sent['tokens'])

        assert len(self.image_filenames) == len(self.raw_captions)
        assert len(self.image_filenames) == len(self.token_captions)
                
    def __len__(self):
        return len(self.raw_captions)
    
    def __str__(self):
        return f"CaptionDataset(size={len(self)})"

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            dataset = copy.deepcopy(self)
            dataset.image_filenames = [dataset.image_filenames[ii] for ii in indices]
            dataset.raw_captions = [dataset.raw_captions[ii] for ii in indices]
            dataset.token_captions = [dataset.token_captions[ii] for ii in indices]
            return dataset
        else:
            image_path = self.image_filenames[idx]
            tokens = self.token_captions[idx]

            image = Image.open(
                os.path.join(self.image_folder, image_path)).convert('RGB')

            if self.transform is not None:
                image = self.transform(image)

            if self.token_processer is not None:
                tokens = self.token_processer(tokens)

            return image, tokens

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
# dataset = CaptionDataset(json_file="~/.dataset/dataset_flickr8k.json", image_folder="~/.dataset/flickr8k", split='train')
# plot_image_caption_grid(dataset, num_images=5)

# %%
# Vocabulary Encoding

from collections import Counter, defaultdict

class Vocabulary:
    def __init__(self, default_idx=None, default_word=None):
        self.word2idx = defaultdict(lambda: default_idx)  # NOTE:  <unk> index
        self.idx2word = defaultdict(lambda: default_word)
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

    def get(self, key, other_value):
        if isinstance(key, str):
            return self.word2idx.get(key, other_value)
        elif isinstance(key, int):
            return self.idx2word.get(key, other_value)
        else:
            raise ValueError(f"Invalid argument type: {type(key)}")

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
    def from_corpus(corpus, min_word_freq=1):
        """
        :param special_tokens, special <start>, <end>, <null> tokens
        :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
        """
        vocab = Vocabulary(default_word='<unk>', default_idx=3)

        special_tokens=['<null>', '<start>', '<end>', '<unk>']
        for token in special_tokens:
            vocab.add_word(token)

        for sentence in corpus:
            for token in sentence:
                vocab.word_freq[token] += 1

        for word, freq in vocab.word_freq.items():
            if freq >= min_word_freq:
                vocab.add_word(word)

        return vocab

    def save(self, output_file):
        with open(output_file, 'w') as f:
            json.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word, 'idx': self.idx, 'word_freq': self.word_freq}, f)

    def load(self, input_file):
        with open(input_file, 'r') as f:
            data = json.load(f)

        self.word2idx = defaultdict(lambda: data['word2idx']['<unk>'], data['word2idx'])
        self.idx2word = defaultdict(lambda: '<unk>', {int(k): v for k, v in data['idx2word'].items()})
        self.idx = data['idx']
        self.word_freq = Counter(data['word_freq'])



# %%
# vocab = Vocabulary.from_corpus(dataset.token_captions, min_word_freq=3)
vocab = Vocabulary()
vocab.load("flickr8k_vocab.json")
 
# plt.figure(figsize=(10, 5))
# plt.bar(vocab.word_freq.keys(), vocab.word_freq.values())
# plt.title("Vocabulary Word Frequency")
# plt.xlabel("Words")
# plt.ylabel("Frequency (log)")
# plt.yscale('log')
# plt.xticks()
# plt.show()

# %%
# Build data loader to support train / test / validate split and batch size

def build_data_loader(json_file, image_folder, 
                      transform=None, 
                      token_processer=None,
                      batch_size=128,
                      max_train=None,
                      max_val=None):
    train_dataset = CaptionDataset(
        json_file=json_file, 
        image_folder=image_folder, 
        transform=transform,
        token_processer=token_processer,
        split='train')
    val_dataset = CaptionDataset(
        json_file=json_file, 
        image_folder=image_folder, 
        transform=transform,
        token_processer=token_processer,
        split='val')
    test_dataset = CaptionDataset(
        json_file=json_file, 
        image_folder=image_folder, 
        transform=transform,
        token_processer=token_processer,
        split='test')

    if max_train:
        train_dataset = train_dataset[:max_train]
    if max_val:
        val_dataset = val_dataset[:max_val]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

class TokenEncoder:
    def __init__(self, vocab, max_len=30):
        self.vocab = vocab
        self.max_len = max_len

    def __call__(self, tokens_caption):
        """
        Process the token for training, 
        unify the length with truncation and padding; 
        replace word with low frequency with <unk>.
        Add <start> and <end> tokens
        Return encoded tokens
        """
        tokens = tokens_caption[:self.max_len]
        # Encode captions
        enc_c = ([self.vocab['<start>']] +
                 [self.vocab.get(word, self.vocab['<unk>']) for word in tokens] +
                 [self.vocab['<end>']] +
                 [self.vocab['<null>']] * (self.max_len - len(tokens)))
        return torch.tensor(enc_c)

vocab = Vocabulary()
vocab.load("flickr8k_vocab.json")

# np.percentile(([len(s) for s in dataset.token_captions]), 98)

if False:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        # NOTE: Depends on the pre-trained model 
        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader, valid_loader, test_loader = build_data_loader(
        json_file="~/.dataset/dataset_flickr8k.json", 
        image_folder="~/.dataset/flickr8k",
        transform=transform, 
        token_processer=TokenEncoder(vocab, 20),
        batch_size=8)