# %%
import itertools
import torch
from data import build_data_loader, TokenEncoder, vocab
from model import Decoder, inception_transform

lr=4e-4
momentum=0.9
weight_decay=1e-4
vocab_size = len(vocab)
hidden_size = 512
feature_size = 1536
embedding_size = 512
num_layers = 512
dropout = 0.5
json_file = "~/.dataset/dataset_flickr8k.json"
image_folder = "~/.dataset/flickr8k"
max_cap_len = 20

# %%
def epoch(model, train_data_loader, optimizer):
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    train_data_loader = itertools.islice(train_data_loader, 10)

    for i, (imgs, caps) in enumerate(train_data_loader):
        captions_in = caps[:, :-1]  # N, T-1
        targets = caps[:, 1:]  # N, T-1

        scores = model((imgs, captions_in), True)

        # Exclude <pad> token from loss computation
        pad_mask = targets == vocab['<pad>'] # N, T-1
        scores = scores.permute(0, 2, 1) # N, C, T-1
        scores = scores.masked_fill(pad_mask.unsqueeze(1), float('-inf'))

        loss = loss_fn(scores, targets)
        total_loss += loss.item()
        num_batches += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / num_batches
    print(f"Average loss: {avg_loss}")


def train(vocab, train_loader, valid_loader, test_loader, n_epoch=10):
    model = Decoder(vocab_size, hidden_size, feature_size, embedding_size, num_layers, dropout)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for _ in range(n_epoch):
        epoch(model, train_loader, optimizer)

train_transform = inception_transform
train_loader, valid_loader, test_loader = build_data_loader(
    json_file=json_file, 
    image_folder=image_folder,
    transform=train_transform,
    token_processer=TokenEncoder(vocab, max_cap_len),
    batch_size=1)

# %%
train(vocab, train_loader, valid_loader, test_loader, n_epoch=10)