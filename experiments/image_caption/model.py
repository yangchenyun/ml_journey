# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import timm

from data import build_data_loader, TokenEncoder, vocab


# %% Create the inceptionv4 model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

model = timm.create_model('inception_v4', pretrained=True)
model.eval()  # only in eval mode

config = resolve_data_config({}, model=model)
transform = create_transform(**config)
inception_transform = transform

def restore_img(img, config):
    """
    img: np.array
    """
    return img * np.array(config['mean']) + np.array(config['std'])

train_loader, valid_loader, test_loader = build_data_loader(
    json_file="~/.dataset/dataset_flickr8k.json", 
    image_folder="~/.dataset/flickr8k",
    transform=transform, 
    token_processer=TokenEncoder(vocab, 20),
    batch_size=8)

# %% Test the model's features
def load_inception_classes():
    import urllib
    url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    urllib.request.urlretrieve(url, filename) 
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories
categories = load_inception_classes()

if False:
    batch = next(iter(train_loader))
    # NOTE: would only use features in our models
    features = model.forward_features(batch[0])
    probabilities = torch.nn.functional.softmax(model.forward_head(features), 1)

    # Print top categories per image
    for i in range(probabilities.size(0)):
        top5_prob, top5_catid = torch.topk(probabilities[i], 5) # get the top 5 probabilities and their corresponding class ids
        for j in range(top5_prob.size(0)):
            print(categories[top5_catid[j]], top5_prob[j].item()) # print the class name and probability

        img = batch[0][i].permute(1, 2, 0).numpy()
        img = restore_img(img, config)
        plt.imshow(img)
        plt.show()

# %% 
# Build the LSTM component

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList([torch.nn.LSTMCell(input_size, hidden_size)])
        for i in range(num_layers - 1):
            self.layers.append(torch.nn.LSTMCell(hidden_size, hidden_size))

    def forward(self, x, h0, c0):
        """
        In the forward 
        """
        T, N, D = x.shape
        h = [h0[i] for i in range(self.num_layers)]
        c = [c0[i] for i in range(self.num_layers)]

        # NOTE: compute the forward pass layer by layer
        for t in range(T):
            for i in range(self.num_layers):
                h[i], c[i] = self.layers[i](x[t], (h[i], c[i]))
                x[t] = h[i]

        return h[-1]

lstm = LSTM(512, 512, 2)
 
# Generate random torch tensors to test the lstm layer
if False:
    x = torch.randn(10, 8, 512) # T, N, D
    h0 = torch.randn(2, 8, 512) # L, N, D
    c0 = torch.randn(2, 8, 512)
    h = lstm(x, h0, c0)
    print(h.shape)

# %%
# Build attention layer

class Attention(torch.nn.Module):
    """
    Attention layer computes a projection matrix, which is used to samples the image features spaces.
    The last hidden state and the current encoded image computes the encoded images as input.

    Fn_Attn(h_t-1, token_t) -> attention matrix
    [1536, 8, 8] * attention matrix -> summing out -> [1536, 1, 1]

    Use a linear relu linear layer to emulate the computation step
    """
    def __init__(self, hidden_size, enc_dim, attn_dim):
        super(Attention, self).__init__()
        self.attn_token = torch.nn.Linear(hidden_size, attn_dim)
        self.attn_features = torch.nn.Linear(enc_dim, attn_dim)
        self.attn_full = torch.nn.Linear(attn_dim, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, h, features):
        """
        h: (N, hidden_size)
        features: (N, n_pixels, n_features)
        """
        attn_h = self.attn_token(h) # N, attn_dim
        attn_img = self.attn_features(features) # N, n_pixels, attn_dim
        attn_weights = self.attn_full(self.relu(attn_h.unsqueeze(1) + attn_img)).squeeze(2) # N, n_pixels
        logits = self.softmax(attn_weights) # N, n_pixels
        attn_weighted_enc = (features * logits.unsqueeze(2)).sum(1) # N, n_features
        return attn_weighted_enc

attn = Attention(512, 128, 8)
h = torch.randn(8, 512)
features = torch.randn(8, 128)
attn_m = attn(h, features)

# %%
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Image encoder
        self.encoder = timm.create_model('inception_v4', features_only=True, pretrained=True)
        self.encoder.eval()  # Only use in eval mode
        for p in self.encoder.parameters(): p.requires_grad = False
        # TODO: use AvgPool explicit control the enc_dim

    def forward(self, images):
        N = images.shape[0]
        features = self.encoder(images) # N, D, W, H
        enc = features[-1] # Use last feature, [N, 1536, 8, 8])
        enc = enc.reshape(N, -1, 8*8).permute(0, 2, 1) # N, W*H, D
        return enc


class Decoder(torch.nn.Module):
    def __init__(self, n_embd, embd_dim, enc_dim, attn_dim, hidden_size, dropout_p):
        super(Decoder, self).__init__()

        self.n_embd = n_embd
        self.embd_dim = embd_dim
        self.enc_dim = enc_dim
        self.attn_dim = attn_dim
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self._null = vocab["<null>"]
        self._start = vocab["<start>"]
        self._end = vocab["<end>"]

        # Word embedding encoder
        self.embd = torch.nn.Embedding(n_embd, embd_dim)

        # Attention
        self.attn = Attention(hidden_size, enc_dim, attn_dim)

        # LSTM cell
        self.rnn = torch.nn.LSTMCell(embd_dim + enc_dim, hidden_size)

        # Linear layer to project to embedding
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc = torch.nn.Linear(hidden_size, n_embd)
    
    def init_cell(self, features, cell_size):
        """
        Use gaussian random sampler to sample the encoded feature space to the cell_size.
        :param features: (N, W*H, D)
        :param cell_size: int
        :return: (N, cell_size, D)
        """
        N, WH, D = features.shape
        features = features.reshape(N, -1)
        sampled_features = features @ torch.randn(WH*D, cell_size)
        return sampled_features

    def forward(self, features, captions):
        """
        :param: features, (N, pixels, C)
        :param: captions, (N, T)
        """
        N, T = captions.shape

        embd_tokens = self.embd(captions).permute(1, 0, 2) # T, N, W

        # h0, c0, (N, H)
        h = self.init_cell(features, self.hidden_size)
        c = self.init_cell(features, self.hidden_size)
        h_out = []

        for t in range(T):
            x_t = torch.cat((self.attn(h, features), embd_tokens[t]), dim=1)
            h, c = self.rnn(x_t, (h, c))
            h_out.append(h)

        h_out = torch.stack(h_out) # T, N, H
        preds = self.fc(self.dropout(h_out)).permute(1, 0, 2) # N, T, E

        return preds
    
    def sample(self, features, max_length):
        with torch.no_grad():
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            word = self._start * np.ones(N, dtype=np.int32)
            word = torch.LongTensor(word)

            # h0, c0, (N, H)
            h = self.init_cell(features, self.hidden_size)
            c = self.init_cell(features, self.hidden_size)

            for t in range(max_length):
                embd_token = self.embd(word)
                x_t = torch.cat((self.attn(h, features), embd_token), dim=1)
                h, c = self.rnn(x_t, (h, c))
                output_logits = self.fc(self.dropout(h)) # 1, N, E

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.numpy()

            return captions


dec = Decoder(len(vocab), 512, 1536, 512, 512, 0.5)
if False:
    total_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print(f"Number of word embedding encoder parameters: {sum(p.numel() for p in dec.embd.parameters() if p.requires_grad)}")
    print(f"Number of attention layer parameters: {sum(p.numel() for p in dec.attn.parameters() if p.requires_grad)}")
    print(f"Number of LSTM cell parameters: {sum(p.numel() for p in dec.rnn.parameters() if p.requires_grad)}")
    print(f"Number of linear layer parameters: {sum(p.numel() for p in dec.fc.parameters() if p.requires_grad)}")

# %%
enc = Encoder()
images, captions = next(iter(train_loader))
preds = dec(enc(images), captions)
word_idxs = torch.argmax(
    torch.nn.functional.softmax(preds, dim=2), dim=2)
 
def decode_caption(word_idxs, vocab):
    """
    Greedy algorithm to find the index in the vocab and return the text.
    """
    captions = []
    for pred in word_idxs:
        caption = []
        for idx in pred:
            if idx == vocab['<end>']:
                break
            caption.append(vocab.idx2word[idx.item()])
        captions.append(' '.join(caption))
    return captions

captions = decode_caption(word_idxs, vocab)
# print(captions[0])
print(decode_caption(dec.sample(enc(images), 20), vocab))