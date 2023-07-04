# %%
from typing import Dict

import torch
from data import build_data_loader, TokenEncoder, Vocabulary, build_extracted_data_loader
from model import Decoder, Encoder, inception_transform
from dataclasses import dataclass
from trainer import TrainerConfig, TrainerModel, Trainer, TrainerArgs

#%%
@dataclass
class CaptionModelConfig(TrainerConfig):
    optimizer: str = "Adam"
    lr_scheduler: str = "OneCycleLR"

    epochs: int = 100
    batch_size: int = 256
    max_train: int = None # Force loading all data into memory
    max_val: int = -1

    save_step: int = 100 
    print_step: int = 25
    plot_step: int = 25
    dashboard_logger: str = "tensorboard"

    # model config
    max_cap_len: int = 20

    vocab_size: int = -1

    hidden_size = 512
    full_feature_size = 1536
    pca_feature_size = 512
    embedding_size = 512
    attn_size = 512
    dropout = 0.1

    # data config
    json_file = "~/.dataset/dataset_flickr8k.json"
    image_folder = "~/.dataset/flickr8k"

class CaptionModel(TrainerModel):
    def __init__(self, vocab, config):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vocab = vocab
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder(
            config.vocab_size, config.hidden_size, config.full_feature_size, config.embedding_size, config.attn_size, config.dropout)\
                    .to(self.device)

    def forward(self, images, captions):
        scores = self.decoder(
            self.encoder(images),
            captions)
        return scores

    def train_step(self, batch, criterion):
        images, caps = batch
        images = images.to(self.device)
        caps = caps.to(self.device)

        captions_in = caps[:, :-1]  # N, T-1
        targets = caps[:, 1:]  # N, T-1
        logits = self(images, captions_in)
        logits = logits.permute(0, 2, 1) # N, T, V -> N, V, T
        loss = criterion(logits, targets)
        return {"model_outputs": logits}, {"loss": loss}
    
    eval_step = train_step

    @staticmethod
    def get_criterion():
        return torch.nn.CrossEntropyLoss()

    def get_data_loader(
        self, config, assets, is_eval, samples, verbose, num_gpus, rank=0
    ):  # pylint: disable=unused-argument
        train_loader, val_loader, test_loader = build_data_loader(
            json_file=config.json_file, 
            image_folder=config.image_folder,
            transform=inception_transform,
            token_processer=TokenEncoder(self.vocab, config.max_cap_len),
            batch_size=config.batch_size,
            max_train=config.max_train,
            max_val=config.max_val)
        if is_eval:
            return val_loader
        else:
            return train_loader


class ExtractedCaptionModel(TrainerModel):
    """CaptionModel use extracted features as input."""

    def __init__(self, config, pca=True):
        super().__init__()
        self.pca = pca
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        feature_size = config.pca_feature_size if pca else config.full_feature_size
        self.decoder = Decoder(
            config.vocab_size, config.hidden_size, feature_size, config.embedding_size, config.attn_size, config.dropout)\
                    .to(self.device)

    def forward(self, features, captions):
        scores = self.decoder(features, captions)
        return scores

    def train_step(self, batch, criterion):
        features, caps = batch
        features = features.to(self.device)
        caps = caps.to(self.device)
        captions_in = caps[:, :-1]  # N, T-1
        targets = caps[:, 1:]  # N, T-1
        logits = self(features, captions_in)
        logits = logits.permute(0, 2, 1) # N, T, V -> N, V, T
        loss = criterion(logits, targets)
        return {"model_outputs": logits}, {"loss": loss}

    eval_step = train_step

    @staticmethod
    def get_criterion():
        return torch.nn.CrossEntropyLoss()

    def get_data_loader(
        self, config, assets, is_eval, samples, verbose, num_gpus, rank=0
    ):  # pylint: disable=unused-argument
        if is_eval:
            val_loader = build_extracted_data_loader(
                    "flickr8k_val_features.h5", 
                    batch_size=config.batch_size,
                    max_sample=config.max_val,
                    pca=self.pca)
            return val_loader
        else:
            train_loader = build_extracted_data_loader(
                    "flickr8k_train_features.h5", 
                    shuffle=True,
                    batch_size=config.batch_size,
                    max_sample=config.max_train,
                    pca=self.pca)
            return train_loader

def main():
    # %%
    # init args and config
    train_args = TrainerArgs()

    vocab = Vocabulary.load("flickr8k_vocab.json")

    config = CaptionModelConfig(
        optimizer_params={ 'weight_decay': 1e-4, },
        # lr=4e-4,
        lr_scheduler_params={ 
                             'max_lr': 1e-3,
                             'div_factor': 10,
                             'final_div_factor': 4,
                             'epochs': 100,
                             'steps_per_epoch': 118,
                             'pct_start': 0.25,
                             },
        vocab_size = len(vocab),
    )

    # init the model from config
    # model = CaptionModel(vocab, config)
    model = ExtractedCaptionModel(config, pca=True)

    # init the trainer and 🚀
    trainer = Trainer(
        train_args,
        config,
        config.output_path,
        model=model,
        train_samples=model.get_data_loader(config, None, False, None, None, None),
        eval_samples=model.get_data_loader(config, None, True, None, None, None),
        parse_command_line_args=True,
    )

    # %%
    trainer.fit()


if __name__ == "__main__":
    main()
