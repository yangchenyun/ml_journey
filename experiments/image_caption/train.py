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
    lr_scheduler: str = "StepLR"

    epochs: int = 100
    batch_size: int = 8
    max_train: int = 64
    max_val: int = 16

    print_step: int = 10
    save_step: int = 2000
    plot_step: int = 5
    dashboard_logger: str = "tensorboard"

    # model config
    max_cap_len: int = 20

    vocab_size: int = -1

    hidden_size = 512
    feature_size = 512 # use pca features
    embedding_size = 512
    attn_size = 512
    dropout = 0.1

    # data config
    json_file = "~/.dataset/dataset_flickr8k.json"
    image_folder = "~/.dataset/flickr8k"

class CaptionModel(TrainerModel):
    def __init__(self, vocab, config):
        super().__init__()

        self.vocab = vocab
        self.encoder = Encoder()
        self.decoder = Decoder(
            config.vocab_size, config.hidden_size, config.feature_size, config.embedding_size, config.attn_size, config.dropout)

    def forward(self, images, captions):
        scores = self.decoder(
            self.encoder(images),
            captions)
        return scores

    def train_step(self, batch, criterion):
        images, caps = batch
        captions_in = caps[:, :-1]  # N, T-1
        targets = caps[:, 1:]  # N, T-1
        logits = self(images, captions_in)
        logits = logits.permute(0, 2, 1) # N, T, V -> N, V, T
        loss = criterion(logits, targets)
        return {"model_outputs": logits}, {"loss": loss}

    def eval_step(self, batch, criterion):
        images, caps = batch
        captions_in = caps[:, :-1]  # N, T-1
        targets = caps[:, 1:]  # N, T-1
        logits = self(images, captions_in)
        logits = logits.permute(0, 2, 1) # N, T, V -> N, V, T
        loss = criterion(logits, targets)
        return {"model_outputs": logits}, {"loss": loss}

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

    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder(
            config.vocab_size, config.hidden_size, config.feature_size, config.embedding_size, config.attn_size, config.dropout)

    def forward(self, features, captions):
        scores = self.decoder(features, captions)
        return scores

    def train_step(self, batch, criterion):
        features, caps = batch
        captions_in = caps[:, :-1]  # N, T-1
        targets = caps[:, 1:]  # N, T-1
        logits = self(features, captions_in)
        logits = logits.permute(0, 2, 1) # N, T, V -> N, V, T
        loss = criterion(logits, targets)
        return {"model_outputs": logits}, {"loss": loss}

    def eval_step(self, batch, criterion):
        features, caps = batch
        captions_in = caps[:, :-1]  # N, T-1
        targets = caps[:, 1:]  # N, T-1
        logits = self(features, captions_in)
        logits = logits.permute(0, 2, 1) # N, T, V -> N, V, T
        loss = criterion(logits, targets)
        return {"model_outputs": logits}, {"loss": loss}

    @staticmethod
    def get_criterion():
        return torch.nn.CrossEntropyLoss()

    def get_data_loader(
        self, config, assets, is_eval, samples, verbose, num_gpus, rank=0
    ):  # pylint: disable=unused-argument
        if is_eval:
            val_loader = build_extracted_data_loader("flickr8k_val_features.h5", pca=True)
            return val_loader
        else:
            train_loader = build_extracted_data_loader("flickr8k_train_features.h5", pca=True, shuffle=True)
            return train_loader

def main():
    # %%
    # init args and config
    train_args = TrainerArgs()

    vocab = Vocabulary.load("flickr8k_vocab.json")

    config = CaptionModelConfig(
        lr=1e-3,
        optimizer_params={ 'weight_decay': 1e-4, },
        lr_scheduler_params={ 'step_size': 20, 'gamma': 0.1 },
        vocab_size = len(vocab),
    )

    # init the model from config
    # model = CaptionModel(vocab, config)
    model = ExtractedCaptionModel(config)

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