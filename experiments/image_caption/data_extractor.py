import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data import TokenEncoder, build_data_loader, build_extracted_data_loader, Vocabulary
from model import transform, Encoder, feature_extractor

vocab = Vocabulary.load("flickr8k_vocab.json")

if __name__ == "__main__":
    
    # %% Verify the extractor and data loader work as expected
    train_loader, val_loader, test_loader = build_data_loader(
        json_file="~/.dataset/dataset_flickr8k.json", 
        image_folder="~/.dataset/flickr8k",
        transform=transform, 
        token_processer=TokenEncoder(vocab, 20),
        train_shuffle=False,
        batch_size=128)


    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--extract_train', action='store_true', help='Extract features from train_loader')
    parser.add_argument('--extract_test', action='store_true', help='Extract features from test_loader')
    parser.add_argument('--extract_val', action='store_true', help='Extract features from val_loader')
    args = parser.parse_args()

    if args.extract_train:
        feature_extractor(train_loader, 'flickr8k_train_features.h5')
            
    if args.extract_test:
        feature_extractor(test_loader, 'flickr8k_test_features.h5')
            
    if args.extract_val:
        feature_extractor(val_loader, 'flickr8k_val_features.h5')

    # Verify the extracted features

    # extracted_train_loader = build_extracted_data_loader(
    #     "flickr8k_train_features.h5", pca=False, batch_size=128)

    # enc = Encoder()
    # for i, (extracted_data, original_data) in enumerate(zip(extracted_train_loader, train_loader)):
    #     orig_images, orig_captions = original_data
    #     extracted_features, extracted_captions = extracted_data
    #     assert torch.allclose(orig_captions, extracted_captions, rtol=1e-03, atol=1e-03)
    #     assert torch.allclose(enc(orig_images), extracted_features, rtol=1e-03, atol=1e-03)