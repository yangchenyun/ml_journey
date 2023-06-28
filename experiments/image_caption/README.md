# Image Caption Project Plan

Goal: train a model, when given an image, would returns a sentence to describe the image.

Reference paper: 
https://arxiv.org/abs/1502.03044

## Data
Microsoft COCO 2017 with the original split of train, validation, test.

The dataset could be assessed via `pytorch.vision`.

Tasks
- [x] Read and understand the structure of data and its labels
- [x] Download three split of dataset
- [x] Prepare data batch, each data entry should contain a pair of image and captions

Data preparation
- Downsampling with PCA projection.

Data augmentation for images
- [x] Normalize the data with subtracting mean and variance.

Data preparation
- [x] Use one-hot encoding for the character sequence data, specially treatment of `<END>` and `<START>` token.
- [x] Pad token with `<NULL>` for the same length.

How to not compute loss for `<NULL>` tokens?

## Model Architecture
Would reuse pre-trained image models for until the fc layer as a encoder for image.
For the language model, would use GRU with 2-3 layers, pick the parameters to fit into the GPU.
Using linear layer for the encoder and decoder.

The sequence model is a many-to-many architecture, it is conditioned on the encoded image vector and the initial `<START>` token.

The initial hidden state is concatenation of the image layer + rnn hidden state; the forward pass would compute the next hidden state at the same dimension.
 
For decoding, we will start with a greedy decoder first and later could switch to beam search decoder.

Questions
- Activation function choices
- How to deal with the image representation vector, should we create residual connection between each itertion?
- Does beam search could be applied during training as well?

Tasks
- [x] Setup and validate the pre-trained model 
- [x] Connect image model with GRU, confirm the model would emit sequence of logits
- [x] Create the attention component to compute the encoded images
- [x] Validate model output, use greedy decoder to sample the initial model.
- [x] Validate the connector between data loader and model. Sampling the models with a few batches of data.

## Training configuration
Batch
- Each batch would contain N images and N captions, each caption is a sequence of (T,O). `T` would be different for each data entry.
- During training, we use teacher forcing to summarize losses token by token.

Unknowns and questions:
- Scheduled learning for the model later on?
- How to ensure the length of model output is the same as the labelled data? How to penalize different length output?

Setup the training
- Optimizer, use RMSProp or Adam
- Softmax loss between the generated sequence and labelled sequence

Parameter initialization 
- Follow the uniform sqrt(k) configuration for weights
- Initialize the forget biases to be 1

Hyperparameter
- Learning rate, learning schedule and weight decay could be copied from popular papers 
- Dropout hyper parameter

Tasks
- [ ] Setup the training epoch with pytorch, including checkpoints (would be used for sampling)
- [ ] Setup with `wandb` to watch for training progress
- [ ] Setup with tensorboard to watch for progress
- [ ] Train the models on 1% of data and try to overfit the model

Interpretation and understanding
- How to interprete the image to hidden layer, input to hidden layer weights (over time)?