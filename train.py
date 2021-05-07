import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_train_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNNWithAttention
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import configparser 

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    # if path does not exist, create dir
    if not os.path.exists(args["model_path"]):
        os.makedirs(args["model_path"])
    
    # resize image to desired shape for CNN, normalize rgb values and randomize flip
    transform = transforms.Compose([ 
        transforms.Resize((int(args["image_size"]), int(args["image_size"]))),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # load vocab
    with open(args["vocab_path"], 'rb') as f:
        vocab = pickle.load(f)

    # create data_loader
    data_loader = get_train_loader(args["image_dir"], args["caption_path"], args["train_path"], vocab, 
                             transform, int(args["batch_size"]),
                             shuffle=True, num_workers=int(args["num_workers"])) 
    embedding_matrix = None
    if args["glove"] == "True":
 
        #glove embeddings
        embeddings_index = {} 
        f = open(os.path.join("", 'glove.6B.200d.txt'), encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        # represents each word in vocab through a 200D vector
        embedding_dim = 200
        i = 0
        embedding_matrix = np.zeros((len(vocab), embedding_dim))
        for word in vocab.word2idx:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector	
                i+=1
  
    # init the encoder and decoder models
    encoder = EncoderCNN(int(args["encoded_image_size"]), args["cnn"]).to(device)
    decoder = DecoderRNNWithAttention(int(args["embed_size"]), int(args["attention_size"]), int(args["hidden_size"]), len(vocab), encoder_size=int(args["encoder_size"]), glove = args["glove"], embedding_matrix = embedding_matrix).to(device)

    # init loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    params = list(decoder.parameters()) + list(encoder.adaptive_pool.parameters())
    optimizer = torch.optim.Adam(params, lr=float(args["learning_rate"]))

    losses = []
    not_improved = 0
    # Train the models
    total_step = len(data_loader)
    for epoch in range(int(args["num_epochs"])):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)

            # Forward, backward and optimize
            features = encoder(images)
            # print("F: ", features.shape)
            scores, captions, lengths, alphas = decoder(features, captions, lengths, device)
            
            targets = captions[:, 1:] #remove start token
            # Remove padded words to calculate score
            targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]
            scores = pack_padded_sequence(scores, lengths, batch_first=True)[0]

            # cross entropy loss and doubly stochastic attention regularization
            loss = criterion(scores, targets)
            loss += 1.0 * ((1 - alphas.sum(dim=1))**2).mean()
            
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if (i+1) % int(args["log_step"]) == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch+1, int(args["num_epochs"]), i+1, total_step, loss.item(), np.exp(loss.item())))

            # Save the model checkpoints
            if (i+1 + epoch*total_step) % int(args["save_step"]) == 0:
                losses.append(loss.item())
                if len(losses>2):
                    if losses[-1] > losses[-2]:
                        not_improved+= 1
                        if not_improved == 3:
                            print("Early stopping...")
                            break
                    else:
                        torch.save(decoder.state_dict(), os.path.join(
                            args["model_path"], 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                        torch.save(encoder.state_dict(), os.path.join(
                            args["model_path"], 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


if __name__ == '__main__':
    config = configparser.ConfigParser() 
    config.read("config.ini") 
    args = config["train"]
    
    main(args)
