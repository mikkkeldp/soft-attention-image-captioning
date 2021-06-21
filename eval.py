import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_validation_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNNWithAttention
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from tqdm import tqdm
import configparser 

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Load vocabulary wrapper
    with open(args["vocab_path"], 'rb') as f:
        vocab = pickle.load(f)
        
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((int(args["image_size"]), int(args["image_size"]))),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Build data loader
    data_loader = get_validation_loader(args["image_dir"], args["caption_path"], args["val_path"], vocab, 
                             transform, int(args["batch_size"]),
                             num_workers=int(args["num_workers"]))


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


    # Build models
    encoder = EncoderCNN(int(args["encoded_image_size"]), args["cnn"],device).eval().to(device)  # eval mode (batchnorm uses moving mean/variance)


    decoder = DecoderRNNWithAttention(int(args["embed_size"]), int(args["attention_size"]), int(args["hidden_size"]), len(vocab), encoder_size=int(args["encoder_size"]), glove = args["glove"], embedding_matrix = embedding_matrix).eval().to(device)


    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args["encoder_path"]))
    decoder.load_state_dict(torch.load(args["decoder_path"]))
    
    ground_truth = []
    predicted = []
    print("Evaluating on test set...")
   
    for _, (images, captions, ids) in tqdm(enumerate(data_loader)):
        
        # Set mini-batch dataset
        images = images.to(device)
        features = encoder(images,int(args["batch_size"]), int(args["encoder_size"]), ids)
        sampled_seq, _ = decoder.sample_beam_search(features, vocab, device)
        
        sampled_seq = sampled_seq[0][1:-1]
        captions = [c[1:-1] for c in captions[0]]

        ground_truth.append(captions)
        predicted.append(sampled_seq)
        
    print("BLEU-1: ", corpus_bleu(ground_truth, predicted, weights=(1, 0, 0, 0)))
    print("BLEU-2: ", corpus_bleu(ground_truth, predicted, weights=(0.5, 0.5, 0, 0)))
    print("BLEU-3: ", corpus_bleu(ground_truth, predicted, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0, 0)))
    print("BLEU-4: ", corpus_bleu(ground_truth, predicted))


    
if __name__ == '__main__':
    config = configparser.ConfigParser() 
    config.read("config.ini") 
    args = config["eval"]

    
    main(args)
