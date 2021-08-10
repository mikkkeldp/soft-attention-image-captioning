import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle
from PIL import Image
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size, cnn, device):
        """Load the pretrained ResNet-101 and remove last layers."""
        super(EncoderCNN, self).__init__()
        self.device = device

        if cnn == "vgg":
            vgg = models.vgg19(pretrained=True)
            self.cnn = nn.Sequential(*list(vgg.features.children())[:43])

        elif cnn == "resnet":
            resnet = models.resnet101(pretrained=True)
            self.cnn = nn.Sequential(*list(resnet.children())[:-3])

        elif cnn == "inception":
            inception = models.inception_v3(pretrained=True)
            self.cnn = nn.Sequential(*list(inception.children())[:-5])


        if torch.cuda.is_available():
                self.cnn.cuda()
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))


    def forward(self, images, batch_size,encoder_size):
        """Extract feature vectors from input images."""
        with torch.no_grad(): #freeze layer weights
            feat_vecs = self.cnn(images)  # (batch_size, encoder_size, feature_map size, feature_map size)

        feat_vecs = self.adaptive_pool(feat_vecs)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        feat_vecs = feat_vecs.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)

        return feat_vecs


class Attention(nn.Module):
    def __init__(self, encoder_size, hidden_size, attention_size):
        """Set the layers"""
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_size, attention_size) # linear layer to transform encoded image
        self.hidden_att = nn.Linear(hidden_size, attention_size) # linear layer to transform previous hidden output
        self.full_att = nn.Linear(attention_size, 1) # linear layer to calculate pre-softmax values
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) # because dim0 is for batch

    def forward(self, encoder_out, hidden_out):
        """Generate attention encoded input from encoder output and previous hidden output"""
        # print("a: enc out: ", encoder_out.shape)
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_size)
        att2 = self.hidden_att(hidden_out)  # (batch_size, attention_size)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_size)
        return attention_weighted_encoding, alpha


class DecoderRNNWithAttention(nn.Module):
    def __init__(self, embed_size, attention_size, hidden_size, vocab_size, encoder_size, glove = False, embedding_matrix = None, max_seg_length=40):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNNWithAttention, self).__init__()

        self.attention = Attention(encoder_size=encoder_size, hidden_size=hidden_size, attention_size=attention_size)
        self.embed = nn.Embedding(vocab_size, embed_size)

        if glove == "True":
            self.embed.weight.requires_grad = False
            self.embed.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.lstmcell = nn.LSTMCell(embed_size+encoder_size, hidden_size, bias=True)

        self.init_hidden = nn.Linear(encoder_size, hidden_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_cell = nn.Linear(encoder_size, hidden_size)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(hidden_size, encoder_size)  # linear layer to create a sigmoid-activated gate according to paper
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()

        self.vocab_size = vocab_size
        self.max_seg_length = max_seg_length

    def init_weights(self):
        """Initialize the weights of learnable layers"""
        self.embed.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1,0.1)

    def init_hidden_state(self, encoder_out):
        """Mean of encoder output features as initial hidden and cell state"""


        mean_encoder_out = encoder_out.mean(dim=1)

        hidden = self.init_hidden(mean_encoder_out) # set hidden layer as mean of encoder values
        cell = self.init_cell(mean_encoder_out) # set lstm cell state as mean of encoder values
        return hidden, cell

    def forward(self, encoder_out, captions, lengths, device):
        """Decode image feature vectors and generates captions."""
        batch_size, encoder_size, vocab_size = encoder_out.size(0), encoder_out.size(-1), self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_size)
        num_pixels = encoder_out.size(1) #number of feature maps (196 for vgg) - 14x14
        embeddings = self.embed(captions) # (batch_size, max_caption_length, embed_size)

        hidden, cell = self.init_hidden_state(encoder_out) # (batch_size, hidden_size)
        lengths = [l - 1 for l in lengths]
        max_length = max(lengths)

        predictions = torch.zeros(batch_size, max_length, vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_length, num_pixels).to(device)
        for t in range(max_length):
            batch_size_t = sum([l > t for l in lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], hidden[:batch_size_t])
            gate = self.sigmoid(self.f_beta(hidden[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_size)
            attention_weighted_encoding = gate * attention_weighted_encoding
            hidden, cell = self.lstmcell(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (hidden[:batch_size_t], cell[:batch_size_t]))  # (batch_size_t, hidden_size)
            predictions[:batch_size_t, t, :] = self.fc(hidden)  # (batch_size_t, vocab_size)
            alphas[:batch_size_t, t, :] = alpha

        return predictions, captions, lengths, alphas


    def sample(self, encoder_out, vocab,batch_size,encoder_size, device):
        """Generate captions for given image features using greedy search."""

        batch_size = encoder_out.size(0)
        encoder_size = encoder_out.size(-1)
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_size)
     
        hidden, cell = self.init_hidden_state(encoder_out) # (batch_size, hidden_size)
        inputs = self.embed(torch.tensor([vocab('<start>')]).to(device)).repeat(batch_size, 1) #for single sample, bs = 1
        complete_seqs_loc = [] # selected feature maps
        sampled_ids = []
        betas=[]
        alphas = torch.zeros(batch_size, self.max_seg_length, 196).to(device)
        for t in range(self.max_seg_length):

            attention_weighted_encoding, alpha = self.attention(encoder_out, hidden)
            _, max_idx = torch.max(alpha, dim=1)
            complete_seqs_loc.append(max_idx.item())
            beta = self.f_beta(hidden)
            betas.append(beta)
            gate = self.sigmoid(beta)
            attention_weighted_encoding = gate * attention_weighted_encoding
            hidden, cell = self.lstmcell(
                torch.cat([inputs, attention_weighted_encoding], dim=1),
                (hidden, cell))

            _, predicted = self.fc(hidden).max(1) #get index of max node in softmax (corresponds to word ID)
            alphas[:1, t, :] = alpha
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)

        sampled_ids = torch.stack(sampled_ids, 1).tolist() # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = [[vocab('<start>')]+s for s in sampled_ids] #add start token to prediction

        return sampled_ids, complete_seqs_loc, alphas


    def sample_beam_search(self, encoder_out, vocab, device, beam_size=4):
        k = beam_size
        vocab_size = len(vocab)
        encoder_size = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, -1, encoder_size)
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(k, num_pixels, encoder_size)  # (k, num_pixels, encoder_dim)
        k_prev_words = torch.LongTensor([[vocab('<start>')]] * k).to(device)  # (k, 1)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        complete_seqs = list()
        complete_seqs_scores = list()
        complete_seqs_loc = []
        hidden, cell = self.init_hidden_state(encoder_out)
        step = 1
        while True:
            embeddings = self.embed(k_prev_words).squeeze(1)
            awe, alpha = self.attention(encoder_out, hidden)

            complete_seqs_loc.append(torch.max(alpha,dim=1).indices[0].item())
            gate = self.sigmoid(self.f_beta(hidden))
            awe = gate * awe
            hidden, cell = self.lstmcell(torch.cat([embeddings, awe], dim=1), (hidden, cell))

            scores = self.fc(hidden)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)  # (s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != vocab('<end>')]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            hidden = hidden[prev_word_inds[incomplete_inds].long()]
            cell = cell[prev_word_inds[incomplete_inds].long()]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > self.max_seg_length:
                break
            step += 1

        try:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
        except: 
            print("\nNo caption flag!")
            return [[1, 4, 88, 638, 4, 639, 46, 4, 635, 18, 2]], None

        seq = complete_seqs[i]
        return [seq] , complete_seqs_loc
