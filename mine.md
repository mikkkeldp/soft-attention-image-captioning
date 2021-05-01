# Image captioning with attention mechanism

## Implementation

Implementation of Hard Attention for image captioning described in [Show, Attend and Tell](https://arxiv.org/abs/1502.03044). For our implementation we chose to make use of ResNet instead of the proposed VGG-16 model in the encoder stage. 


## Environment setup
Setup conda environment and install required packages.
```
conda create --name env
conda activate env
conda install pytorch pydot keras Pillow nltk
```

## Running instructions

#### Build vocabulary
Build vocabulary by tokenizing the words within in the Flickr8k training set. 
```
python build_vocab.py --caption_path <string> --train_path <string> --vocab_path <string> --threshold <int>
```
All command line arguments are optional. 
<!-- **Command line arguments**:
- *caption_path*: Path to captions file. (default: ./dataset/Flickr8k.token.txt)
- *train_path*: Path to training images. (default: ./dataset/Flickr_8k.trainImages.txt)
- *vocab_path*: Path to where vocab file should be saved. (default: ./data/vocab.pkl)
- *threshold*: The minimum word frequency for a word to be included in the vocabulary. (default: 1) -->

#### Training model
Train the model with the following command:
```
python train.py --model_path <string> --vocab_path <string> --image_dir <string> --caption_path <string> --train_path <string> --image_size  <int> --log_step <int> --save_step <int> --embed_size <int> --encode_image_size <int> --attention_size <int> --hidden_size <int> --num_peochs <int> --batch_size <int> --num_workers <int> --learning_rate <float> 
```
All command line arguments are optional. 
<!-- **Command line arguments**:
- *model_path*: Path for saving trained models
- *vocab_path*: Path for vocabulary wrapper
- *image_dir*: Directory for images
- *caption_path*: Path for caption file
- *train_path*: Path for train split file
- *image_size*: Input image size
- *log_step*: Step size for printing log info
- *save_step*: Step size for saving trained models
- *embed_size*: Dimension of word embedding vectors
- *encoded_image_size*: Dimension of encoded image
- *attention_size*: Dimension of attention layers
- *hidden_size*: Dimension of lstm hidden states
- *num_epochs*: Number of epochs 
- *batch_size*: Batch size 
- *num_workers*: Number of parallel workers 
- *learning_rate*: Learning rate of model -->

#### Evaluate model
Evaluate the model on the Flickr8k testing set
```
python eval.py --encoder_path <string> --decoder_path <string> --vocab_path <string> --image_dir <string> --caption_path <string> --val_path <string> --image_size <int> --embed_size <int> --encoded_image_size <int> --attention_size <int> --hidden_size <int> --batch_size <int> --num_workers <int>
```

All command line arguments are optional. 

## Overview of Show attend and tell approach
For the task of image captioning, a model is required that can predict the words of the caption in a correct sequence given the image. This can be modeled as finding the caption that maximizes the following log probability:    


<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;logp(S|I) = \sum^N_{t=0}log\;p(S_t|I,S_0,S_1,...,S_{t-1})\;(Eqn.\;1)"  height="45"/>
</p>
where S is the caption, I the image and S<sub>t</sub>, the word at time *t*.

The probability of a word depends on the previously generated words and the image, hence the conditioning on these variables in the equation. The training data consists of various images with multiple descriptions/interpretations manually produced by humans. The training phase involves finding the parameters in the model that maximizes the probability of captions given the image in the training set.

RNN's provide a method of conditioning on the previous variables using a fixed sized hidden vector. This hidden vector is then used to predict the next word just like a feed forward neural network.



<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;p(S_t|I,S_0,S_1,...,S_{t-1}) \approx p(S_t|h_t) \;(Eqn.\;2)"  height="17"/>
</p>

In this equation you can see that the hidden state (h<sub>t</sub>) represents the previously generated words S<sub>0</sub>, ..., S<sub>t-1</sub>.

Equation 2 is used to model the probability distribution over all words in the vocabulary using a fully connected layer followed by a softmax:


<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space; p(S_t|h_t) = softmax(L_hh_t+L_II)"  height="17"/>
</p>

where L<sub>h</sub> and L<sub>I</sub> are weight matrices of the fully connected layer with inputs taken as one concatenated vector from h<sub>t</sub> and I. From this distribution, we select the word with the maximum probability as the next word in the caption. Now at the step t+1, the conditioning on the previously generated words should also involve this newly generated word(St). But the RNN hidden state (h<sub>t</sub>) is conditioned on S<sub>0</sub>, ..., S<sub>t-1</sub>. So S<sub>t</sub> is then combined with h<sub>t</sub> through a linear layer followed by a non-linearity to produce h<sub>t+1</sub> which is conditioned on S<sub>0</sub>, ..., S<sub>t</sub>.


<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space; h_{t+1} = tanh(W_hh_t + W_sS_t)\;(Eqn.\; 3)"  height="17"/>
</p>
The overall RNN architecture is given below.
<p align="center">
<img src="jpg/arch.png" width="400" height="350" />
</p>

Since RNN is basically like the conventional feed forward neural comprising of linear and non-linear layers, the back-propagation of loss during training is straight-forward without performing heavy inference computations. The only difference with a normal neural network is that the clubbing of the previous hidden vector and newly generated word is done through the same set of parameters at each step. This is equivalent to feeding the output to the same network as input and hence the name recurrent neural network. This avoids blowing up the size of the network which otherwise would require a new set of parameters at each step. The RNN unit can be represented as shown in the below figure.

<p align="center">
<img src="jpg/compare.png" width="400" height="200" />
</p>

The image is represented as a vector by taking the penultimate layer (before the classification layer) output from any of the standard convolutional networks viz. VGGnet, GoogleNet etc. This produces a feature vector which represents the image in a vector.

For words, we want the representation should be such that the vectors for the words similar in meaning or context should lie close to each other in the vector space. An algorithm that converts a word to a vector is called word2vec and it is arguably the most popular one. word2vec draws inspiration from the famous quote “You shall know the word from the company it keeps” to learn the representation for words. GLoVE is another learning algorithm for obtaining vector representations for words that performs better than word2vec but is more complex.

Given a word “computer”, the task is to predict the context words “learning”, “and” and  “vision” from first sentence and “we”, “love” and “vision” from last sentence. Therefore the training objective becomes maximising the log probability of these context words given the word “computer”. The formulation is:

<!-- <p align="center">
<img src="jpg/obj.png" width="350" height="60" />
</p> -->

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space; Objective = Maximize \sum^T_{t=1}\sum_{-m \geq j \geq m} logP(S_{w_{t+j}}|S_{w_{t}})"  height="47"/>
</p>

where m is the context window size ((max length of caption - t (here 2)) and t runs over the length of the corpus (i.e. every word in the collection of sentences). S<sub>w<sub>n</sub></sub> is the corresponding word vector to S<sub>n</sub>. P(S<sub>w<sub>t+j</sub></sub>|S<sub>w<sub>t</sub></sub>) is modelled by the similarity or inner product between the context word vector and center word vector. For each word there are two vectors associated with it viz. when it appears as context and when it is the center word represented by R and S respectively.

Therefore, 

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space; P(S_{w_{t+j}}|S_{w_{t}}) = \frac{e^{R}_{w_{t+j}}}{\sum^M_{i=1}e^{R^T_iS_{w_t}}} "  height="47"/>
</p>

Here denominator is the normalization term that takes the similarity of center word vector with the context vectors of every other word in vocabulary so that probability sums to one.

### Concept of a Attention Mechanism

When there is clutter in the scene it becomes very difficult for simpler systems to generate a complex description of the image. We, humans, do it by getting a holistic picture of the image at first glance. Our focus then shifts to different regions in the image as we go on describing the image. For machines, a similar attention mechanism has been proposed to mimic human behavior. This attention mechanism allows only important features from an image to come to the forefront when needed. At each step, the salient region of the image is determined and is fed into the RNN instead of using features from the whole image. The system gets a focused view from the image and predicts the word relevant to that region. The region where attention is focused needs to be determined on the basis of previously generated words. Otherwise, newly generated words may be coherent within the region but not in the description being generated.

Until now the output of the **fully connected layer** was used as input to the RNN. This output corresponds to the entire image. For attention we need to get a feature corresponding only to a small subsection of the image. The output of a **convolutional layer** encodes local information and not the information pertaining to the whole cluttered image.

The outputs of the convolutional layer are 2D feature maps where each location was influenced by a small region in the image corresponding to the size (receptive field) of the convolutional kernel.
<p align="center">
<img src="jpg/conv.png" width="240" height="160" />
</p>
Just before the output layer, there is a fully connected layer which is like one stretched vector and represents the whole input image whereas the convolutional layer outputs (all the layers before the fully connected one) are like a 2D image with many dimensions. The vector extracted from a single feature map at a particular location and across all the dimensions signify the feature for a local region of the image.  
<br>
<br>
At each time step, we want to determine the location on the feature map that is relevant to the current step. The feature vector from this location will be fed into the RNN. So we model the probability distribution over locations based on previously generated words. Let L<sub>t</sub> be a random variable with *n* dimensions with each value corresponding to a spatial location on feature map. L<sub>t,i</sub>=1 means the i<sup>th</sup> location is selected for generating the word at the *t* step. Let a<sub>i</sub> be the feature vector at the i<sup>th</sup> location taken from convolutional maps.
<br>
<br>
The value we need is
<p align="center">
<img src="jpg/eq4.png" width="450" height="30" />
</p>
<!-- <p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space; P(L_{t,i}=1|I,S_0,S_1,\ldots , S_{t-1}) \approx P(L+{t,i}|h_t) = \beta_{t,i}\propto a_i^Th_t\;(Eqn.\;4) "  height="47"/>
</p> -->

Here probability of choosing a location (β<sub>t,i</sub>) has been taken as directly proportional to the dot product i.e. similarity between vector at that location and the RNN hidden vector.

Now on the basis of probability distribution, the feature vector corresponding to the location with the maximum probability can be used for making the prediction, but using an aggregated vector from all the location vectors weighted according to the probabilities makes the training converge simply and fast. Therefore we choose to focus/attend to several locations, with the probability p(L<sub>t,i</sub>|h<sub>t</sub>) indicating the importance. This is known as soft attention. In the stochastic mechanism (hard attention), a **single** location is sampled on the basis of probability distribution and only that location is used in the RNN unit.

So let the z<sub>t</sub> be the context or aggregated vector which is to be fed into the RNN.
<p align="center">
<img src="jpg/zt.png" width="120" height="50" />
</p>
So that Equation 2 becomes
<p align="center">
<img src="jpg/becomes.png" width="220" height="30" />
</p>
So this mechanism simulate human behavior by focusing their attention to various parts of the image while describing it and naturally with the focused view, a more sophisticated description can be generated for the image which caters to even the finer level details in the image. Below is an example of the RNN generating words along the corresponding attention.
<br>
<br>
<p align="center">
<img src="jpg/example.png" width="420" height="320" />
</p>

### Show attend and tell in more detail

...


## Results

The model was validated on the standard Flickr8k dataset. State of the art accuracy is achieved. We've included the best performing model (Merge-EfficientNetB7-Glove-RV) from our previous repo. Results are as follows:


| Model                                                         | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---------------------------------------------------------------|--------|--------|--------|--------|
| [ Hard attention (Xu et al.,2016)](https://arxiv.org/pdf/1502.03044.pdf)           |   **67**   |  **45.7**  |  31.4  |  21.3  |
| [Google NIC (Vinyals et al., 2014)](https://arxiv.org/pdf/1411.4555.pdf)           |   63   |   41   |   27   |    -   |
| [Log Bilinear (Kiros et al., 2014)](http://proceedings.mlr.press/v32/kiros14.pdf)  |  65.6  |  42.4  |  27.7  |  17.7  |
| [Merge (Tanti et al., 2017)](https://arxiv.org/pdf/1708.02043.pdf)                 | 60.1   | 41.1   | 27.4   | 17.9   |
| Merge-EfficientNetB7-Glove-RV                                                      | 63.62  | 40.47  | 26.63  | 16.92  |
| Hard attention ResNet                                                              | 66.03  | 45.45  | **31.81**  | **22.14**  |


## Extentions
<!-- - Soft attention implementation (also known as Global attention) -->
- Add attention visualization utility 
- Add config file to remove command line arguments and improve ease of use. 
- Add support for other pre-trained CNN's for feature extraction.