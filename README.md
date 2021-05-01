# Attention-Modeling for image captioning 


## Implementation

Implementation of Hard Attention for image captioning described in [Show, Attend and Tell](https://arxiv.org/abs/1502.03044). For our implementation we chose to make use of ResNet instead of the proposed VGG-16 model in the encoder stage. 

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
- Soft attention implementation (also known as Global attention )
- Add attention visualization utility 
