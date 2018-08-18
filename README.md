# Text classification with Convolution Neural Networks (CNN)
The approach followed is quit similar to the one of Denny and the original paper of Yoon Kim [1]. You can find the implementation of Yoon Kim on [GitHub](https://github.com/yoonkim/CNN_sentence) as well.


## Model
The implemented [model](cnn_model.py) has multiple convolutional layers in parallel to obtain several features of one text. Through different kernel sizes of each convolution layer the window size varies and the text will be read with a n-gram approach. The default values are 3 convolution layers with kernel size of 3, 4 and 5.<br>

## Results
For all runs I used a learning rate reduction if their's no improvement on validation loss by factor 0.1 after 4 epochs. The optimizer for all runs was Adadelta.<br>As already described I used 5 runs to get a final mean of loss / accuracy.
 can get better results by tuning some parameters:
- Increase feature maps
- Add / remove filter sizes
- Use another embeddings (e.g. fast text)
- Increase / decrease maximum words in vocabulary and sequence


## Requirements
* Python 3.6
* Keras 2.0.8
* TensorFlow 1.1
* Scikit 0.19.1

## References
[1] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)<br>

## Author
Yash Gunjal
