
# Ideas

While contributing to this repo, it is hard not to come up with many ideas. This markdown file
serves as a list of ideas.

## Promising

* Intuitively, determining phoneme corresponds to multiple graphemes. Introducing
multi-headed attention to the spectrogram model adds in that appropriate structural prior.
  * [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation](https://arxiv.org/abs/1804.09849v2)
  * [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  * [Improving Language Understanding with Unsupervised Learning](https://blog.openai.com/language-unsupervised/)
  * [A Neural Network Model That Can Reason - Prof. Christopher Manning](https://www.youtube.com/watch?time_continue=451&v=24AX4qJ7Tts)
* Typically, Batch Norm followed by ReLU tends to perform better.
  * [Batch Normalization before or after ReLU?](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/)
* The Spectrogram model does not apply techniques that have produced state-of-the-art results
in language modeling. Try using techniques such as Embedding Dropout, BPTT, Weight Drop, Layer norm, ELMo,
and Embeddings.
  * [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation](https://arxiv.org/abs/1804.09849v2)
  * [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf)
  * [LSTM and QRNN Language Model Toolkit](https://github.com/salesforce/awd-lstm-lm)
  * [Notes on state of the art techniques for language modeling](http://www.fast.ai/2017/08/25/language-modeling-sota/)
  * [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
  * [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
* June 2018 paper from the Tacotron 2 team describes using a combination of L1 and L2 loss to deal with
noisy audio data, try this.
  * [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558)
* A DeepMind paper introduced WaveRNN, a simpler model able to achieve similar fidelity audio to WaveNet.
WaveRNN does not require specialized kernels to achieve realtime performance.
  * [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435)
* Igor Babuschkin's WaveNet repository and the WaveNet paper suggest using a casual convolution
with an initial filter width ranging between 2 to 32. Try using NV-WaveNet's support for a filter
width of 2.
  * [Igor Babuschkin, a WaveNet author, GitHub issue](https://github.com/ibab/tensorflow-wavenet/issues/83#issuecomment-249399098)
  * [WAVENET: A GENERATIVE MODEL FOR RAW AUDIO](https://arxiv.org/pdf/1609.03499.pdf)

## Interesting

* ELMo learns a linear combination of the vectors stacked above each input word for each end task, which
markedly improves performance over just using the top LSTM layer. Similarly, it'd be interesting to try
to learn a linear combination for WaveNet layers.
  * [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
* Setting the right learning rate and optimizer can be critical to performance. It'd be
interesting to look at different optimizers and schedulers.
  * [[P] Beating Adam optimizer with Stochastic Gradient Descent](https://www.reddit.com/r/MachineLearning/comments/8os4fl/p_beating_adam_optimizer_with_stochastic_gradient/)
  * [Estimating an Optimal Learning Rate For a Deep Neural Network](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)
  * [Improving the way we work with learning rate.](https://techburst.io/improving-the-way-we-work-with-learning-rate-5e99554f163b)
  * [Setting the learning rate of your neural network.](https://www.jeremyjordan.me/nn-learning-rate/)
  * [Igor Babuschkin, a WaveNet author, GitHub issue](https://github.com/ibab/tensorflow-wavenet/issues/143#issuecomment-253400332)
* The Linda Johnson, in the LJ Speech dataset, tends to speak differently for different chapters.
At the moment, there is little to no signal for which book or chapter the current textual piece lies;
therefore, it'd be interesting to try using a global chapter or book embedding.\
  * [Linda Johnson Speech Dataset Book Embedding](https://github.com/keithito/keithito.github.io/issues/1)
* Google TFGAN demonstrated that a GAN architecture applied to Tacotron to produce realistic spectrograms.
Training a GAN, NVIDIA demonstrated progressive growing achieved state-of-the-art results. Spectrograms
are scaled by frame width and frame hop; therefore, it'd be interesting to apply several GAN techniques to
generate realistic spectrograms.
  * [Progressive Growing of GANs for Improved Quality, Stability, and Variation](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)
  * [TFGAN: A Lightweight Library for Generative Adversarial Networks](https://ai.googleblog.com/2017/12/tfgan-lightweight-library-for.html)
* Typically, it's common practice to normalize neural network features to 0 mean and 1 standard
deviation. Although the Tacotron 2 did not try this, it'd be interesting to see if this improved
convergence.
  * [Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between](http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
  * [Setting up the data and the model](http://cs231n.github.io/neural-networks-2/)
  * [Normalized Spectrogram](https://www.google.com/search?q=mean+normalization+spectrogram&oq=mean+normalization+spectrogram&aqs=chrome..69i57.4190j0j1&sourceid=chrome&ie=UTF-8)
* For speaker's, like Linda Johnson, that have a higher frequency then 7.6 kHz, it might be useful
to use a larger mel-spectrogram with more channels. Naively changing only the mel-filter bands,
would force the model to work with a more compact representation.
* Igor Babuschkin, a WaveNet author, suggests to use TanH instead of RELu in the WaveNet model.
  * [Igor Babuschkin, a WaveNet author, GitHub issue](https://github.com/ibab/tensorflow-wavenet/issues/143#issuecomment-252895937)

## Nice to Try

* Attention is critical to the success of the spectrogram model; therefore, to optimize for a
sharper attention, consider adding an attention loss.
  * [A Regularized Framework for Sparse and Structured Neural Attention](https://arxiv.org/pdf/1705.07704.pdf)
* Scheduled sampling has been proposed as a technique to help recurrent models recover from errors. Naive
implementations of scheduled sampling require unraveling recurrent operations, slowing down training.
Another possible implementation is to feedback a batch of predictions following teacher-forcing randomly.
  * [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099)
  * [A Word of Caution on Scheduled Sampling for Training RNNs](http://www.inference.vc/scheduled-sampling-for-rnns-scoring-rule-interpretation/)
  * [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
* End-to-end neural models tend to work better. There is no fundamental reason that the signal
and feature model cannot combined in an end-to-end differentiable model. It'd be interesting
to try combining both the feature and signal model into an end-to-end differentiable model.
