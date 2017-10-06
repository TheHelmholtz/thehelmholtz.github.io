---
layout: post
title:  "Unsupervised Learning Signal"
date:   2017-10-06 12:00:00 +0800
permalink: /blog/learning_signal
comments: true
categories: jekyll update
---

I think the autoencoder paradigm is flawed. This post outlines a way of categorizing unsupervised objectives, and
explains why I think autoencoders, or the explanatory approach, might be insufficient.

> TLDR: You shouldn't just ask a model to explain what it has already observed. Because the model can overfit, and
> there's nothing you can do about it.

I divide unsupervised models into 3 categories: explanatory, predictive, and discriminative. 

- __Explanatory__ models explain observation using some posterior. For example, VAEs map input to a posterior, then use
samples from that posterior to reconstruct/explain the input. Boltzmann Machines, Topic Models, Gaussian Mixture Models
are all examples of this type.
- __Predictive__ models, given parts of the input, predict the remaining parts. Most of the autoregressive models, such
as LSTM language models, NADE [1], PixelRNN [2], WaveNet [3], and Context Encoder [4], are all examples of this category.
- __Discriminative__ models map input to some kind of semantic information, using self-supervisory signals. GAN is
probably the most famous example in this category. Other examples include Context Prediction [5] and word embedding.

The state of affairs is such that both predictive and discriminative methods are making very good progress. PixelRNNs
generates pretty sharp images. WaveNet generates pretty convincing speech and music, GANs give very convincing images on
LSUN, which is by no means a simple dataset, and Context Prediction learns features that allow very effective semantic
image retrieval. A few things are missing here and there, but the general trend seems pretty good.

Explanatory approaches, on the other hand, have been quite stuck after VAEs came out. Nobody has gotten it to work well
even on moderately complex datasets such as CIFAR and LSUN. What's wrong?

There might be many reasons why VAEs don't work yet, but in this post I'll focus on just one: *you shouldn't just ask a
model to explain what it has observed*.

The difference between science and pseudo-science is that science makes testable predictions, whereas pseudo-science
does not. Explanatory models are a bit like pseudo-science: the model might seem fairly reasonable, and even convincing
at times, but since it never makes a prediction about anything, you can't really tell if it's wrong. If it overfits,
there's nothing you can do about it.

In comparison, methods under the predictive approach and the discriminative approach have to do more than "just
explain". The predictive approach has to predict unseen data, which naturally limits overfitting. Discriminative models
make prediction as well, just that the prediction isn't in the same domain as the input.

If prediction is indeed so important, can we incorporate prediction into explanatory models?


References:

[1] Benigno Uria, Marc-Alexandre Côté, Karol Gregor, Iain Murray, Hugo Larochelle, Neural Autoregressive Distribution Estimation

[2] Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu, Pixel Recurrent Neural Networks

[3] WaveNet: A Generative Model For Raw Audio

[4] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros, Context Encoders: Feature Learning by Inpainting

[5] Carl Doersch, Abhinav Gupta, Alexei A. Efros, Unsupervised Visual Representation Learning by Context Prediction

