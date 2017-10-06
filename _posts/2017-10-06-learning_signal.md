---
layout: post
title:  "Learning Signal"
date:   2017-10-06 12:00:00 +0800
permalink: /blog/learning_signal
comments: true
categories: jekyll update
---

This post is about what kind of objective function unsupervised learning should use. In particular, I'll discuss what I
call explanatory, predictive and discriminative objectives. It's a stub I hope I'll extend later.

Let's clear the basic definitions:

- __Explanatory__: Given input, compute posterior, explain input. Examples include VAEs and BMs, and many of the more
traditional probabilistic models
- __Predictive__: Given parts of the input, predict other parts of the input. Examples include many of the autoregressive
models such as NADE [1], PixelRNN [2] and WaveNet [3]. Another example is Context Encoder [4].
- __Discriminative__: Given input, compute a single or several energy values. Examples include GANs, Context Prediction
[5] and word embedding.

The state of affairs is such that both predictive and discriminative methods are making very good progress. The
explanatory approaches, however, still do not scale to bigger datasets. Why is that so?

To answer this question, I'd like to take a step back and ask a very different question: what's the difference between
science and pseudo-science?

Both science and pseudo-science explain what we observe. Very often, though not always, the explanations are backed up
using some sort of model. But the real difference between science and pseudo-science is their predictive power.
Scientific models make testable predictions, whereas pseudo-scientific models do not.

The point I'm making is that, it's not good enough to come up with sensible model that explains your observation. How
would we know the model is not overfitting? Instead, we must train the model to generalize. The predictive approach
requires exactly that, and the discriminative approaches enforce that to some extent as well.

Of course, the problem with predictive and discriminative methods is that they are total black-boxes. With the
explanatory approach we can carefully design our models, then do inference with those models. With the predictive and
discriminative approaches, the idea of a prior and a posterior seems less relevant, and so the structure of the model is
assumed to be implicit.

If only there is a way to combine them with inference using a prescribed model.


References:

[1] Benigno Uria, Marc-Alexandre Côté, Karol Gregor, Iain Murray, Hugo Larochelle, Neural Autoregressive Distribution Estimation

[2] Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu, Pixel Recurrent Neural Networks

[3] WaveNet: A Generative Model For Raw Audio

[4] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros, Context Encoders: Feature Learning by Inpainting

[5] Carl Doersch, Abhinav Gupta, Alexei A. Efros, Unsupervised Visual Representation Learning by Context Prediction

