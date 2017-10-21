---
layout: post
title:  "Alchemists had Priors"
date:   2017-10-06 12:00:00 +0800
permalink: /blog/learning_signal
comments: true
categories: jekyll update
---

This post is about alternative learning signals for unsupervised learning. I provide a way of roughly categorizing
unsupervised learning methods, discuss why the mainstream explanatory approach is unsatisfactory, and list a few
desirable properties that we should look for in designing unsupervised learning signals.


# 3 types of learning signal

I divide unsupervised models into 3 categories: explanatory, predictive, and discriminative. 

- __Explanatory__ models explain observation using some **posterior**. For example, VAEs map input to a posterior, then use
samples from that posterior to reconstruct/explain the input. Boltzmann Machines, Topic Models, Gaussian Mixture Models
are all examples of this type.
- __Predictive__ models provide the **likelihood** of unobserved data given observed data. Most of the autoregressive models, such
as LSTM language models, NADE [1], PixelRNN [2], WaveNet [3], and Context Encoder [4], are examples of this category.
- __Discriminative__ models tell real data from fake data by evaluating data's **unnormalized marginal probability**.
They are trained using domain-specific self-supervisory signals.  Examples include GANs, word embedding, Context
Prediction [5], and Triplet Siamese Network for video patches [7].


# Explanatory approach isn't enough

The state of affairs is such that apart from the explanatory approaches, the other 2 approaches are all making very good
progress. PixelRNNs generate pretty sharp images. WaveNet generates pretty convincing speech and music, Context
Prediction learns features that allow very effective semantic image retrieval, word embedding has always worked wonder,
and GANs give very convincing images on LSUN, which is by no means a simple dataset. A few things are still missing here
and there, but the general trend seems pretty promising.

Explanatory approaches, on the other hand, have been quite stuck after VAEs came out. Nobody has gotten it to work well
even on moderately complex datasets such as CIFAR and LSUN (the best so far seems to be conv-DRAW [6], if we do not
count GAN's derivatives such as ALI). What's wrong?

There might be many reasons why VAEs don't work yet, but in this post I'll focus on just one: *you shouldn't just ask a
model to explain what it has observed*.


# Don't just explain

The difference between science and pseudo-science is that science makes testable predictions, whereas pseudo-science
does not. Explanatory models are a bit like pseudo-science: the model might seem fairly reasonable, and even convincing
at times, but since it never makes a prediction about anything, you can't really tell if it's wrong. If it overfits,
there's nothing you can do about it. Or, rather, there are so many things you can do about it, you'll never find out
which one actually is right. This is the reason that motivates this post: there are so many bloody priors we can choose
to limit overfitting, which one is right? I'm bloody tired of looking for priors. I'm pretty sure alchemists had their
priors too.

In comparison, the other approaches have to do more than "just explain". The predictive approach has to predict unseen
data, which naturally limits overfitting. Discriminative models are also less likely to overfit, if they are trained
using very convincing fake samples (see [here](/blob/wgan_energy) for an analogy).

# So what?

Ideally, we'd like our unsupervised models to be explanatory, predictive, and discriminative at the same time. If our
model cannot explain its observation, it probably doesn't really understand the data. If it cannot make good
predictions, it probably overfits. If it cannot discriminate real samples from fake ones, it probably overfits as well.

But how do we do that? Well, I'll leave this part open :)



# References:

[1] Benigno Uria, Marc-Alexandre Côté, Karol Gregor, Iain Murray, Hugo Larochelle, _Neural Autoregressive Distribution Estimation_

[2] Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu, _Pixel Recurrent Neural Networks_

[3] TOO MANY NAMES, _WaveNet: A Generative Model For Raw Audio_

[4] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros, _Context Encoders: Feature Learning by Inpainting_

[5] Carl Doersch, Abhinav Gupta, Alexei A. Efros, _Unsupervised Visual Representation Learning by Context Prediction_

[6] Karol Gregor, Frederic Besse, Danilo Rezende, Ivo Danihelka, Daan Wiestra, _Towards Conceptual Compression_

[7] Xiaolong Wang, Abhinav Gupta, _Unsupervised Learning of Visual Representation using Videos_

