---
layout: page
title: Research Interests
footer: true
permalink: /direction/
---

## Current topics

#### Self-supervised Data Modelling

In traditional autoencoders and variants, we map input to latent representation, then use that latent representation to
reconstruct the input. This approach is what I call *asking a model to explain what it has already seen*. I now think
this approach is terribly flawed, because it lacks *validation signal*. How do we know our model is not overfitting?

There are two possible ways to provide validation signal *during* training: only feed the encoder with part of the input
and ask it to predict the remaining parts, and using a discriminator to judge the quality of the generative model. I
think these two approaches are symmetric, and can be synthesized into a single coherent probabilistic framework of
learning.


#### Interpretable Models

Our current prescribed probabilistic models have two obvious problems: they are not hierarchical, and they contain way
too many factors to be interpretable. I'm working on models that use a much much smaller number of factors in the latent
representation, but each factor has a significant number of categories. There is a set of very elegant assumptions that
underlie this kind of model, which makes them easily interpretable. But right now this model doesn't work, yet.

#### Approximate + Exact Inference

Neural nets are now quite good at producing *intuition*. I believe the next step is to combine this powerful intuition
with the accuracy of exact inference methods. Right now I'm working on combining MCMC with variational methods.

## Forward-looking topics

#### Discrete Entity Modelling

The world we observe usually consists of a large set of discrete objects, yet the representation we use in our NNs are
primarily unimodal. To model the discrete structure of our world, we must move towards representations that are
multimodal, where each mode may represent a separate object.


#### Attention and Memory

Inference with multimodal distributions can be tricky. I believe attention, implemented as a kind of noisy gradient
descent (MCMC with Langevin dynamics), has the potential to handle multimodal posteriors. In addition, through the act
of binding, attention allows very complex inference to be performed with a very limited set of computational resources
(sequential execution).

Memory, on the other hand, is something we need if we want learning to be efficient. Our current models work reasonablly
well at capturing population-level statistics. However, retaining instance-level information is an ability that is still
lacking. It is widely understood that memory can be modelled using a multimodal distribution, every mode of which is a
memory segment. For this reason, memory is closely tied to attention, because memory recall is essentially an inference
process on a multimodal distribution.


#### Binding

Binding is the unconscious act of associating signals from multiple sensory modality and across time to the same
underlying "object identity". The binding problem is interesting to me because it involves the modelling of multiple
entities in the same scene, and is an excellent testing ground for discrete entity modelling methods. Also, by combining
signals from multiple sensory modalities and across time, we can make the recognition of "object identity" much more
robust to variations in specific modalities.

